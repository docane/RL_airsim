import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class A2C(nn.Module):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1024, 512)
        self.mu = nn.Linear(512, action_size)
        self.sigma = nn.Linear(512, action_size)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        mu = self.mu(x)
        sigma = F.sigmoid(self.sigma(x))
        value = self.value(x)
        return mu, sigma, value


class A2Cagent:
    def __init__(self, action_size, max_action):
        self.action_size = action_size
        self.max_action = max_action

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = A2C(self.action_size).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        mu = mu.item()
        sigma = sigma.item()
        sigma += 1e-5
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        action = dist.sample([1])[0]
        action = torch.clip(action, -self.max_action, self.max_action)
        return action.item()

    def train_model(self, state, action, reward, next_state, done):
        action = torch.tensor(action)
        mu, sigma, value = self.model(state)
        _, _, next_value = self.model(next_state)
        target = reward + (1 - done) * self.discount_factor * next_value[0]
        with torch.no_grad():
            advantage = target - value[0]
        dist = torch.distributions.Normal(loc=mu[0], scale=sigma[0])
        action_prob = dist.log_prob(action)[0]
        cross_entropy = -torch.log(action_prob + 1e-5)
        actor_loss = torch.mean(cross_entropy * advantage)
        with torch.no_grad():
            critic_loss = 0.5 * torch.square(target - value[0])
        critic_loss = torch.mean(critic_loss)
        loss = 0.1 * actor_loss + critic_loss
        torch.tensor(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
