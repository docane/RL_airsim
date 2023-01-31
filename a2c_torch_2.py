import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return x


class Actor(nn.Module):
    def __init__(self, action_size):
        super(Actor, self).__init__()
        self.dense = nn.Linear(3136, 512)
        self.mu = nn.Linear(512, action_size)
        self.sigma = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.dense(x))
        mu = self.mu(x)
        sigma = F.sigmoid(self.sigma(x))
        return mu, sigma


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense = nn.Linear(3136, 512)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.dense(x))
        value = self.value(x)
        return value


class A2Cagent:
    def __init__(self, action_size, max_action):
        self.action_size = action_size
        self.max_action = max_action

        self.discount_factor = 0.99
        self.extractor_lr = 0.001
        self.actor_lr = 0.0001
        self.critic_lr = 0.001

        self.extractor = Extractor()
        self.actor = Actor(action_size)
        self.critic = Critic()

        self.extractor_optim = torch.optim.Adam(self.extractor.parameters(), lr=self.extractor_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.actor.parameters(), lr=self.critic_lr)

    def get_action(self, state):
        mu, sigma = self.actor(state)
        dist = torch.distributions.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = torch.clip(action, -self.max_action, self.max_action)
        return action.item()

    def actor_learn(self, state, action, reward, next_state, done):
        action = torch.tensor(action)
        mu, sigma = self.actor(state)
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + (1 - done) * self.discount_factor * next_value[0]
        advantage = target - value[0]
        dist = torch.distributions.Normal(loc=mu[0], scale=sigma[0])
        action_prob = dist.log_prob(action)
        cross_entropy = -torch.log(action_prob + 1e-5)
        loss = torch.mean(cross_entropy * advantage)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, state, reward, next_state, done):
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + (1 - done) * self.discount_factor * next_value[0]
        loss = F.mse_loss(target, value[0])
        torch.mean(loss)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def train_model(self, state, action, reward, next_state, done):
        action = torch.tensor(action)
        mu, sigma = self.actor(state)
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + (1 - done) * self.discount_factor * next_value[0]
        with torch.no_grad():
            advantage = target - value[0]
        dist = torch.distributions.Normal(loc=mu[0], scale=sigma[0])
        action_prob = dist.log_prob(action)
        cross_entropy = -torch.log(action_prob + 1e-5)
        actor_loss = torch.mean(cross_entropy * advantage)
        with torch.no_grad():
            critic_loss = F.mse_loss(target, value[0])
        critic_loss = torch.mean(critic_loss)
        F.mse_loss(target, value[0])
        loss = 0.1 * actor_loss + critic_loss
        torch.tensor(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
