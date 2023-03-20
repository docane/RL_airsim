import datetime as dt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from ddpg_based_image.replaybuffer import ReplayBuffer


class Actor(Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()

        self.action_size = action_dim

        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.action = Dense(self.action_size, activation='tanh')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        action = self.action(x)

        return action


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.x1 = Dense(64, activation='relu')
        self.a1 = Dense(64, activation='relu')
        self.h2 = Dense(64, activation='relu')
        self.h3 = Dense(32, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]

        x = self.x1(state)
        a = self.a1(action)
        h = concatenate([x, a], axis=-1)
        x = self.h2(h)
        x = self.h3(x)
        q = self.q(x)
        return q


class DDPGagent(object):
    def __init__(self, env):
        self.gamma = 0.99
        self.batch_size = 512
        self.buffer_size = 20000
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001
        self.tau = 0.001

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.action_dim)
        self.target_actor = Actor(self.action_dim)

        self.actor.build(input_shape=(None, self.state_dim))
        self.target_actor.build(input_shape=(None, self.state_dim))

        self.critic = Critic()
        self.target_critic = Critic()

        state_in = Input((self.state_dim,))
        action_in = Input((self.action_dim,))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        self.actor_opt = Adam(self.actor_learning_rate)
        self.critic_opt = Adam(self.critic_learning_rate)

        self.buffer = ReplayBuffer(self.buffer_size)

        self.save_episode_reward = []

        self.writer = tf.summary.create_file_writer(
            f'summary/airsim_ddpg_{dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')

    def update_target_network(self, tau):
        theta = self.actor.get_weights()
        target_theta = self.target_actor.get_weights()
        for i in range(len(theta)):
            target_theta[i] = tau * theta[i] + (1 - tau) * target_theta[i]
        self.target_actor.set_weights(target_theta)

        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = tau * phi[i] + (1 - tau) * target_phi[i]
        self.target_critic.set_weights(target_phi)

    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(q - td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss

    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic_q = self.critic([states, actions])
            loss = -tf.reduce_mean(critic_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))
        return loss

    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)

    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] * self.gamma * q_values[i]
        return y_k

    def load_weights(self, path):
        self.actor.load_weights(path + 'airsim_ddpg_actor.tf')
        self.critic.load_weights(path + 'airsim_ddpg_critic.tf')

    def save_weights(self):
        self.actor.save_weights(f'./airsim_ddpg_actor.tf')
        self.critic.save_weights(f'./airsim_ddpg_critic.tf')

    def get_action(self, state):
        action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
        action = action.numpy()[0]
        return action

    def draw_tensorboard(self, score, e):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=e)

    def train(self, max_episode_max):
        self.update_target_network(1.0)
        total_time = 0
        for ep in range(int(max_episode_max)):
            pre_noise = np.zeros(self.action_dim)
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                action = self.get_action(state)
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add_buffer(state, action, reward, next_state, done)

                if self.buffer.buffer_count() >= 1000:
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)
                    target_qs = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32),
                                                    self.target_actor(
                                                        tf.convert_to_tensor(next_states, dtype=tf.float32))])
                    y_i = self.td_target(rewards, target_qs.numpy(), dones)
                    actions = np.reshape(actions, (-1, 1))
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    self.update_target_network(self.tau)

                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            total_time += time
            print('Episode:', ep + 1, 'Total Time:', total_time, 'Reward:', round(episode_reward, 2))
            self.save_episode_reward.append(episode_reward)
            self.draw_tensorboard(episode_reward, ep)
            self.save_weights()
