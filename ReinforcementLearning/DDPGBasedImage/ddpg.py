import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Lambda, concatenate, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from replaybuffer import ReplayBuffer


# class DDPG(Model):
#     def __init__(self, action_size):
#         super(DDPG, self).__init__()
#         self.action_size = action_size
#
#         self.conv1 = Conv2D(32, 8, strides=4, activation='relu')
#         self.conv2 = Conv2D(64, 4, strides=2, activation='relu')
#         self.conv3 = Conv2D(64, 3, strides=1, activation='relu')
#         self.flatten = Flatten()
#
#         self.actor_fc1 = Dense(2048, activation='relu')
#         self.actor_fc2 = Dense(512, activation='relu')
#         self.action = Dense(self.action_size)
#
#         self.critic_fc1 = Dense(2048, activation='relu')
#         self.critic_fc_a = Dense(2048, activation='relu')
#         self.critic_fc2 = Dense(512, activation='relu')
#         self.value = Dense(1)
#
#     def call(self, state_action):
#         state = state_action[0]
#         action = state_action[1]
#
#         x = self.conv1(state)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#
#         actor_x = self.actor_fc1(x)
#         actor_x = self.actor_fc2(actor_x)
#         a = self.action(actor_x)
#
#         critic_x = self.critic_fc1(x)
#         critic_a = self.critic_fc_a(action)
#         critic_h = concatenate([critic_x, critic_a], axis=-1)
#         critic_h = self.critic_fc2(critic_h)
#         v = self.value(critic_h)
#
#         return a, v


class Extractor(Model):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = Flatten()

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x


class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound
        self.action_size = action_dim
        self.dense1 = Dense(2048, activation='relu')
        self.dense2 = Dense(512, activation='relu')
        self.action = Dense(self.action_size, activation='sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        action = self.action(x)

        action = Lambda(lambda x: self.action_bound)(action)

        return action


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.x1 = Dense(2048, activation='relu')
        self.a1 = Dense(2048, activation='relu')
        self.h1 = Dense(512, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]
        x = self.x1(state)
        a = self.a1(action)
        h = concatenate([x, a], axis=-1)
        x = self.h1(h)
        q = self.q(x)
        return q


class DDPGagent(object):
    def __init__(self, env):
        self.gamma = 0.95
        self.batch_size = 128
        self.buffer_size = 20000
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001
        self.tau = 0.001

        self.env = env
        self.state_dim = (84, 84, 3)
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        # self.model = DDPG(self.action_dim)
        # self.target_model = DDPG(self.action_dim)
        #
        # state_in = Input((self.state_dim,))
        # action_in = Input((self.action_dim,))
        # self.model([state_in, action_in])
        # self.model.build(input_shape=(None, self.state_dim))
        # self.target_model.build(input_shape=(None, self.state_dim))

        self.extractor = Extractor()

        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)

        self.critic = Critic()
        self.target_critic = Critic()

        self.extractor.build(input_shape=(None, 84, 84, 3))
        # self.actor.build(input_shape=(None, 7, 7, 64))
        # self.target_actor.build()

        state_in = Input((64 * 7 * 7,))
        action_in = Input((64 * 7 * 7,))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        self.actor_opt = Adam(self.actor_learning_rate)
        self.critic_opt = Adam(self.critic_learning_rate)

        self.buffer = ReplayBuffer(self.buffer_size)

        self.save_episode_rewward = []

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

    # def learn(self, states, actions, td_targets):
    #     with tf.GradientTape() as tape:
    #         states = self.extractor(states)
    #         actions = self.actor(states, training=True)
    #         q = self.critic([states, actions], training=True)
    #         critic_q = self.critic([states, actions])
    #         actor_loss = -tf.reduce_mean(critic_q)
    #         critic_loss = tf.reduce_mean(tf.square(q - td_targets))
    #
    #     actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    #     critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
    #
    #     self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    #     self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            states = self.extractor(states)
            q = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(q - td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            states = self.extractor(states)
            actions = self.actor(states, training=True)
            critic_q = self.critic([states, actions])
            loss = -tf.reduce_mean(critic_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)

    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] * self.gamma * q_values
        return y_k

    def load_weights(self, path):
        self.actor.load_weights(path + 'airsim_actor.h5')
        self.critic.load_weights(path + 'airsim_critic.h5')

    def train(self, max_episode_max):
        self.update_target_network(1.0)

        for ep in range(int(max_episode_max)):
            pre_noise = np.zeros(self.action_dim)
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                state = self.extractor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                print(action)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add_buffer(state, action, reward, next_state, done)

                if self.buffer.buffer_count() > 1000:
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)
                    target_qs = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32),
                                                    self.target_actor(
                                                        tf.convert_to_tensor(next_states, dtype=tf.float32))])

                    y_i = self.td_target(rewards, target_qs.numpy(), dones)
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    self.update_target_network(self.tau)

                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward)
            self.save_episode_rewward.append(episode_reward)
