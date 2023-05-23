import time
import datetime as dt
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
from keras.initializers.initializers_v1 import RandomUniform
import tensorflow as tf
from replaybuffer import ReplayBuffer
import os
import gym

gym.envs.register(id='car_env-v0', entry_point='car_env_state_15:AirSimCarEnv')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


class Actor(Model):
    def __init__(self):
        super(Actor, self).__init__()

        self.dense1 = Dense(120, activation='relu')
        self.dense2 = Dense(240, activation='relu')
        self.steering = Dense(1, activation='tanh', kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.throttle = Dense(1, activation='sigmoid', kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        steering = self.steering(x)
        throttle = self.throttle(x)
        h = concatenate([steering, throttle], axis=-1)

        return h


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.x1 = Dense(120, activation='relu')
        self.x2 = Dense(240, activation='relu')
        self.a1 = Dense(240, activation='relu')
        self.h1 = Dense(240, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]

        x = self.x1(state)
        x = self.x2(x)
        a = self.a1(action)
        h = concatenate([x, a], axis=-1)
        x = self.h1(h)
        q = self.q(x)
        return q


class DDPGagent(object):
    def __init__(self, env):
        self.gamma = 0.99
        self.batch_size = 64
        self.buffer_size = 1000
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001
        self.tau = 0.001

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound_high = env.action_space.high
        self.action_bound_low = env.action_space.low

        self.actor = Actor()
        self.target_actor = Actor()

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

        self.actor.summary()
        self.critic.summary()

        self.buffer = ReplayBuffer(self.buffer_size)

        self.save_episode_reward = []

        self.writer = tf.summary.create_file_writer(
            f'summary/airsim_ddpg_{dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        self.save_model_dir = f'./models/airsim_ddpg_model_{dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}/'

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

    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] * self.gamma * q_values[i]
        return y_k

    @staticmethod
    def ou_noise(x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)

    def load_weights(self, path):
        self.actor.load_weights(path + 'airsim_ddpg_actor.h5')
        self.critic.load_weights(path + 'airsim_ddpg_critic.h5')

    def save_weights(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.actor.save_weights(path + './airsim_ddpg_actor.h5')
        self.critic.save_weights(path + './airsim_ddpg_critic.h5')

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
            step, episode_reward, done = 0, 0, False
            state = self.env.reset()
            actor_loss, critic_loss = 0, 0
            time.sleep(2)
            # time.sleep(0.2)
            while not done:
                action = self.get_action(state)
                # print('State:', state)
                # print('Action:', action)
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                action = np.clip(action + noise, self.action_bound_low, self.action_bound_high)
                next_state, reward, done, _ = self.env.step(action)
                # print(reward)
                self.buffer.add_buffer(state, action, reward, next_state, done)

                if self.buffer.buffer_count() > 500:
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)
                    target_qs = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32),
                                                    self.target_actor(
                                                        tf.convert_to_tensor(next_states, dtype=tf.float32))])
                    y_i = self.td_target(rewards, target_qs.numpy(), dones)

                    with tf.GradientTape() as tape_c:
                        q = self.critic([states, actions], training=True)
                        critic_loss = tf.reduce_mean(tf.square(q - y_i))
                    critic_grads = tape_c.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                    with tf.GradientTape() as tape_a:
                        actions = self.actor(states, training=True)
                        critic_q = self.critic([states, actions])
                        actor_loss = -tf.reduce_mean(critic_q)
                    actor_grads = tape_a.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                    self.update_target_network(self.tau)
                    # print(f'Actor Loss: {actor_loss}', f'Critic Loss: {critic_loss}')

                pre_noise = noise
                state = next_state
                episode_reward += reward
                step += 1

            total_time += step
            log = f'Episode: {ep + 1}'
            log += f' Total Time: {total_time}'
            log += f' Reward: {round(episode_reward, 2)}'
            log += f' Actor Loss: {actor_loss}'
            log += f' Critic Loss: {critic_loss}'
            print(log)
            self.save_episode_reward.append(episode_reward)
            # self.draw_tensorboard(episode_reward, ep)
            # self.save_weights(self.save_model_dir)

            # if ep % 100 == 99:
            #     del self.env
            #     os.system('taskkill /im Coastline.exe /t /f')
            #     time.sleep(3)
            #     os.system('start /d "C:\\Coastline (2)\\Coastline\\WindowsNoEditor" run.bat')
            #     time.sleep(5)
            #     self.env = gym.make('car_env-v0', ip_address='127.0.0.1')
