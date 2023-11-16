import datetime as dt
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
import tensorflow as tf
from replaybuffer import ReplayBuffer
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


class Actor(Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()

        self.action_size = action_dim

        self.dense1 = Dense(300, activation='relu')
        self.dense2 = Dense(600, activation='relu')
        self.steering = Dense(1, activation='tanh')
        self.throttle = Dense(1, activation='sigmoid')
        self.brake = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        steering = self.steering(x)
        throttle = self.throttle(x)
        brake = self.brake(x)
        action = concatenate([steering, throttle, brake], axis=-1)

        return action


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.x1 = Dense(300, activation='relu')
        self.x2 = Dense(600, activation='relu')
        self.a1 = Dense(600, activation='relu')
        self.h1 = Dense(600, activation='relu')
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
        self.buffer_size = 100000
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001
        self.tau = 0.001

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound_high = env.action_space.high
        self.action_bound_low = env.action_space.low

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

        # self.actor.summary()
        # self.critic.summary()

        self.buffer = ReplayBuffer(self.buffer_size)

        self.now = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.writer = None
        self.save_model_dir = f'./models/airsim_ddpg_model_{self.now}/'

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

    # TD 타겟 계산
    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.gamma * q_values[i]
        return y_k

    # 크리틱 신경망 업데이트
    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(q - td_targets))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        return critic_loss

    # 액터 신경망 업데이트
    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic_q = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(critic_q)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss

    def load_weights(self, path):
        self.actor.load_weights(path + 'airsim_ddpg_actor.h5')
        self.critic.load_weights(path + 'airsim_ddpg_critic.h5')
        self.target_actor.load_weights(path + 'airsim_ddpg_target_actor.h5')
        self.target_critic.load_weights(path + 'airsim_ddpg_target_critic.h5')

    def save_weights(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.actor.save_weights(path + 'airsim_ddpg_actor.h5')
        self.critic.save_weights(path + 'airsim_ddpg_critic.h5')
        self.target_actor.save_weights(path + 'airsim_ddpg_target_actor.h5')
        self.target_critic.save_weights(path + 'airsim_ddpg_target_critic.h5')

    def get_action(self, state):
        action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
        action = action.numpy()[0]
        return action

    def draw_tensorboard(self, episode, score, avg_step, mean_distance_per_step,
                         mean_distance_to_road_center, actor_loss, critic_loss):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Step/Episode', avg_step, step=episode)
            tf.summary.scalar('Mean Moving Distance Per Step/Episode', mean_distance_per_step, step=episode)
            tf.summary.scalar('Mean Distance to Road Center/Episode', mean_distance_to_road_center, step=episode)
            tf.summary.scalar('Actor Loss/Episode', actor_loss, step=episode)
            tf.summary.scalar('Critic Loss/Episode', critic_loss, step=episode)

    def evaluation(self, episode):
        eval_index = [0, 370, 770]
        eval_step = 0
        evaluation_rewards = []
        distances_to_road_center = []
        for i in eval_index:
            episode_reward = 0
            done = False
            state = self.env.reset(i)
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                distances_to_road_center.append(info['track_distance'][0])
                eval_step += 1
            evaluation_rewards.append(episode_reward)
        evaluation_rewards = np.array(evaluation_rewards)
        distances_to_road_center = np.array(distances_to_road_center)

        mean_evaluation_reward = np.mean(evaluation_rewards)
        mean_distance_to_road_center = np.mean(distances_to_road_center)

        log = f'Eval Step: {eval_step}'
        log += f' Mean Eval Step: {eval_step / len(eval_index)}'
        log += f' Mean Eval Reward: {mean_evaluation_reward}'
        log += f' Mean Eval Track Distance: {mean_distance_to_road_center}'
        print(log)

        with self.writer.as_default():
            tf.summary.scalar('Evaluation Reward', mean_evaluation_reward, step=episode // 100)
            tf.summary.scalar('Evaluation Mean Distance to Road Center',
                              mean_distance_to_road_center,
                              step=episode // 100)

    def train(self, max_episode_max):
        self.update_target_network(1.0)
        self.writer = tf.summary.create_file_writer(f'summary/airsim_ddpg_{self.now}')
        total_time = 0
        avg_step = 0

        for ep in range(int(max_episode_max)):
            pre_noise = np.zeros(self.action_dim)
            step, episode_reward, done = 0, 0, False
            actor_losses, critic_losses = [], []
            moving_distances, distances_to_road_center = [], []
            state = self.env.reset()
            while not done:
                action = self.get_action(state)
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                action = np.clip(action + noise, self.action_bound_low, self.action_bound_high)
                next_state, reward, done, info = self.env.step(action)
                self.buffer.add_buffer(state, action, reward, next_state, done)

                if self.buffer.buffer_count() >= 10000:
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)
                    target_qs = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32),
                                                    self.target_actor(
                                                        tf.convert_to_tensor(next_states, dtype=tf.float32))])
                    td_targets = self.td_target(rewards, target_qs.numpy(), dones)

                    critic_loss = self.critic_learn(states, actions, td_targets)
                    actor_loss = self.actor_learn(states)

                    self.update_target_network(self.tau)
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)

                pre_noise = noise
                state = next_state

                episode_reward += reward
                step += 1
                moving_distances.append(np.linalg.norm(info['position'][:2] - info['preposition'][:2]))
                distances_to_road_center.append(info['track_distance'][0])

            total_time += step
            avg_step = 0.9 * avg_step + 0.1 * step if avg_step != 0 else step

            moving_distances = np.array(moving_distances)
            distances_to_road_center = np.array(distances_to_road_center)
            critic_losses = np.array(critic_losses)
            actor_losses = np.array(actor_losses)

            mean_critic_loss = np.round(critic_losses.mean(), 5) if critic_losses.size != 0 else 0
            mean_actor_loss = np.round(actor_losses.mean(), 5) if actor_losses.size != 0 else 0
            mean_distance_per_step = np.mean(moving_distances)
            mean_distance_to_road_center = np.mean(distances_to_road_center)

            log = f'Episode: {ep + 1}'
            log += f' Step: {step}'
            log += f' Total Time: {total_time}'
            log += f' Avg Step: {round(avg_step, 2)}'
            log += f' Reward: {round(episode_reward, 2)}'
            log += f' Actor Loss: {mean_actor_loss}'
            log += f' Critic Loss: {mean_critic_loss}'
            print(log)

            self.draw_tensorboard(ep, episode_reward, avg_step, mean_distance_per_step,
                                  mean_distance_to_road_center, mean_actor_loss, mean_critic_loss)
            self.save_weights(self.save_model_dir)

            # 100 스텝마다 에이전트 평가 진행
            if ep % 100 == 99:
                self.evaluation(ep)

    @staticmethod
    def ou_noise(x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
