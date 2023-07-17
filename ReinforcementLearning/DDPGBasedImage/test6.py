import gym
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Lambda, concatenate, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

gym.envs.register(id='car_env-v0', entry_point='car_env_2:AirSimCarEnv')

env = gym.make('car_env-v0', ip_address='127.0.0.1', image_shape=(84, 84))
state = env.reset()
print(state.shape)


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
        self.action = Dense(self.action_size, activation='tanh')

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


extractor = Extractor()
extractor.build(input_shape=(None, 84, 84, 3))
x = extractor(state)
print(x.shape)
actor = Actor(1, 1)
actor.build(input_shape=(None, 84, 84, 3))
action = actor(x)
print(action)
