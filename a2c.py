import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd


class A2C(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=4, activation='tanh')
        self.conv2 = Conv2D(64, 4, strides=2, activation='tanh')
        self.conv3 = Conv2D(64, 3, strides=1, activation='tanh')
        self.flatten = Flatten()
        self.dense = Dense(2048, activation='tanh')
        self.mu = Dense(action_size, activation='tanh')
        self.sigma = Dense(action_size, activation='sigmoid')
        self.value = Dense(1)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = sigma + 1e-5
        value = self.value(x)
        return mu, sigma, value


class A2Cagent:
    def __init__(self, action_size, max_action):
        self.action_size = action_size
        self.max_action = max_action

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = A2C(self.action_size)

        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)

    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            mu, sigma, value = self.model(state)
            _, _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            advantage = tf.stop_gradient(target - value[0])
            dist = tfd.Normal(loc=mu, scale=sigma)
            action_prob = dist.prob([action])[0]
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            loss = 0.1 * actor_loss + critic_loss
            print('value:', value)
            print('mu:', mu, 'sigma:', sigma)
            print('Actor Loss:', actor_loss, 'Critic Loss:', critic_loss)

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return loss, sigma
