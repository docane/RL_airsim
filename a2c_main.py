import numpy as np
import gym
from a2c import A2Cagent
import cv2 as cv

gym.envs.register(id='car_env-v0', entry_point='car_env_2:AirSimCarEnv')

env = gym.make('car_env-v0', ip_address='127.0.0.1', image_shape=(84, 84))

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action = env.action_space.high[0]

agent = A2Cagent(action_size, max_action)

for e in range(1000000):
    done = False
    score = 0
    losses = []
    sigmas = []
    state = env.reset()
    state = np.reshape(state, [1, 84, 84, 3])
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        cv.imshow('asdf', next_state)
        cv.waitKey(1)
        next_state = np.reshape(next_state, [1, 84, 84, 3])
        score += reward
        loss, sigma = agent.train_model(state, action, reward, next_state, done)
        losses.append(loss)
        sigmas.append(sigma)
        state = next_state
        if done:
            log = 'episode: {:3d}'.format(e)
            log += ' | score: {:3.0f}'.format(score)
            log += ' | loss: {:3.3f}'.format(np.mean(losses))
            log += ' | sigma: {:3.3f}'.format(np.mean(sigmas))
            print(log)
