import gym
import numpy as np

gym.envs.register(id='car_env-v0', entry_point='car_env_state_1:AirSimCarEnv')


def main():
    env = gym.make('car_env-v0', ip_address='127.0.0.1')
    # print(env.observation_space)
    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.shape[0])
    # print(env.action_space.high[0])
    state = env.reset()
    print(state)
    # print(env.observation_space)


if __name__ == '__main__':
    main()
