import gym
from ddpg_keras_2 import DDPGagent

gym.envs.register(id='car_env-v0', entry_point='car_env_3:AirSimCarEnv')


def main():
    env = gym.make('car_env-v0', ip_address='127.0.0.1', image_shape=(84, 84))
    agent = DDPGagent(env)
    agent.train(1000000)


if __name__ == '__main__':
    main()
