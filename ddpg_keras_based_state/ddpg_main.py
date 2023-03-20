import gym
from state_ddpg_1 import DDPGagent

gym.envs.register(id='car_env-v0', entry_point='car_env_state_13:AirSimCarEnv')


def main():
    env = gym.make('car_env-v0', ip_address='127.0.0.1')
    agent = DDPGagent(env)
    agent.train(1000000)


if __name__ == '__main__':
    main()
