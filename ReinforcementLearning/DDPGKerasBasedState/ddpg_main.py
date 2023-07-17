import gym
from ddpg_learn import DDPGagent

gym.envs.register(id='car_env-v0', entry_point='car_env_state_16_17:AirSimCarEnv')


def main():
    env = gym.make('car_env-v0', ip_address='127.0.0.1')
    agent = DDPGagent(env)
    # agent.load_weights('./models/airsim_ddpg_model_2023_05_31_20_59_57/')
    agent.train(1000000)


if __name__ == '__main__':
    main()
