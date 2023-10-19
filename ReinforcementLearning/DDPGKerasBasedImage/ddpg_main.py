from ddpg_learn import DDPGagent
from car_env import AirSimCarEnv


def main():
    env = AirSimCarEnv()
    agent = DDPGagent(env)
    # agent.load_weights('./models/airsim_ddpg_model_2023_09_24_03_00_11/')
    agent.train(1000000)


if __name__ == '__main__':
    main()
