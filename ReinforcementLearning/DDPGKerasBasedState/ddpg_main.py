from ddpg_learn import DDPGagent
from car_env import AirSimCarEnv


def main():
    env = AirSimCarEnv()
    agent = DDPGagent(env)
    # agent.load_weights('./models/airsim_ddpg_model_2023_05_31_20_59_57/')
    agent.train(1000000)


if __name__ == '__main__':
    main()
