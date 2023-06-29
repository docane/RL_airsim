import gym
from state_ddpg_10 import DDPGagent
import tensorflow as tf

gym.envs.register(id='car_env-v0', entry_point='car_env_state_16_12:AirSimCarEnv')


def main():
    env = gym.make('car_env-v0', ip_address='127.0.0.1')
    agent = DDPGagent(env)
    # agent.load_weights('./models/airsim_ddpg_model_2023_04_06_14_46_01/')
    # agent.load_weights('./models/airsim_ddpg_model_2023_04_05_13_11_57/')
    #     agent.load_weights('./models/airsim_ddpg_model_2023_05_09_15_05_55/')
    #     agent.load_weights('./models/airsim_ddpg_model_2023_05_24_15_10_24/')
    #     agent.load_weights('./models/airsim_ddpg_model_2023_05_25_15_18_45/')
    #     agent.load_weights('./models/airsim_ddpg_model_2023_05_26_09_48_44/')
    # agent.load_weights('./models/airsim_ddpg_model_2023_05_26_14_04_18/')
    # agent.load_weights('./models/airsim_ddpg_model_2023_05_27_01_22_48/')
    # agent.load_weights('./models/airsim_ddpg_model_2023_05_27_02_25_46/')
    # agent.load_weights('./models/airsim_ddpg_model_2023_05_31_17_08_55/')
    # agent.load_weights('./models/airsim_ddpg_model_2023_05_31_21_43_03/')
    # agent.load_weights('./models/airsim_ddpg_model_2023_06_01_13_53_19/')
    # agent.load_weights('./models/airsim_ddpg_model_2023_06_02_12_49_09/')
    agent.load_weights('./models/airsim_ddpg_model_2023_06_28_14_18_23/')

    state = env.reset()

    while True:
        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0]
        state, reward, done, _ = env.step(action)
        if done:
            state = env.reset()
            break
            # continue


if __name__ == '__main__':
    main()
