import gym
from ddpg_learn import DDPGagent
import tensorflow as tf

gym.envs.register(id='car_env-v0', entry_point='car_env:AirSimCarEnv')


def main():
    env = gym.make('car_env-v0', ip_address='127.0.0.1')
    agent = DDPGagent(env)
    agent.load_weights('./models/airsim_ddpg_model_2023_09_07_20_01_58/')
    state = env.reset()
    while True:
        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            state = env.reset()


if __name__ == '__main__':
    main()
