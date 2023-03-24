import gym
from state_ddpg_3 import DDPGagent
import tensorflow as tf

gym.envs.register(id='car_env-v0', entry_point='car_env_state_14:AirSimCarEnv')


def main():
    env = gym.make('car_env-v0', ip_address='127.0.0.1')
    agent = DDPGagent(env)
    agent.load_weights('./models/2023-02-16-16-21-50/')
    state = env.reset()

    while True:
        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0]
        state, reward, done, _ = env.step(action)
        if done:
            state = env.reset()
            continue


if __name__ == '__main__':
    main()
