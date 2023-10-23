from ddpg_learn import DDPGagent
from car_env import AirSimCarEnv
import tensorflow as tf


def main():
    env = AirSimCarEnv()
    agent = DDPGagent(env)
    agent.load_weights('./models/airsim_ddpg_model_2023_10_19_21_40_47/')
    state = env.reset(0)
    while True:
        action = agent.actor([tf.convert_to_tensor([state['state']], dtype=tf.float32),
                              tf.convert_to_tensor([state['image']], dtype=tf.float32)]).numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            state = env.reset(0)


if __name__ == '__main__':
    main()
