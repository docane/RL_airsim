import gym
gym.envs.register(id='car_env-v0', entry_point='car_env_state_1:AirSimCarEnv')

env = gym.make('car_env-v0', ip_address='127.0.0.1')

while True:
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(state)