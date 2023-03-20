import numpy as np
import gym
from gym import spaces
import time
import airsim


class AirSimEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self):
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()


class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address):
        super().__init__()

        self.start_ts = 0

        self.state = {
            'position': np.zeros(3),
            'pose': np.zeros(3),
            'collision': False
        }

        self.car = airsim.CarClient(ip=ip_address)

        low = np.array(
            [np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             0],
            dtype=np.float32)

        high = np.array(
            [np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             1],
            dtype=np.float32)

        self.observation_space = spaces.Box(low, high, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.car_controls = airsim.CarControls()
        self.car_state = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)
        self.car.simPause(True)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 0.5
        self.car_controls.steering = float(action)

        self.car.setCarControls(self.car_controls)
        self.car.simContinueForTime(0.1)

    def _get_obs(self):
        self.car_state = self.car.getCarState()

        self.state['position'] = self.car_state.kinematics_estimated.position.to_numpy_array()
        self.state['position'][0] = self.state['position'][0] / 1000
        self.state['position'][1] = self.state['position'][1] / 500
        self.state['pose'] = self.car_state.kinematics_estimated.orientation.to_numpy_array()
        self.state['collision'] = self.car.simGetCollisionInfo().has_collided
        temp = []
        for v in self.state['position']:
            temp.append(v)
        for v in self.state['pose']:
            temp.append(v)
        if not self.state['collision']:
            temp.append(0)
        else:
            temp.append(1)
        return np.array(temp)

    def _compute_reward(self):
        # reward = self.car_state.speed / 10
        reward = 0.0
        done = 0
        if self.state['collision']:
            reward = -0.1
            done = 1
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, {}

    def reset(self):
        self._setup_car()
        return self._get_obs()
