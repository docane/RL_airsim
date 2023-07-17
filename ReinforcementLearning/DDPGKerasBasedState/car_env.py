import math
import numpy as np
import gym
from gym import spaces
import time
import airsim
import pandas as pd


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
            'position': np.zeros(2),
            'pose': np.zeros(3)
        }

        self.car = airsim.CarClient(ip=ip_address)

        low = np.array(
            [np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min],
            dtype=np.float32)

        high = np.array(
            [np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max],
            dtype=np.float32)

        self.observation_space = spaces.Box(low, high, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.trajectory = pd.read_csv('data/airsim_rec.txt', sep='\t')
        self.x = np.reshape(np.array(self.trajectory['POS_X'].values, dtype=np.float32), (-1, 1))
        self.y = np.reshape(np.array(self.trajectory['POS_Y'].values, dtype=np.float32), (-1, 1))
        self.max_x = np.max(self.x)
        self.min_x = np.min(self.x)
        self.max_y = np.max(self.y)
        self.min_y = np.min(self.y)

        self.x = (self.x - self.min_x) / (self.max_x - self.min_x)
        self.y = (self.y - self.min_y) / (self.max_y - self.min_y)

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)

        self.car_controls.brake = 0
        self.car_controls.throttle = 0.5
        self.car_controls.steering = 0
        self.car.setCarControls(self.car_controls)

        # print(self.car.simGetVehiclePose())
        # print(len(self.trajectory))

        rand = np.random.randint(0, len(self.trajectory))
        randrow = self.trajectory.iloc[rand]
        # print(randrow['POS_X'])
        self.car.simSetVehiclePose(airsim.Pose(airsim.Vector3r(randrow['POS_X'],
                                                               randrow['POS_Y'],
                                                               randrow['POS_Z']),
                                               airsim.Quaternionr(randrow['Q_X'],
                                                                  randrow['Q_Y'],
                                                                  randrow['Q_Z'],
                                                                  randrow['Q_W'])), True)

        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 0.5
        self.car_controls.steering = float(action)

        self.car.setCarControls(self.car_controls)

    def _get_obs(self):
        self.car_state = self.car.getCarState()

        self.state['position'] = self.car_state.kinematics_estimated.position.to_numpy_array()
        self.state['position'][0] = (self.state['position'][0] - self.min_x) / (self.max_x - self.min_x)
        self.state['position'][1] = (self.state['position'][1] - self.min_y) / (self.max_y - self.min_y)
        self.state['pose'] = self.car_state.kinematics_estimated.orientation.to_numpy_array()
        self.state['collision'] = self.car.simGetCollisionInfo().has_collided
        # print(self.car_state.kinematics_estimated.orientation)
        temp = []
        for v in self.state['position'][:2]:
            temp.append(v)
        for v in self.state['pose'][1:]:
            temp.append(v)

        return np.array(temp)

    def _compute_reward(self):
        state_x = self.state['position'][0]
        state_y = self.state['position'][1]
        # print(state_x)
        # print(state_y)

        distance = np.sqrt(np.square(state_x - self.x) + np.square(state_y - self.y))
        # print(distance.min())

        reward = gaussian(distance.min(), sigma=0.001) / 400
        if reward < 0.1:
            reward = -0.1
        else:
            reward = 0.5
        # print(reward)
        done = 0
        if self.state['collision']:
            reward += -5
            done = 1
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        time.sleep(0.05)
        return obs, reward, done, {}

    def reset(self):
        self._setup_car()
        return self._get_obs()


def gaussian(x, mean=0.0, sigma=1.0):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
