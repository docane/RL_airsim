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
            'preposition': np.zeros(3),
            'position': np.zeros(3),
            'pose': np.zeros(3),
            'prepose': np.zeros(3),
            'linear_velocity': np.zeros(2),
            'angular_velocity': np.zeros(2)
        }

        self.car = airsim.CarClient(ip=ip_address)

        low = np.array(
            [np.finfo(np.float32).min,
             np.finfo(np.float32).min,
             np.finfo(np.float32).min,
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
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max],
            dtype=np.float32)

        self.observation_space = spaces.Box(low, high, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.trajectory = pd.read_csv('./data/airsim_rec_1.txt', sep='\t')
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

        self.rand = np.random.randint(0, len(self.trajectory))
        randrow = self.trajectory.iloc[self.rand]
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

        self.state['preposition'] = self.state['position']
        self.state['prepose'] = self.state['pose']

        self.state['position'] = self.car_state.kinematics_estimated.position.to_numpy_array()
        self.state['position'][0] = (self.state['position'][0] - self.min_x) / (self.max_x - self.min_x)
        self.state['position'][1] = (self.state['position'][1] - self.min_y) / (self.max_y - self.min_y)

        self.state['pose'] = airsim.to_eularian_angles(self.car_state.kinematics_estimated.orientation)

        self.state['linear_velocity'] = self.car_state.kinematics_estimated.linear_velocity.to_numpy_array()
        self.state['angular_velocity'] = self.car_state.kinematics_estimated.angular_velocity.to_numpy_array()
        # print(self.state['linear_velocity'])
        # print(self.state['angular_velocity'])

        self.state['collision'] = self.car.simGetCollisionInfo().has_collided

        temp = []
        for v in self.state['position'][:2]:
            temp.append(v)
        temp.append(self.state['pose'][2])
        for v in self.state['linear_velocity'][:2]:
            temp.append(v)
        for v in self.state['angular_velocity'][:2]:
            temp.append(v)
        # print(np.array(temp))

        return np.array(temp)

    def _compute_reward(self):
        min_speed = 3
        max_speed = 30
        beta = 3
        thresh_dist = 3.5
        pts = [np.array([self.x[i], self.y[i]]) for i in range(len(self.x))]
        car_pre_pt = self.state['preposition'][:2]
        car_pt = self.state['position'][:2]
        # print(car_pt)
        dist = 10000000
        # print(len(pts))
        for i in range(len(pts) - 1):
            dist = min(dist,
                       np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                                      ) / np.linalg.norm(pts[i] - pts[i + 1]), )
            # print(np.argmin(np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
            #                                ) / np.linalg.norm(pts[i] - pts[i + 1])))
            # print(min(np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))) / np.linalg.norm(pts[i] - pts[i + 1])))
        # print(dist)

        if dist > thresh_dist:
            reward = -3
        else:
            reward_dist = math.exp(-beta * dist)  # - 0.5
            reward_speed = ((self.car_state.speed - min_speed) / (max_speed - min_speed))  # - 0.5
            reward = reward_dist + reward_speed
        print(reward)
        done = 0
        if self.state['collision']:
            done = 1
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        # time.sleep(0.25)
        time.sleep(0.1)
        return obs, reward, done, {}

    def reset(self):
        self._setup_car()
        return self._get_obs()


def gaussian(x, mean=0.0, sigma=1.0):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
