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
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'target_point': np.zeros(2),
            'next_target_point': np.zeros(2),
            'angle': np.zeros(1),
            'track_distance': np.zeros(1)
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
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).max,
             np.finfo(np.float32).min],
            dtype=np.float32)

        self.observation_space = spaces.Box(low, high, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.trajectory = pd.read_csv('./data/airsim_rec_2.txt', sep='\t')
        self.x = np.reshape(np.array(self.trajectory['POS_X'].values, dtype=np.float32), (-1, 1))
        self.y = np.reshape(np.array(self.trajectory['POS_Y'].values, dtype=np.float32), (-1, 1))
        self.max_x = np.max(self.x)
        self.min_x = np.min(self.x)
        self.max_y = np.max(self.y)
        self.min_y = np.min(self.y)
        self.max_x = 1000
        self.min_x = -1000
        self.max_y = 1000
        self.min_y = -1000
        self.x = (self.x - self.min_x) / (self.max_x - self.min_x)
        self.y = (self.y - self.min_y) / (self.max_y - self.min_y)
        self.velocity_divide = 2000

        self.pts = [np.array([self.x[i], self.y[i]]) for i in range(len(self.x))]
        self.temp = [0]

        # 5m 단위로 경로 포인트 잡기
        for i in range(len(self.trajectory)):
            for j in range(self.temp[-1], len(self.trajectory)):
                if math.sqrt((self.pts[j][0] - self.pts[self.temp[-1]][0]) ** 2 + (
                        self.pts[j][1] - self.pts[self.temp[-1]][1]) ** 2) > 5 / 2000:
                    self.temp.append(j)
                    break

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)

        self.car_controls.brake = 0
        self.car_controls.throttle = 0.5
        self.car_controls.steering = 0
        self.car.setCarControls(self.car_controls)

        self.rand = np.random.randint(0, len(self.trajectory) - 200)
        # self.rand = 0
        randrow = self.trajectory.iloc[self.rand]
        self.car.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(randrow['POS_X'],
                                        randrow['POS_Y'],
                                        randrow['POS_Z']),
                        airsim.Quaternionr(randrow['Q_X'],
                                           randrow['Q_Y'],
                                           randrow['Q_Z'],
                                           randrow['Q_W'])), True)

        self.state['target_point'][0] = self.x[5] / np.linalg.norm(self.pts[5]) * 5
        self.state['target_point'][1] = self.y[5] / np.linalg.norm(self.pts[5]) * 5

        # self.car.simPause(True)
        # self.car.simContinueForTime(2)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 0.5
        self.car_controls.steering = float(action)

        self.car.setCarControls(self.car_controls)
        # self.car.simContinueForTime(0.5)

    def _get_obs(self):
        self.car_state = self.car.getCarState()
        kin_est = self.car_state.kinematics_estimated

        self.state['preposition'] = self.state['position']
        self.state['prepose'] = self.state['pose']

        self.state['position'] = kin_est.position.to_numpy_array()
        self.state['position'][0] = (self.state['position'][0] - self.min_x) / (self.max_x - self.min_x)
        self.state['position'][1] = (self.state['position'][1] - self.min_y) / (self.max_y - self.min_y)

        self.state['pose'] = np.array(airsim.to_eularian_angles(kin_est.orientation)) / np.pi

        self.state['linear_velocity'] = kin_est.linear_velocity.to_numpy_array() / self.velocity_divide
        self.state['angular_velocity'] = kin_est.angular_velocity.to_numpy_array() / np.pi

        self.state['collision'] = self.car.simGetCollisionInfo().has_collided

        car_pt = self.state['position'][:2]

        dist = np.array([math.sqrt(((car_pt[0] - self.pts[i][0]) ** 2) + ((car_pt[1] - self.pts[i][1]) ** 2)) for i in
                         range(len(self.pts))])

        min_dist_index = np.argmin(dist)

        min_dist_temp_index = 0

        for i in range(len(self.temp)):
            if min_dist_index < self.temp[i]:
                min_dist_temp_index = i - 1
                break
        route_point = [index for index in range(min_dist_temp_index, min_dist_temp_index + 3)]

        self.target_point = np.array(
            [(self.x[self.temp[route_point[2]]][0] + car_pt[0]) / 2,
             (self.y[self.temp[route_point[2]]][0] + car_pt[1]) / 2])
        v1 = self.target_point - car_pt[:2]
        v1_norm = np.linalg.norm(v1)
        v2 = v1 / v1_norm * 5

        self.state['target_point'][0] = self.state['next_target_point'][0]
        self.state['target_point'][1] = self.state['next_target_point'][1]
        self.state['next_target_point'][0] = v2[0]
        self.state['next_target_point'][1] = v2[1]

        # 바로 뒤에 있는 타겟 포인트
        first_target_point = np.array(
            [self.x[self.temp[route_point[0]]][0],
             self.y[self.temp[route_point[0]]][0]])

        # 바로 앞에 있는 타겟 포인트
        second_target_point = np.array(
            [self.x[self.temp[route_point[1]]][0],
             self.y[self.temp[route_point[1]]][0]])

        target_dir_vec = (second_target_point - first_target_point) / np.linalg.norm(
            first_target_point - second_target_point)
        car_dir_vec = self.state['linear_velocity'][:2] / (
                np.linalg.norm(self.state['linear_velocity'][:2]) + 0.0000001)
        ip = car_dir_vec[0] * target_dir_vec[1] - car_dir_vec[1] * target_dir_vec[0]
        theta = math.asin(ip)
        self.state['angle'][0] = theta

        min_dist = np.min(dist)
        if self.pts[min_dist_index][1] < car_pt[1]:
            pass
        else:
            if self.pts[min_dist_index][0] < car_pt[0]:
                pass
            else:
                min_dist = -min_dist
        self.state['track_distance'][0] = min_dist

        temp = []
        for v in self.state['position'][:2]:
            temp.append(v)
        temp.append(self.state['pose'][2])
        for v in self.state['linear_velocity'][:2]:
            temp.append(v)
        temp.append(self.state['angular_velocity'][2])
        for v in self.state['next_target_point'][:2]:
            temp.append(v)
        temp.append(self.state['angle'][0] / np.pi)
        temp.append(self.state['track_distance'][0])

        return np.array(temp)

    def _compute_reward(self):
        car_pt = self.state['position'][:2]

        v1 = self.target_point - car_pt[:2]
        # v1 = self.state['next_target_point'] - car_pt[:2]
        v1_norm = np.linalg.norm(v1)
        v2 = self.state['next_target_point']
        dist_reward = 1 / (5 - np.linalg.norm(v2 - (self.state['linear_velocity'][:2]))) / 10000
        # print('Distance Reward:', dist_reward)

        car_dir_vec = self.state['linear_velocity'][:2] / (np.linalg.norm(self.state['linear_velocity'][:2]) + 0.00001)
        # print(car_dir_vec)
        target_dir_vec = v1 / (v1_norm + 0.00001)
        ip = car_dir_vec[0] * target_dir_vec[0] + car_dir_vec[1] * target_dir_vec[1]
        theta = math.acos(ip)
        angular_reward = (1 / (theta / np.pi) / 10)
        print('Angular Reward:', angular_reward)

        # reward = dist_reward
        reward = angular_reward

        done = 0
        if self.state['collision']:
            # reward -= 0.1
            done = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        time.sleep(0.5)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, {}

    def reset(self):
        self._setup_car()
        time.sleep(2)
        return self._get_obs()
