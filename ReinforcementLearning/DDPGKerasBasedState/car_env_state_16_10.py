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
             np.finfo(np.float32).max],
            dtype=np.float32)

        self.observation_space = spaces.Box(low, high, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.trajectory = pd.read_csv('data/airsim_rec_1.txt', sep='\t')
        self.x = np.reshape(np.array(self.trajectory['POS_X'].values, dtype=np.float32), (-1, 1))
        self.y = np.reshape(np.array(self.trajectory['POS_Y'].values, dtype=np.float32), (-1, 1))
        self.max_pos = 1000
        self.min_pos = -1000
        self.x = (self.x - self.min_pos) / (self.max_pos - self.min_pos)
        self.y = (self.y - self.min_pos) / (self.max_pos - self.min_pos)
        self.velocity_divide = 2000

        self.pts = np.reshape(np.array([(self.x[i], self.y[i]) for i in range(len(self.trajectory))]), (-1, 2))
        self.pts_0 = np.concatenate((self.pts[-135:], self.pts[:1000]))
        self.pts_1 = np.concatenate((self.pts[-1000:], self.pts[:130]))

        self.temp_0 = [0]
        for i in range(500):
            for j in range(self.temp_0[-1], len(self.pts_0)):
                if math.sqrt((self.pts_0[j][0] - self.pts_0[self.temp_0[-1]][0]) ** 2 + (
                        self.pts_0[j][1] - self.pts_0[self.temp_0[-1]][1]) ** 2) > 5 / 2000:
                    self.temp_0.append(j)
                    break

        self.temp_1 = [0]
        for i in range(500):
            for j in range(self.temp_1[-1], len(self.pts_1)):
                if math.sqrt((self.pts_1[j][0] - self.pts_1[self.temp_1[-1]][0]) ** 2 + (
                        self.pts_1[j][1] - self.pts_1[self.temp_1[-1]][1]) ** 2) > 5 / 2000:
                    self.temp_1.append(j)
                    break

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)

        self.car_controls.brake = 0
        self.car_controls.throttle = 0.5
        self.car_controls.steering = 0
        self.car.setCarControls(self.car_controls)

        randint = np.random.randint(0, len(self.trajectory))
        # randint = 1010
        if randint < 946:
            self.direction = 0
            start_index = randint
        else:
            self.direction = 1
            start_index = randint - 1885
        start_row = self.trajectory.iloc[start_index]
        self.car.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(start_row['POS_X'],
                                        start_row['POS_Y'],
                                        start_row['POS_Z']),
                        airsim.Quaternionr(start_row['Q_X'],
                                           start_row['Q_Y'],
                                           start_row['Q_Z'],
                                           start_row['Q_W'])), True)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 0.5
        self.car_controls.steering = float(action)

        self.car.setCarControls(self.car_controls)

    def _get_obs(self):
        self.car_state = self.car.getCarState()
        kin_est = self.car_state.kinematics_estimated

        self.state['preposition'] = self.state['position']
        self.state['prepose'] = self.state['pose']

        self.state['position'] = kin_est.position.to_numpy_array()
        self.state['position'][:2] = (self.state['position'][:2] - self.min_pos) / (self.max_pos - self.min_pos)

        self.state['pose'] = np.array(airsim.to_eularian_angles(kin_est.orientation)) / np.pi

        self.state['linear_velocity'] = kin_est.linear_velocity.to_numpy_array() / self.velocity_divide
        self.state['angular_velocity'] = kin_est.angular_velocity.to_numpy_array() / np.pi

        self.state['collision'] = self.car.simGetCollisionInfo().has_collided

        car_pt = self.state['position'][:2]

        if self.direction == 0:
            dist = np.array(
                [math.sqrt(((car_pt[0] - self.pts_0[i][0]) ** 2) + ((car_pt[1] - self.pts_0[i][1]) ** 2)) for i in
                 range(len(self.pts_0))])
            min_dist_index = np.argmin(dist)
            if min_dist_index > (len(self.pts_0) - 56):
                self.direction = 1

        elif self.direction == 1:
            dist = np.array(
                [math.sqrt(((car_pt[0] - self.pts_1[i][0]) ** 2) + ((car_pt[1] - self.pts_1[i][1]) ** 2)) for i in
                 range(len(self.pts_1))])
            min_dist_index = np.argmin(dist)
            if min_dist_index > (len(self.pts_1) - 45):
                self.direction = 0

        if self.direction == 0:
            pts = self.pts_0
            temp = self.temp_0
        else:
            pts = self.pts_1
            temp = self.temp_1

        dist = np.array(
            [math.sqrt(((car_pt[0] - pts[i][0]) ** 2) + ((car_pt[1] - pts[i][1]) ** 2)) for i in range(len(pts))])

        min_dist_index = np.argmin(dist)

        min_dist_temp_index = 0
        for i in range(len(temp)):
            if min_dist_index < temp[i]:
                min_dist_temp_index = i - 1
                break
        route_point = [index for index in range(min_dist_temp_index, min_dist_temp_index + 3)]

        self.target_point = np.array(
            [(pts[temp[route_point[2]]][0] + car_pt[0]) / 2,
             (pts[temp[route_point[2]]][1] + car_pt[1]) / 2])
        v1 = self.target_point - car_pt[:2]
        v1_norm = np.linalg.norm(v1)
        v2 = v1 / v1_norm * 5

        self.state['target_point'][0] = self.state['next_target_point'][0]
        self.state['target_point'][1] = self.state['next_target_point'][1]
        self.state['next_target_point'][0] = v2[0]
        self.state['next_target_point'][1] = v2[1]

        # 바로 뒤에 있는 타겟 포인트
        first_target_point = np.array(
            [self.x[temp[route_point[0]]][0],
             self.y[temp[route_point[0]]][0]])

        # 바로 앞에 있는 타겟 포인트
        second_target_point = np.array(
            [self.x[temp[route_point[1]]][0],
             self.y[temp[route_point[1]]][0]])

        target_dir_vec = (second_target_point - first_target_point) / np.linalg.norm(
            first_target_point - second_target_point)
        car_dir_vec = self.state['linear_velocity'][:2] / (
                np.linalg.norm(self.state['linear_velocity'][:2]) + 0.0000001)
        ip = car_dir_vec[0] * target_dir_vec[1] - car_dir_vec[1] * target_dir_vec[0]
        theta = math.asin(ip)
        self.state['angle'][0] = theta

        min_dist = np.min(dist)
        if (self.pts[min_dist_index][1] < car_pt[1]) & (self.direction == 0):
            pass
        else:
            if (self.pts[min_dist_index][0] < car_pt[0]) & (self.direction == 0):
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
        v1_norm = np.linalg.norm(v1)
        car_dir_vec = self.state['linear_velocity'][:2] / (np.linalg.norm(self.state['linear_velocity'][:2]) + 0.00001)
        target_dir_vec = v1 / (v1_norm + 0.00001)
        ip = car_dir_vec[0] * target_dir_vec[0] + car_dir_vec[1] * target_dir_vec[1]
        theta = math.acos(ip)
        # if self.direction == 1:
        #     theta = np.pi - theta
        angular_reward = (1 / (theta / np.pi) / 100)
        print('Angular Reward:', angular_reward)

        reward = angular_reward

        done = 0
        if self.state['collision']:
            # reward -= 10
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
