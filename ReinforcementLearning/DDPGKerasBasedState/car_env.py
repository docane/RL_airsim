import numpy as np
import gym
from gym import spaces
import time
import airsim
import pandas as pd
from airsim import Vector3r


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

        self.state = {
            'preposition': np.zeros(3),
            'position': np.zeros(3),
            'pose': np.zeros(3),
            'prepose': np.zeros(3),
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'target_point': np.zeros(2),
            'next_target_point': np.zeros(2),
            'angle': np.zeros(1)
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
             np.finfo(np.float32).max],
            dtype=np.float32)

        self.observation_space = spaces.Box(low, high, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.trajectory = pd.read_csv('data/airsim_rec_2.txt', sep='\t')

        pos_x = self.trajectory['POS_X'].values.astype(np.float32)
        pos_y = self.trajectory['POS_Y'].values.astype(np.float32)
        pos_z = self.trajectory['POS_Z'].values.astype(np.float32)
        self.max_pos = 1000.0
        self.min_pos = -1000.0
        x = (pos_x - self.min_pos) / (self.max_pos - self.min_pos)
        y = (pos_y - self.min_pos) / (self.max_pos - self.min_pos)
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        self.velocity_divide = 2000.0

        self.pts = np.column_stack((x, y))
        self.pts_1 = np.column_stack((pos_x.astype(float), pos_y.astype(float), pos_z.astype(float)))
        self.temp = [0]

        # 5m 단위로 경로 포인트 잡기
        for i in range(1, len(self.trajectory)):
            distance = np.linalg.norm(self.pts[i] - self.pts[self.temp[-1]])
            if distance > (5 / 2000):
                self.temp.append(i)
        self.temp = np.array(self.temp)

        self.car.simPlotLineStrip(points=[Vector3r(x, y, z + 0.5) for x, y, z in self.pts_1], is_persistent=True)
        self._success = False

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)

        if self._success:
            start_index = 0
        else:
            start_index = np.random.randint(0, len(self.trajectory) - 200)
        # start_index = 0
        self._car_position_init(start_index)
        self._do_action(0)

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

        dist = np.linalg.norm(self.pts - car_pt, axis=1)
        min_dist_index = dist.argmin()

        min_dist_temp_index = 0
        for i in range(len(self.temp)):
            if min_dist_index < self.temp[i]:
                min_dist_temp_index = i - 1
                break
        self.route_point = [index for index in range(min_dist_temp_index, min_dist_temp_index + 3)]
        v1 = self.pts[self.temp[self.route_point[2]]] - car_pt
        v1_norm = np.linalg.norm(v1)
        v2 = (v1 / v1_norm) * 5

        self.state['target_point'][0] = self.state['next_target_point'][0]
        self.state['target_point'][1] = self.state['next_target_point'][1]
        self.state['next_target_point'][0] = v2[0]
        self.state['next_target_point'][1] = v2[1]

        first_target_point = self.pts[self.temp[self.route_point[0]]]
        second_target_point = self.pts[self.temp[self.route_point[1]]]

        car_vel = self.state['linear_velocity'][:2]
        car_vel_norm = np.linalg.norm(car_vel)

        track_vector = second_target_point - first_target_point
        track_vector_norm = np.linalg.norm(track_vector)

        target_dir_vec = track_vector / track_vector_norm
        car_dir_vec = car_vel / car_vel_norm if car_vel_norm != 0 else car_vel * 0
        ip = car_dir_vec[0] * target_dir_vec[1] - car_dir_vec[1] * target_dir_vec[0]
        theta = np.arcsin(ip)
        self.state['angle'][0] = theta / np.pi

        temp = [self.state['position'][0],
                self.state['position'][1],
                self.state['pose'][2],
                self.state['linear_velocity'][0],
                self.state['linear_velocity'][1],
                self.state['angular_velocity'][2],
                self.state['next_target_point'][0],
                self.state['next_target_point'][1],
                self.state['angle'][0]]

        return np.array(temp)

    def _compute_reward(self):
        car_pt = self.state['position'][:2]
        car_vel = self.state['linear_velocity'][:2]
        car_vel_norm = np.linalg.norm(car_vel)

        first_target_point = self.pts[self.temp[self.route_point[0]]]
        second_target_point = self.pts[self.temp[self.route_point[1]]]
        track_vector = second_target_point - first_target_point
        track_vector_norm = np.linalg.norm(track_vector)

        dot_product = np.dot(track_vector, car_vel)
        angle_radian = np.arccos(np.clip(dot_product / (track_vector_norm * car_vel_norm), -1, 1)) if car_vel_norm != 0 else -1
        if angle_radian != -1:
            vxcostheta = car_vel_norm * np.cos(angle_radian)
            vxsintheta = car_vel_norm * np.sin(angle_radian)
        else:
            vxcostheta = 0
            vxsintheta = 0

        trackpos = min(np.linalg.norm(self.pts - car_pt, axis=1))
        vxtrackpos = trackpos * car_vel_norm

        reward = (vxcostheta - vxsintheta - trackpos - 100 * vxtrackpos) * 1000
        print('Reward:', reward)

        done = self._check_done()
        if self.state['collision']:
            # reward -= 1
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

    def close(self):
        self.car.reset()
        self.car.enableApiControl(False)

    def _car_position_init(self, index):
        while True:
            row = self.trajectory.iloc[index]
            self.car.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(row['POS_X'],
                                            row['POS_Y'],
                                            row['POS_Z']),
                            airsim.Quaternionr(row['Q_X'],
                                               row['Q_Y'],
                                               row['Q_Z'],
                                               row['Q_W'])), True)

            time.sleep(0.1)
            car_pt = self.car.getCarState().kinematics_estimated.position.to_numpy_array()
            min_dist = min(np.linalg.norm((self.pts_1 - car_pt), axis=1))
            if min_dist < 1:
                break

    def _check_done(self):
        car_pt = self.state['position'][:2]

        dist = np.linalg.norm(self.pts - car_pt, axis=1)
        min_dist_index = dist.argmin()

        if min_dist_index > 770:
            self._success = True
            return True
        else:
            self._success = False
            return False

    @staticmethod
    def gaussian(x, mean=0.0, sigma=1.0):
        return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
