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

    def render(self, mode='rgb_array'):
        return self._get_obs()


class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address='127.0.0.1'):
        super().__init__()

        self.client = airsim.CarClient(ip=ip_address)
        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.state = {
            'position': np.zeros(3),
            'preposition': np.zeros(3),
            'linear_velocity': np.zeros(3),
            'pose': np.zeros(3),
            'prepose': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'angle': np.zeros(1),
            'track_distance': np.zeros(1)
        }

        low = np.array([np.finfo(np.float32).min for _ in range(13)], dtype=np.float32)
        high = np.array([np.finfo(np.float32).max for _ in range(13)], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, shape=(13,), dtype=np.float32)

        action_low = np.array([-1, 0, 0], dtype=np.float32)
        action_high = np.array([1, 1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(3,), dtype=np.float32)

        self.trajectory = pd.read_csv('./data/airsim_rec.txt', sep='\t')

        pos_x = self.trajectory['POS_X'].values.astype(np.float32)
        pos_y = self.trajectory['POS_Y'].values.astype(np.float32)
        pos_z = self.trajectory['POS_Z'].values.astype(np.float32)
        self.max_pos = 1000.0
        self.min_pos = -1000.0

        self.xy_points = np.column_stack((pos_x, pos_y))
        self.xyz_points = np.column_stack((pos_x.astype(float),
                                           pos_y.astype(float),
                                           pos_z.astype(float)))

        self.client.simPlotLineStrip(points=[Vector3r(x, y, z + 0.5) for x, y, z in self.xyz_points],
                                     is_persistent=True)
        self._success = False
        self.timestep = 0

    def _setup_car(self, start_index: int = None):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simPause(True)

        if start_index is None:
            if self._success:
                start_index = 0
            else:
                start_index = np.random.randint(0, len(self.trajectory) - 1100)
        self._car_position_init(start_index)

    def __del__(self):
        self.client.reset()

    def _do_action(self, actions):
        self.car_controls.brake = float(actions[2])
        self.car_controls.throttle = float(actions[1])
        self.car_controls.steering = float(actions[0])

        self.client.setCarControls(self.car_controls)

    def _get_obs(self):
        self.car_state = self.client.getCarState()
        kin_est = self.car_state.kinematics_estimated

        self.state['preposition'] = self.state['position']
        self.state['prepose'] = self.state['pose']
        self.state['position'] = kin_est.position.to_numpy_array()
        self.state['pose'] = np.array(airsim.to_eularian_angles(kin_est.orientation))
        self.state['linear_velocity'] = kin_est.linear_velocity.to_numpy_array()
        self.state['angular_velocity'] = kin_est.angular_velocity.to_numpy_array()
        self.state['collision'] = self.client.simGetCollisionInfo().has_collided

        car_point = self.state['position'][:2]

        distances = np.linalg.norm(self.xy_points - car_point, axis=1)
        min_distance_index = distances.argmin()
        self.current_point = self.xy_points[min_distance_index]

        # 가장 가까운 포인트에서 1m 앞 포인트 찾기
        self.target_point = self.xy_points[min_distance_index]
        for i in range(min_distance_index, len(self.xy_points)):
            distance = np.linalg.norm(self.xy_points[i] - self.xy_points[min_distance_index])
            if distance > 1:
                self.target_point = self.xy_points[i]
                break

        car_vel = self.state['linear_velocity'][:2]
        car_vel_norm = np.linalg.norm(car_vel)

        track_vector = self.target_point - self.current_point
        track_vector_norm = np.linalg.norm(track_vector)
        target_dir_vec = track_vector / track_vector_norm

        car_dir_vec = car_vel / car_vel_norm if car_vel_norm != 0 else car_vel * 0
        ip = np.clip(car_dir_vec[0] * target_dir_vec[1] - car_dir_vec[1] * target_dir_vec[0], -1, 1)
        theta = np.arcsin(ip)
        self.state['angle'][0] = theta

        min_dist = np.min(distances)
        self.state['track_distance'][0] = min_dist

        normalized_position = self.min_max_scaler(self.state['position'][:2], self.min_pos, self.max_pos, -1, 1)
        normalized_preposition = self.min_max_scaler(self.state['preposition'][:2], self.min_pos, self.max_pos, -1, 1)
        normalized_linear_velocity = self.min_max_scaler(self.state['linear_velocity'], -30, 30, -1, 1)
        normalized_pose = self.state['pose'][0] / np.pi
        normalized_prepose = self.state['prepose'][0] / np.pi
        normalized_angular_velocity = self.state['angular_velocity'][2] / np.pi
        normalized_angle = self.state['angle'][0] / np.pi
        normalized_target_point = self.min_max_scaler(self.target_point, self.min_pos, self.max_pos, -1, 1)

        obs = [normalized_position[0],
               normalized_position[1],
               normalized_preposition[0],
               normalized_preposition[1],
               normalized_linear_velocity[0],
               normalized_linear_velocity[1],
               normalized_pose,
               normalized_prepose,
               normalized_angular_velocity,
               normalized_target_point[0],
               normalized_target_point[1],
               normalized_angle,
               min_dist]

        return np.array(obs)

    def _compute_reward(self):
        car_vel = self.state['linear_velocity'][:2]
        car_vel_norm = np.linalg.norm(car_vel)

        track_vector = self.target_point - self.current_point
        track_vector_norm = np.linalg.norm(track_vector)

        dot_product = np.dot(track_vector, car_vel)
        angle_radian = np.arccos(
            np.clip(dot_product / (track_vector_norm * car_vel_norm), -1, 1)) if car_vel_norm != 0 else -1
        if angle_radian != -1:
            vxcostheta = car_vel_norm * np.cos(angle_radian)
            vxsintheta = car_vel_norm * np.sin(angle_radian)
        else:
            vxcostheta = 0
            vxsintheta = 0

        reward = ((vxcostheta - vxsintheta) / 3)

        done = self._check_done()
        if self._success:
            reward += 100
        if self.state['collision']:
            done = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        self.client.simPause(False)
        time.sleep(0.1)
        self.client.simPause(True)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        self.timestep += 1
        return obs, reward, done, self.state

    def reset(self, start_index: int = None):
        self._setup_car(start_index)
        time.sleep(0.01)
        self.timestep = 0
        return self._get_obs()

    def close(self):
        self.client.reset()
        self.client.enableApiControl(False)

    def _car_position_init(self, index: int):
        while True:
            row = self.trajectory.iloc[index]
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(row['POS_X'],
                                            row['POS_Y'],
                                            row['POS_Z']),
                            airsim.Quaternionr(row['Q_X'],
                                               row['Q_Y'],
                                               row['Q_Z'],
                                               row['Q_W'])), True)
            self.client.simPause(False)
            time.sleep(0.01)
            self.client.simPause(True)
            car_point = self.client.getCarState().kinematics_estimated.position.to_numpy_array()
            min_distance = np.min(np.linalg.norm((self.xyz_points - car_point), axis=1))
            if min_distance < 1:
                break

    def _check_done(self):
        car_point = self.state['position'][:2]

        distances = np.linalg.norm(self.xy_points - car_point, axis=1)
        min_distance = np.min(distances)
        min_distance_index = distances.argmin()

        if self.state['collision']:
            return True

        if min_distance > 1:
            return True

        if self.timestep >= 499:
            return True

        if min_distance_index > 2100:
            self._success = True
            return True

        self._success = False
        return False

    @staticmethod
    def min_max_scaler(x, x_min, x_max, target_min=0, target_max=1):
        return target_min + (x - x_min) * (target_max - target_min) / (x_max - x_min)
