import numpy as np
import gym
from gym import spaces
import time
import airsim
import pandas as pd
from airsim import Vector3r
import cv2


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
            'pose': np.zeros(3),
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'target_linear_velocity_ratio': np.zeros(2),
            'angle': np.zeros(1),
            'track_distance': np.zeros(1)
        }

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

        self.state_space = spaces.Box(low, high, shape=(10,), dtype=np.float32)
        self.image_space = spaces.Box(low=0, high=1, shape=(28, 28, 1), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({'state': self.state_space,
                                                  'image': self.image_space})

        action_low = np.array([-1, 0], dtype=np.float32)
        action_high = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)

        self.trajectory = pd.read_csv('data/airsim_rec_6.txt', sep='\t')

        pos_x = self.trajectory['POS_X'].values.astype(np.float32)
        pos_y = self.trajectory['POS_Y'].values.astype(np.float32)
        pos_z = self.trajectory['POS_Z'].values.astype(np.float32)
        self.max_pos = 1000.0
        self.min_pos = -1000.0

        self.velocity_divide = 2000.0

        self.xy_points = np.column_stack((pos_x, pos_y))
        self.xyz_points = np.column_stack((pos_x.astype(float),
                                           pos_y.astype(float),
                                           pos_z.astype(float)))

        self.image_request = airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)

        # 5m 단위로 경로 포인트 잡기
        self.route_points_5m = self._capture_route_points_5m()

        self.client.simPlotLineStrip(points=[Vector3r(x, y, z + 0.5) for x, y, z in self.xyz_points],
                                     thickness=30,
                                     is_persistent=True)
        self._success = False

    def _setup_car(self, start_index: int = None):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simPause(True)

        if start_index is None:
            if self._success:
                start_index = 0
            else:
                start_index = np.random.randint(0, len(self.trajectory) - 400)
        self._car_position_init(start_index)

    def __del__(self):
        self.client.reset()

    def _do_action(self, actions):
        self.car_controls.brake = 0
        self.car_controls.throttle = float(actions[1])
        self.car_controls.steering = float(actions[0])

        self.client.setCarControls(self.car_controls)

    def _get_obs(self):
        response = self.client.simGetImages([self.image_request])[0]
        while len(response.image_data_uint8) == 0:
            response = self.client.simGetImages([self.image_request])[0]
        image = self.transform_obs(response)

        self.car_state = self.client.getCarState()
        kin_est = self.car_state.kinematics_estimated

        self.state['position'] = kin_est.position.to_numpy_array()
        self.state['pose'] = np.array(airsim.to_eularian_angles(kin_est.orientation))
        self.state['linear_velocity'] = kin_est.linear_velocity.to_numpy_array()
        self.state['angular_velocity'] = kin_est.angular_velocity.to_numpy_array()
        self.state['collision'] = self.client.simGetCollisionInfo().has_collided

        car_point = self.state['position'][:2]

        dist = np.linalg.norm(self.xy_points - car_point, axis=1)
        min_dist_index = dist.argmin()

        # 5m 단위의 경로 포인트 중에서 가장 가까운 뒤에 있는 경로 포인트 구하기
        min_dist_5m_index = 0
        for i in range(len(self.route_points_5m)):
            if min_dist_index < self.route_points_5m[i]:
                min_dist_5m_index = i - 1
                break

        # 앞, 뒤에서 가장 가까운 경로 포인트 잡기
        self.route_point = [index for index in range(min_dist_5m_index, min_dist_5m_index + 2)]

        target_point = self.xy_points[self.route_points_5m[self.route_point[1]]]
        target_linear_velocity = target_point - car_point
        target_linear_velocity_norm = np.linalg.norm(target_linear_velocity)
        target_linear_velocity_ratio = target_linear_velocity / target_linear_velocity_norm

        self.state['target_linear_velocity_ratio'][0] = target_linear_velocity_ratio[0]
        self.state['target_linear_velocity_ratio'][1] = target_linear_velocity_ratio[1]

        first_target_point = self.xy_points[self.route_points_5m[self.route_point[0]]]
        second_target_point = self.xy_points[self.route_points_5m[self.route_point[1]]]

        car_vel = self.state['linear_velocity'][:2]
        car_vel_norm = np.linalg.norm(car_vel)

        track_vector = second_target_point - first_target_point
        track_vector_norm = np.linalg.norm(track_vector)
        target_dir_vec = track_vector / track_vector_norm

        car_dir_vec = car_vel / car_vel_norm if car_vel_norm != 0 else car_vel * 0
        ip = np.clip(car_dir_vec[0] * target_dir_vec[1] - car_dir_vec[1] * target_dir_vec[0], -1, 1)
        theta = np.arcsin(ip)
        self.state['angle'][0] = theta

        min_dist = min(dist)
        self.state['track_distance'][0] = min_dist

        normalized_position = self.min_max_scaler(self.state['position'][:2], self.min_pos, self.max_pos, -1, 1)
        normalized_pose = self.state['pose'][0] / np.pi
        normalized_linear_velocity = self.min_max_scaler(self.state['linear_velocity'], -30, 30, -1, 1)
        normalized_angular_velocity = self.state['angular_velocity'][2] / np.pi
        normalized_angle = self.state['angle'][0] / np.pi

        obs = [normalized_position[0],
               normalized_position[1],
               normalized_pose,
               normalized_linear_velocity[0],
               normalized_linear_velocity[1],
               normalized_angular_velocity,
               target_linear_velocity_ratio[0],
               target_linear_velocity_ratio[1],
               normalized_angle,
               min_dist]

        return {'state': np.array(obs),
                'image': image}

    def _compute_reward(self):
        car_pt = self.state['position'][:2]
        car_vel = self.state['linear_velocity'][:2]
        car_vel_norm = np.linalg.norm(car_vel)

        first_target_point = self.xy_points[self.route_points_5m[self.route_point[0]]]
        second_target_point = self.xy_points[self.route_points_5m[self.route_point[1]]]
        track_vector = second_target_point - first_target_point
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

        trackpos = min(np.linalg.norm(self.xy_points - car_pt, axis=1))
        vxtrackpos = trackpos * car_vel_norm

        reward = vxcostheta - vxsintheta - trackpos - vxtrackpos / 10

        done = self._check_done()
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
        return obs, reward, done, self.state

    def reset(self, start_index: int = None):
        self._setup_car(start_index)
        time.sleep(0.01)
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
            min_distance = min(np.linalg.norm((self.xyz_points - car_point), axis=1))
            if min_distance < 1:
                break

    def _check_done(self):
        car_point = self.state['position'][:2]

        distances = np.linalg.norm(self.xy_points - car_point, axis=1)
        min_distance_index = distances.argmin()

        if min_distance_index > 770:
            self._success = True
            return True
        else:
            self._success = False
            return False

    def _capture_route_points_5m(self):
        route_points_5m = [0]
        for i in range(1, len(self.trajectory)):
            distance = np.linalg.norm(self.xy_points[i] - self.xy_points[route_points_5m[-1]])
            if distance > (5 / 2000):
                route_points_5m.append(i)
        return np.array(route_points_5m)

    def transform_obs(self, response):
        image1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        image = np.reshape(image1d, (response.height, response.width, 3))
        red_pixel = self.filter_red_color(image)
        red_pixel = self.limit_region(red_pixel)
        image = np.expand_dims(cv2.resize(red_pixel, (28, 28), cv2.INTER_AREA), axis=-1)
        image = image / 255
        return image

    @staticmethod
    def gaussian(x, mean=0.0, sigma=1.0):
        return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def min_max_scaler(x, x_min, x_max, target_min=0, target_max=1):
        return target_min + (x - x_min) * (target_max - target_min) / (x_max - x_min)

    @staticmethod
    def compute_curvature(x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** (3 / 2))
        return curvature

    @staticmethod
    def filter_red_color(image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_line_1 = cv2.inRange(image_hsv, (0, 150, 200), (20, 255, 255))
        red_line_2 = cv2.inRange(image_hsv, (160, 150, 200), (179, 255, 255))
        red_line = red_line_1 + red_line_2
        return red_line

    @staticmethod
    def limit_region(image):
        height, _ = image.shape
        new_height = int(height * 0.4)
        image = image[new_height:, :]
        return image
