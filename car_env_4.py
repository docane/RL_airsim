import numpy as np
import gym
from gym import spaces
import time
import airsim
import cv2 as cv
import math


class AirSimEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, image_shape):
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
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
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            'position': np.zeros(3),
            'prev_position': np.zeros(3),
            'pose': None,
            'prev_pose': None,
            'collision': False
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

        self.image_request = airsim.ImageRequest(
            '0', airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 0.5
        self.car_controls.steering = float(action)

        self.car.setCarControls(self.car_controls)
        time.sleep(0.1)

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float64)
        img1d = np.array(255 / np.maximum(np.ones(img1d.size), img1d), dtype=np.uint8)
        img2d = np.reshape(img1d, (response.height, response.width))
        img2d = cv.Canny(img2d, 17, 20, 3)
        lines = cv.HoughLines(img2d, 1, np.pi / 180, 150)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), (int(y0 + 1000 * a)))
                pt2 = (int(x0 - 1000 * (-b)), (int(y0 - 1000 * a)))
                cv.line(img2d, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
        img2d = img2d / 255
        image = np.array(cv.resize(img2d, (84, 84), cv.INTER_AREA), dtype=np.float32)
        return image

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()

        self.state['prev_pose'] = self.state['pose']
        self.state['pose'] = self.car_state.kinematics_estimated.position
        self.state['collision'] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        reward = self.car_state.speed / 100
        done = 0
        if self.state['collision']:
            reward -= 5
            done = 1
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        return self._get_obs()
