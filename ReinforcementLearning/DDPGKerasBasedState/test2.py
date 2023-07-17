import numpy as np
import pandas as pd
import airsim

trajectory = pd.read_csv('data/airsim_rec.txt', sep='\t')

x = np.reshape(np.array(trajectory['POS_X'].values, dtype=np.float32), (-1, 1))
y = np.reshape(np.array(trajectory['POS_Y'].values, dtype=np.float32), (-1, 1))

max_x = np.max(x)
min_x = np.min(x)
max_y = np.max(y)
min_y = np.min(y)
# print(max_x)
# print(min_x)
# print(max_y)
# print(min_y)

norm_x = (x - min_x) / (max_x - min_x)
norm_y = (y - min_y) / (max_y - min_y)

client = airsim.CarClient()
client.enableApiControl(False)


def gaussian(x, mean=0.0, sigma=1.0):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


while True:
    state = client.getCarState()

    car_x, car_y, car_z = state.kinematics_estimated.position.to_numpy_array()
    # print(car_x, car_y, car_z)

    norm_car_x = (car_x - min_x) / (max_x - min_x)
    norm_car_y = (car_y - min_y) / (max_y - min_y)

    distance = np.sqrt(np.square(norm_car_x - norm_x) + np.square(norm_car_y - norm_y))
    # print(distance.min())

    reward = gaussian(distance.min(), sigma=0.001) / 400
    print(reward)

    # print(gaussian(0, sigma=0.001))