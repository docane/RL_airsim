import airsim
import pandas as pd
import numpy as np
import math
import time

client = airsim.CarClient()
client.enableApiControl(False)

trajectory = pd.read_csv('./data/airsim_rec_2.txt', sep='\t')
x = np.reshape(np.array(trajectory['POS_X'].values, dtype=np.float32), (-1, 1))
y = np.reshape(np.array(trajectory['POS_Y'].values, dtype=np.float32), (-1, 1))
max_x = np.max(x)
min_x = np.min(x)
max_y = np.max(y)
min_y = np.min(y)
x = (x - min_x) / (max_x - min_x)
y = (y - min_y) / (max_y - min_y)

state = {
    'preposition': np.zeros(3),
    'position': np.zeros(3),
    'pose': np.zeros(3),
    'prepose': np.zeros(3)
}

# car_state = client.getCarState()
# state['preposition'] = state['position']
# state['prepose'] = state['pose']
#
# state['position'] = car_state.kinematics_estimated.position.to_numpy_array()
# state['pose'] = airsim.to_eularian_angles(car_state.kinematics_estimated.orientation)
#
# state['position'][0] = (state['position'][0] - min_x) / (max_x - min_x)
# state['position'][1] = (state['position'][1] - min_y) / (max_y - min_y)

min_speed = 3
max_speed = 30
beta = 3
thresh_dist = 3.5
pts = [np.array([x[i], y[i]]) for i in range(len(x))]

while True:
    car_state = client.getCarState()
    state['preposition'] = state['position']
    state['prepose'] = state['pose']
    state['position'] = car_state.kinematics_estimated.position.to_numpy_array()
    state['pose'] = airsim.to_eularian_angles(car_state.kinematics_estimated.orientation)
    # state['position'][0] = (state['position'][0] - min_x) / (max_x - min_x)
    # state['position'][1] = (state['position'][1] - min_y) / (max_y - min_y)
    state['linear_velocity'] = car_state.kinematics_estimated.linear_velocity.to_numpy_array()
    state['angular_velocity'] = car_state.kinematics_estimated.angular_velocity.to_numpy_array()

    # print(f'Speed: {car_state.speed}')
    # print(f'Linear Velocity: {state["linear_velocity"]}')
    # print(f'Position: {state["position"]}')
    # print(f'Angular Velocity: {state["angular_velocity"]}')
    #
    # print(f'Linear Veocity Norm: {np.linalg.norm(state["linear_velocity"])}')
    # print(f'Angular Velocity Norm: {np.linalg.norm(state["angular_velocity"])}')

    car_pt = state['position'][:2]
    # print(state['angular_velocity'])
    # print(state['pose'])
    # dist = 10000000

    # for i in range(len(pts) - 1):
    #     dist = min(dist,
    #                np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
    #                               ) / np.linalg.norm(pts[i] - pts[i + 1]), )

    # for i in range(len(pts) - 1):
    #     dist = min(dist,
    #                np.linalg.norm(car_pt[0] - pts[i + 1][0]) - (car_pt[0] - pts[i][0]))
    # print(dist)

    # print(pts[0])

    # reward_dist = math.exp(-beta * dist)
    # reward_speed = ((car_state.speed - min_speed) / (max_speed - min_speed))
    # reward = reward_dist + reward_speed
    # log = f'Distance Reward: {reward_dist}\n'
    # log += f'Speed Reward: {reward_speed}\n'
    # log += f'Total Reward: {reward}\n'
    # print(log)

    ######################################################
    # v1 = state['position'][:2] - state['preposition'][:2]
    # print(v1)

    # dist1 = np.array([math.sqrt(v1[0] ** 2 + v1[1] ** 2)])
    # print(dist1)
    # dist1 = np.linalg.norm(v1)  # 이 표현이 더 깔끔함
    # print(dist1)

    # dist2 = np.array(
    #     [math.sqrt((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) for i in range(len(pts) - 1)])
    # print(dist2)
    # dist2 = np.linalg.norm(pts, axis=1)
    # print(dist2)

    # ip = np.array(
    #     [v1[0] * (pts[i + 1][0] - pts[i][0]) + v1[1] * (pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)])
    # print(ip)
    # ip2 = dist1 * dist2

    # cost = ip / ip2

    # print(f'cos x: {cost}')
    # x = np.arccos(cost)
    # x = np.nanmin(x)
    # print(f'x (radians): {x}')
    # degx = np.degrees(x)
    # print(f'x (degrees): {degx}')
    # print(np.nanmin(degx))
    # for i in range(len(pts) - 1):
    #     v2 = pts[i + 1] - pts[i]

    #######################################
    # temp = np.array(
    #     [math.sqrt(((car_pt[0] - pts[i][0]) ** 2) + ((car_pt[1] - pts[i][1]) ** 2)) for i in range(len(pts))])
    # index = np.argmin(temp)
    # # print(index)
    #
    # temp_x = pts[index + 20][0][0] - pts[index][0][0]
    # temp_y = pts[index + 20][1][0] - pts[index][1][0]
    # v1 = np.array([temp_x, temp_y])
    # # print(car_pt[index + 20])
    # # print(car_pt[index])
    # # print(temp_x)
    # # print(temp_y)
    # # print(v1)
    #
    # temp_x = state['position'][0] - state['preposition'][0]
    # temp_y = state['position'][1] - state['preposition'][1]
    # v2 = np.array([temp_x, temp_y])
    # # print(temp_x)
    # # print(temp_y)
    # # print(v2)
    #
    # dist1 = np.linalg.norm(v1)
    # dist2 = np.linalg.norm(v2)
    #
    # ip = v1[0] * v2[0] + v1[1] * v2[1]
    #
    # ip2 = dist1 * dist2
    #
    # cost = ip / (ip2 + 0.00001)
    # print(cost)
    # theta = math.acos(cost)
    # reward = theta * np.pi

    time.sleep(1)
