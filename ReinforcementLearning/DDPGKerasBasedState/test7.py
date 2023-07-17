import airsim
import pandas as pd
import numpy as np
import math
import time

client = airsim.CarClient()
client.enableApiControl(False)

trajectory = pd.read_csv('data/airsim_rec_2.txt', sep='\t')
x = np.reshape(np.array(trajectory['POS_X'].values, dtype=np.float32), (-1, 1))
y = np.reshape(np.array(trajectory['POS_Y'].values, dtype=np.float32), (-1, 1))

rand = 0
randrow = trajectory.iloc[rand]
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(randrow['POS_X'],
                                                     randrow['POS_Y'],
                                                     randrow['POS_Z']),
                                     airsim.Quaternionr(randrow['Q_X'],
                                                        randrow['Q_Y'],
                                                        randrow['Q_Z'],
                                                        randrow['Q_W'])), True)

pts = [np.array([x[i], y[i]]) for i in range(len(x))]

# dist = [math.sqrt((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) for i in range(len(pts) - 1)]
temp = [0]
# print([math.sqrt((pts[10][0] - pts[0][0]) ** 2 + (pts[10][1] - pts[0][1]) ** 2)])
for i in range(len(trajectory)):
    for j in range(temp[-1], len(trajectory)):
        if math.sqrt((pts[j][0] - pts[temp[-1]][0]) ** 2 + (pts[j][1] - pts[temp[-1]][1]) ** 2) > 5:
            temp.append(j)
            break
print(temp)
# print(x[temp])
# print(y[temp])

while True:
    car_state = client.getCarState()
    car_pt = car_state.kinematics_estimated.position.to_numpy_array()
    dist = np.array(
        [math.sqrt(((car_pt[0] - pts[i][0]) ** 2) + ((car_pt[1] - pts[i][1]) ** 2)) for i in range(len(pts))])
    min_dist_index = np.argmin(dist)
    # print(min_dist_index)
    min_dist_temp_index = 0
    for i in range(len(temp)):
        if min_dist_index < temp[i]:
            min_dist_temp_index = i
            break
    # print(min_dist_temp_index)
    route_point = [index for index in range(min_dist_temp_index, min_dist_temp_index + 5)]
    # print(route_point)
    # print(car_pt[0], car_pt[1])
    # print(x[temp[route_point[1]]][0], y[temp[route_point[1]]][0])
    target_point = np.array([(x[temp[route_point[1]]][0] + car_pt[0]) / 2, (y[temp[route_point[1]]][0] + car_pt[1]) / 2])
    # print(target_point)
    v1 = target_point - car_pt[:2]
    v1_norm = np.linalg.norm(v1)
    print(v1_norm)
    v2 = v1 / v1_norm * 5
    print(v2)
    print(np.linalg.norm(v2))
    # print(v1)
    # print(v1_norm)
    reward = np.linalg.norm(v2 - car_state.kinematics_estimated.linear_velocity.to_numpy_array()[:2])
    # print(reward)

    time.sleep(0.5)
