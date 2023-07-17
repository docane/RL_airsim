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
temp = [0]
for i in range(len(trajectory)):
    for j in range(temp[-1], len(trajectory)):
        if math.sqrt((pts[j][0] - pts[temp[-1]][0]) ** 2 + (pts[j][1] - pts[temp[-1]][1]) ** 2) > 5:
            temp.append(j)
            break
print(temp)

while True:
    car_state = client.getCarState()
    car_pt = car_state.kinematics_estimated.position.to_numpy_array()

    dist = np.array([math.sqrt(((car_pt[0] - pts[i][0]) ** 2) + ((car_pt[1] - pts[i][1]) ** 2)) for i in
                     range(len(pts))])
    min_dist_index = np.argmin(dist)
    min_dist_temp_index = 0
    for i in range(len(temp)):
        if min_dist_index < temp[i]:
            min_dist_temp_index = i
            break
    route_point = [index for index in range(min_dist_temp_index, min_dist_temp_index + 5)]
    first_target_point = np.array(
        [x[temp[route_point[0]]][0],
         y[temp[route_point[0]]][0]]
    )
    second_target_point = np.array(
        [x[temp[route_point[1]]][0],
         y[temp[route_point[1]]][0]]
    )

    time.sleep(0.5)

    track_direction = math.atan2(second_target_point[1] - first_target_point[1],
                                 second_target_point[0] - first_target_point[0])
    print('Track Direction:', track_direction)
    heading = np.array(airsim.to_eularian_angles(car_state.kinematics_estimated.orientation))[2]
    print('Heading:', heading)
    heading_difference = abs(track_direction - heading)
    if heading_difference > math.pi:
        heading_difference = 2 * math.pi - heading_difference
    print('Heading Difference:', heading_difference)
    speed = np.linalg.norm(car_state.kinematics_estimated.linear_velocity.to_numpy_array())
    print('Speed:', speed)
    min_dist = np.min(dist)
    print('Minimum Distance:', min_dist)
    reward = speed * math.cos(heading_difference) - 0.7 * speed * math.sin(heading_difference) - 0.2 * abs(
        min_dist) - 0.3 * speed * abs(min_dist)
    print('Reward:', reward)

    # time.sleep(0.5)
