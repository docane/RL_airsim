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

pts = [np.array([x[i], y[i]]) for i in range(len(x))]

# dist = [math.sqrt((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) for i in range(len(pts) - 1)]
temp = [0]
print([math.sqrt((pts[10][0] - pts[0][0]) ** 2 + (pts[10][1] - pts[0][1]) ** 2)])
for i in range(len(trajectory)):
    for j in range(temp[-1], len(trajectory)):
        if math.sqrt((pts[j][0] - pts[temp[-1]][0]) ** 2 + (pts[j][1] - pts[temp[-1]][1]) ** 2) > 5:
            temp.append(j)
            break
print(temp)
print(x[temp])
print(y[temp])

while True:
    car_state = client.getCarState()
    car_pt = car_state.kinematics_estimated.position.to_numpy_array()
    min_dist_index = np.array(
        [math.sqrt(((car_pt[0] - pts[i][0]) ** 2) + ((car_pt[1] - pts[i][1]) ** 2)) for i in range(len(pts))])
    # temp = np.array(
    #     [math.sqrt(((car_pt[0] - pts[i][0]) ** 2) + ((car_pt[1] - pts[i][1]) ** 2)) for i in range(len(pts))])
    # index = np.argmin(temp)
    # # print(index)
