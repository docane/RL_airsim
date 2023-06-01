import pandas as pd
import numpy as np
import math

trajectory = pd.read_csv('./data/airsim_rec_5.txt', sep='\t')
x = np.reshape(np.array(trajectory['POS_X'].values, dtype=np.float32), (-1, 1))
y = np.reshape(np.array(trajectory['POS_Y'].values, dtype=np.float32), (-1, 1))
pts = np.reshape(np.array([(x[i], y[i]) for i in range(len(trajectory))]), (-1, 2))
# print(pts[945])
# print(pts[1800])
# print(pts[1750])
# print(pts[1001])
# print(pts[890])
# print(pts[1845])
print(pts[1030])
print(pts[0])
print(pts[1835])
print(pts[1086])
print(pts[975])
print(pts[45])
max_pos = 1000
min_pos = -1000
pts = (pts - min_pos) / (max_pos - min_pos)

# 출발지 부분 중간 지점 인덱스 1800
# 도착지 부분 중간 지점 인덱스 945
# 첫번째 루트의 첫번째 인덱스는 1750
# 첫번째 루트의 마지막 인덱스는 1001
# 두번째 루트의 첫번째 인덱스는 890
# 두번째 루트의 마지막 인덱스는 1845
# [ 892.367 -389.323]
# [-16.282  -15.3034]
# [ 17.9418 -27.0295]
# [ 875.454 -408.735]
# [ 870.209 -387.552]
# [ 8.38397 -5.79998]


# pts_0 = np.concatenate((pts[-100:], pts[]))
# print(pts_0)
# pts_1 = np.concatenate((pts[850:]))
# pts_1 = np.concatenate((pts[-1000:], pts[:130]))

# temp_0 = [-135]
# for i in range(1000):
#     for j in range(temp_0[-1], 1000):
#         if math.sqrt((pts_0[j][0] - pts_0[temp_0[-1]][0]) ** 2 + (pts_0[j][1] - pts_0[temp_0[-1]][1]) ** 2) > 5 / 2000:
#             temp_0.append(j)
#             break
#
# temp_1 = [-1000]
# for i in range(1000):
#     for j in range(temp_1[-1], 130):
#         if math.sqrt((pts_1[j][0] - pts_1[temp_1[-1]][0]) ** 2 + (pts_1[j][1] - pts_1[temp_1[-1]][1]) ** 2) > 5 / 2000:
#             temp_1.append(j)
#             break
#
# randint = np.random.randint(0, len(trajectory))
# if randint < 946:
#     direction = 0
#     start_index = randint
# else:
#     direction = 1
#     start_index = randint - 1885
# start_row = trajectory.iloc(start_index)



# print(trajectory)
# print(x)
# print(y)
# print(pts)
# print(temp)
# print(np.shape(pts))
# print(pts_1.shape)
# print(pts_2.shape)
# print(temp_0)
# print(temp_1)
# print(len(temp_0))
# print(len(temp_1))
# print(randint)
# print(direction)
# print(start_index)

# for randint in range(0, len(trajectory)):
#     if randint < 946:
#         print(randint)
#     else:
#         print(randint - 1885)
