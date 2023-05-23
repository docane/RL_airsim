import airsim
import pprint
import numpy
import cv2 as cv


def parse_lidarData(data):
    # reshape array of floats to array of [X,Y,Z]
    points = numpy.array(data.point_cloud, dtype=numpy.dtype('f4'))
    points = numpy.reshape(points, (int(points.shape[0] / 3), 3))

    return points


client = airsim.CarClient()
client.enableApiControl(False)
car_controls = airsim.CarControls()

# while True:
#     state = client.getCarState()
#     s = pprint.pformat(state)
#
#     lidarData = client.getLidarData()
#     if len(lidarData.point_cloud) < 3:
#         print('No points received from Lidar data')
#     else:
#         points = parse_lidarData(lidarData)
#

state = client.getCarState()
s = pprint.pformat(state)

lidardata = client.getLidarData()
points = parse_lidarData(lidardata)
print(points.shape)
cv.imshow(points)