import airsim
import numpy as np

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
# a = client.simGetGroundTruthKinematics()
# print(a.position.to_numpy_array())
# print(a.position.x_val)
# print(a.position.y_val)
# print(a.position.z_val)
# b = client.getCarState()
# print(b.kinematics_estimated.position.to_numpy_array())
# temp = client.simGetWorldExtents()
# print(temp)
# print(np.finfo(np.float32).min)
# print(b)
# print(b.kinematics_estimated.orientation)

temp = client.getCarState()
print(temp.kinematics_estimated.position.to_numpy_array())
print(temp.kinematics_estimated.orientation.to_numpy_array())