import airsim
import time

client = airsim.CarClient()
position_list = []
speed_list = []
# while True:
#     car_state = client.getCarState()
#     speed_list.append(car_state.speed)
#     position_list.append([car_state.kinematics_estimated.position.x_val, car_state.kinematics_estimated.position.y_val])
#     print(speed_list)
#     print(position_list)
#     client.reset()
