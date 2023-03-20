import airsim
import numpy as np
import time

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

car_controls = airsim.CarControls()
state = {
    'position': np.zeros(3),
    'collision': False
}
car_state = None

while True:
    car_controls.throttle = 1
    car_controls.brake = 0
    client.setCarControls(car_controls)
    car_state = client.getCarState()
    state['position'] = car_state.kinematics_estimated.position
    print(state['position'])
    state['collision'] = client.simGetCollisionInfo().has_collided
    if state['collision']:
        client.reset()
    time.sleep(3)
