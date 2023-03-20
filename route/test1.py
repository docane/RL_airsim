import airsim
import time

client = airsim.CarClient()
while True:
    state = client.getCarState()
    print(state.speed)
    print(state.kinematics_estimated.position.x_val)
    print(state.kinematics_estimated.position.y_val)
    client.setCarControls()

    time.sleep(1)