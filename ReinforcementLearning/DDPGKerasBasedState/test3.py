import airsim

client = airsim.CarClient('127.0.0.1', 41451)
client.enableApiControl(False)

while True:
    print(airsim.to_eularian_angles(client.simGetVehiclePose().orientation)[2])
