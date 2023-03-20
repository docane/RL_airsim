import airsim

client = airsim.CarClient('127.0.0.1', port=41451)
client.enableApiControl(False)