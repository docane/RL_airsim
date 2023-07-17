import airsim

client = airsim.CarClient()
client.enableApiControl(True)
print(client.ping())