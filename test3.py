import PIL.Image
import airsim
import numpy as np
from PIL import Image
import cv2 as cv

client = airsim.CarClient()
client.confirmConnection()

image_request = airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)
response = client.simGetImages([image_request])[0]
img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
img1d = img1d / 255
img2d = np.reshape(img1d, (response.height, response.width, 3))
image = np.array(cv.resize(img2d, (84, 84), cv.INTER_AREA))
cv.imshow('asdf', image)
cv.waitKey()
print(image.shape)
