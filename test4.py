import airsim
import numpy as np
import cv2 as cv

client = airsim.CarClient()
client.confirmConnection()

image_request = airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True, False)
response = client.simGetImages([image_request])[0]
img1d = np.array(response.image_data_float, dtype=np.float64)
img1d = np.array(255 / np.maximum(np.ones(img1d.size), img1d), dtype=np.uint8)
img2d = np.reshape(img1d, (response.height, response.width))
img2d = cv.Canny(img2d, 17, 20, 3)
img2d = img2d / 255
image = np.array(cv.resize(img2d, (84, 84), cv.INTER_AREA), dtype=np.float32)
print(image.dtype)
cv.imshow('asdf', image)
cv.waitKey()
print(img2d)

# img1d = cv.cvtColor(img1d, cv.COLOR_RGB2GRAY)
# img1d = img1d / 255
# img2d = np.reshape(img1d, (response.height, response.width))
# img = np.array(cv.resize(img2d, (84, 84), cv.INTER_AREA))
# cv.imshow('asdf', img)
# cv.waitKey()
# print(img.shape)
