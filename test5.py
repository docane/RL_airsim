import airsim
import numpy as np
import cv2 as cv
import math

client = airsim.CarClient()
client.confirmConnection()

image_request = airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True, False)
response = client.simGetImages([image_request])[0]
img1d = np.array(response.image_data_float, dtype=np.float64)
img1d = np.array(255 / np.maximum(np.ones(img1d.size), img1d), dtype=np.uint8)
img2d = np.reshape(img1d, (response.height, response.width))
img2d = cv.Canny(img2d, 17, 20, 3)
lines = cv.HoughLines(img2d, 1, np.pi / 180, 110)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), (int(y0 + 1000 * a)))
        pt2 = (int(x0 - 1000 * (-b)), (int(y0 - 1000 * a)))
        cv.line(img2d, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
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
