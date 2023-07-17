import airsim
import cv2
import numpy as np
import math

client = airsim.CarClient()
client.confirmConnection()

while True:
    image_request = airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)
    response = client.simGetImages([image_request])[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img2d = np.reshape(img1d, (response.height, response.width, 3))
    img_hsv = cv2.cvtColor(img2d, cv2.COLOR_RGB2HSV)
    yellow_line = cv2.inRange(img_hsv, (97, 0, 0), (100, 255, 255))
    img2d = cv2.medianBlur(yellow_line, 3)
    # yellow_line = cv2.inRange(img_hsv, (30, 160, 180), (110, 220, 240))
    # img2d = cv2.inRange(img2d, (80, 130, 130), (190, 200, 200))

    # img_test = img_hsv[130:, 125:130, 2]

    # img2d = cv2.cvtColor(img2d, cv2.COLOR_RGB2GRAY)
    # img2d = cv2.GaussianBlur(img2d, (3, 3), 0)
    # img2d = cv2.Canny(yellow_line, 50, 100)
    # img1d = np.array(response.image_data_float, dtype=np.float64)
    # img1d = np.array(255 / np.maximum(np.ones(img1d.size), img1d), dtype=np.uint8)
    # img2d = np.reshape(img1d, (response.height, response.width, 3))
    # img2d = cv2.Canny(img2d, 17, 20, 3)
    # lines = cv2.HoughLinesP(yellow_line, 1, np.pi / 180, 100, np.array([]), minLineLength=250, maxLineGap=50)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(yellow_line, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imshow('test', img2d)
    # cv2.imshow('test1', img_hsv)
    # cv2.imshow('test2', yellow_line)
    # cv2.imshow('test3', img_test)
    cv2.waitKey(1)
