import cv2 as cv
import numpy as np
import math


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅
    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지
    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv.fillPoly(mask, vertices, color)
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv.bitwise_and(img, mask)
    return ROI_image


file = 'drive.mp4'
cap = cv.VideoCapture(file)
Nframe = 0  # frame 수

while cap.isOpened():
    ret, frame = cap.read()

    if ret:  # 비디오 프레임을 읽기 성공했으면 진행
        frame = cv.resize(frame, (1000, 562))
    else:
        break

    Nframe += 1
    origin = np.copy(frame)
    origin = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
    origin = cv.GaussianBlur(origin, (0, 0), 1)
    origin = cv.Canny(origin, 17, 20, 3)
    point = np.array([[400, 200], [600, 200], [1000, 500], [0, 500]])
    origin = region_of_interest(origin, [point])
    cv.imshow('test', origin)
    lines = cv.HoughLines(origin, 1, np.pi / 180, 250)
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
            cv.line(frame, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow('original', origin)  # 원본영상
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xff == ord('q'):  # 'q'누르면 영상 종료
        break

print("Number of Frame: ", Nframe)  # 영상의 frame 수 출력

cap.release()
cv.destroyAllWindows()
