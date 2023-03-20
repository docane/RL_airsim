import time
import numpy as np
import cv2 as cv
import airsim
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

MODEL_PATH = './models_2023_02_13_13_58_20/fresh_models/model_model.16-0.0171299.h5'
model = load_model(MODEL_PATH)
client = airsim.CarClient()
# client.confirmConnection()
client.enableApiControl(True)

car_controls = airsim.CarControls()
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0
client.setCarControls(car_controls)

image_buf = np.zeros((1, 84, 84, 3))


def get_image():
    """
    Get image from AirSim client
    """
    image_response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgb = image1d.reshape((image_response.height, image_response.width, 3))
    return cv.resize(image_rgb, (84, 84), cv.INTER_AREA)


trajectory = pd.read_csv('../ddpg_keras_based_state/data/airsim_rec.txt', sep='\t')
rand = np.random.randint(0, len(trajectory))
randrow = trajectory.iloc[rand]
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(randrow['POS_X'],
                                                     randrow['POS_Y'],
                                                     randrow['POS_Z']),
                                     airsim.Quaternionr(randrow['Q_X'],
                                                        randrow['Q_Y'],
                                                        randrow['Q_Z'],
                                                        randrow['Q_W'])), True)

while True:
    trajectory = pd.read_csv('../ddpg_keras_based_state/data/airsim_rec.txt', sep='\t')
    rand = np.random.randint(0, len(trajectory))
    randrow = trajectory.iloc[rand]
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(randrow['POS_X'],
                                                         randrow['POS_Y'],
                                                         randrow['POS_Z']),
                                         airsim.Quaternionr(randrow['Q_X'],
                                                            randrow['Q_Y'],
                                                            randrow['Q_Z'],
                                                            randrow['Q_W'])), True)

    done = False
    while not done:
        # Update throttle value according to steering angle
        # if abs(car_controls.steering) <= 1.0:
        #     car_controls.throttle = 0.4 - (0.4 * abs(car_controls.steering))
        # else:
        #     car_controls.throttle = 0.2
        car_controls.throttle = 0.5

        image_buf[0] = get_image()
        image_buf[0] /= 255  # Normalization

        start_time = time.time()

        # Prediction
        model_output = model.predict([image_buf])

        end_time = time.time()
        received_output = model_output[0][0]

        # Rescale prediction to [-1,1] and factor by 0.82 for drive smoothness
        car_controls.steering = round((0.82 * (float((model_output[0][0] * 2.0) - 1))), 2)

        # Print progress
        print('Sending steering = {0}, throttle = {1}, prediction time = {2}'.format(received_output,
                                                                                     car_controls.throttle,
                                                                                     str(end_time - start_time)))

        # Update next car state
        client.setCarControls(car_controls)
        if client.simGetCollisionInfo().has_collided:
            done = True
        # Wait a bit between iterations
        # time.sleep(0.05)
