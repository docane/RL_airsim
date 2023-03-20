import os
import cv2 as cv
import pandas as pd

RAW_DATA_DIR = './raw_data/'
PREPROCESSING_DATA_DIR = './preprocessing_data/'

ALL_DATA = True
if ALL_DATA:
    data_folders = [name for name in os.listdir(RAW_DATA_DIR)]
else:
    data_folders = ['2023-02-13-13-10-42']
# print(data_folders)

full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in data_folders]
full_path_preprocessing_folders = [os.path.join(PREPROCESSING_DATA_DIR, f) for f in data_folders]
# print(full_path_raw_folders)
# print(full_path_preprocessing_folders)

for i in range(len(full_path_raw_folders)):
    current_df = pd.read_csv(os.path.join(full_path_raw_folders[i], 'airsim_rec.txt'), sep='\t')
    for j in range(1, current_df.shape[0] - 1):
        image_filepath = os.path.join(os.path.join(full_path_raw_folders[i], 'images'),
                                      current_df.iloc[j]['ImageFile']).replace('\\', '/')
        image_write_filepath = os.path.join(os.path.join(full_path_preprocessing_folders[i], 'images'),
                                            current_df.iloc[j]['ImageFile']).replace('\\', '/')
        # print(image_write_filepath)
        # if not os.path.exists(os.path.dirname(os.path.join(full_path_preprocessing_folders[i], 'images').replace('\\', '/'))):
        #     try:
        #         os.makedirs(os.path.dirname(os.path.join(full_path_preprocessing_folders[i], 'images').replace('\\', '/')))
        #     except OSError as exc:
        #         if exc.errno != errno.EEXIST:
        #             raise

        img = cv.imread(image_filepath)
        img = cv.resize(img, (84, 84), cv.INTER_AREA)
        cv.imwrite(image_write_filepath, img)
