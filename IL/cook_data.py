import os
import Cooking

chunk_size = 32
train_eval_test_split = [0.8, 0.2, 0.0]
RAW_DATA_DIR = './preprocessing_data/'
COOKED_DATA_DIR = './preprocessing_image_cooked_data/'
COOK_ALL_DATA = True
if COOK_ALL_DATA:
    data_folders = [name for name in os.listdir(RAW_DATA_DIR)]
else:
    data_folders = []
full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in data_folders]
Cooking.cook(full_path_raw_folders, COOKED_DATA_DIR, train_eval_test_split, chunk_size)
