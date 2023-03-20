import pandas as pd
import numpy as np

trajectory = pd.read_csv('./data/airsim_rec_1.txt', sep='\t')
rand = np.random.randint(0, len(trajectory))
print(len(trajectory))
x = np.reshape(np.array(trajectory['POS_X'].values, dtype=np.float32), (-1, 1))
y = np.reshape(np.array(trajectory['POS_Y'].values, dtype=np.float32), (-1, 1))
print(x)
print(y)