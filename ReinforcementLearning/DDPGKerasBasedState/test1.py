import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.optimizers import Adam

df = pd.read_csv('data/airsim_rec.txt', sep='\t')
# print(df)

df = df[['POS_X', 'POS_Y']]
max_x = df['POS_X'].max()
min_x = df['POS_X'].min()
max_y = df['POS_Y'].max()
min_y = df['POS_Y'].min()
x = np.reshape(np.array(df['POS_X'].values, dtype=np.float32), (-1, 1))
y = np.reshape(np.array(df['POS_Y'].values, dtype=np.float32), (-1, 1))
x = (x - min_x) / (max_x - min_x)
y = (y - min_y) / (max_y - min_y)
# print(df['POS_X'].values)
# print(x)
print(max_x)
print(min_x)
print(max_y)
print(min_y)
print(x)
print(y)

# model = keras.Sequential([
#     Dense(units=100, activation='relu'),
#     Dense(units=1)
# ])
#
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
# model.build(input_shape=(None, 1))
# # model.summary()
#
# model.fit(x, y, epochs=100)
print(np.sqrt(np.square(x - 0) + np.square(y - 0)))
