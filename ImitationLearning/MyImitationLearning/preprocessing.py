import pandas as pd

df = pd.read_csv('raw_data/2023-02-09-19-45-37/airsim_rec.txt', sep='\t')
print(df.shape)
print(df)