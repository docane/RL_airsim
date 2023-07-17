import pandas as pd
import datetime as dt

df = pd.read_csv('airsim_rec.txt', sep='\t')
stamps = df['TimeStamp'] / 1000
# print(dt.datetime.fromtimestamp(stamps))
for time in stamps:
    print(dt.datetime.fromtimestamp(time))
#
# print(dt.datetime.fromtimestamp(1676261442986/1000))
# print(dt.datetime.fromtimestamp(1676261443101/1000))
# print(dt.datetime.fromtimestamp(1676261443215/1000))
#
# print(dt.datetime.fromtimestamp(1676261443328/1000))
# print(dt.datetime.fromtimestamp(1676261443440/1000))
# 1676261443556
# print(dt.datetime.fromtimestamp(1676261443556/1000))