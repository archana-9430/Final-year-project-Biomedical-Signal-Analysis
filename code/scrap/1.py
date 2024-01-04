import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# wfdb.plot.plot_all_records(directory='3000051')

# record = wfdb.rdsamp('3000051\\3000051_0001')
# record = wfdb.rdrecord('New folder\\100001_PPG')
record = wfdb.rdrecord('New folder\\brno-university-of-technology-smartphone-ppg-database-but-ppg-1.0.0\\107001\\107001_PPG')

data = record.p_signal
# print(type(data))
df = pd.DataFrame(data)
print(data)
print(data.shape)
df.to_csv("3000051_0001.csv", index = False)
# df2 = pd.read_csv("3000051_0001.csv")
# print(df2)
transposed = df.T
print(transposed)

x = range(0,150)
plt.plot(x, transposed[0:150])
plt.show()
