import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

df1 = pd.read_csv("bidmc03m.csv")
df2 = pd.read_csv("filtered_data.csv")
df3 = pd.read_csv("bidmc03m_py_filtered.csv")
t = np.linspace(0, df1.shape[0] - 1, df1.shape[0])


plt.figure()
plt.plot(t, df1, 'b-', label = 'Original Data')
plt.plot(t, df3, 'g-', linewidth = 2, label = 'Py Filtered Data')
plt.plot(t, df2, 'r-', linewidth = 2, label = 'Filtered Data')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()