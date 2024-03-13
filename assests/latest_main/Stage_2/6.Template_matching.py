import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Load the template and PPG data from CSV files
template_df = pd.read_csv('Templates/template_0.csv')
ppg_df = pd.read_csv('partly_corrupted.csv',skiprows=1)

# Assuming the template and PPG data are in 'value' column of the DataFrame, adjust as per your actual data structure
template_data = template_df.values
ppg_data = ppg_df.values

# Visualize the template and PPG data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(template_data)
plt.title('Template Data')
plt.xlabel('Index')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
plt.plot(ppg_data)
plt.title('PPG Data')
plt.xlabel('Index')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Calculate the cross-correlation between the template and a portion of the PPG data
# You may adjust the portion of data used here for your analysis
window_size = len(template_data)
shift = int(0.2*125)
num_windows = int(ppg_df.shape[0]/window_size)
print(num_windows)
for _,col in ppg_df.items():
    for i in range(num_windows): 
        correlation = correlate(col.values[i*shift:i*shift+window_size].reshape(-1,1), template_data, mode='full')

        # Find the peak in the correlation to determine the appropriate window size
        max_index = np.argmax(correlation)
        peak_value = correlation[max_index]

        print("Peak value of correlation:", peak_value)
        print("Index of peak:", max_index)

        # Visualize the correlation result
        plt.figure(figsize=(8, 4))
        plt.plot(correlation)
        plt.title('Cross-correlation Result')
        plt.xlabel('Index')
        plt.ylabel('Correlation Value')
        plt.axvline(x=max_index, color='r', linestyle='--', label='Peak')
        plt.legend()
        plt.show()
