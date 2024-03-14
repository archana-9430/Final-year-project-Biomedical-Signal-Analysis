import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate , correlation_lags

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
shift = int(0.2*1250)
num_windows = int(ppg_df.shape[0]/window_size)
print('number of windows = ',num_windows)

for _,col in ppg_df.items():
    for i in range(num_windows):
        window = col.values[i*shift:i*shift+window_size]
        print('shape of window = ', window.shape)
        print('shape of template_data = ', template_data.shape)
        correlation_values = correlate(window.reshape(-1,1), template_data.reshape(-1,1), mode='same')
        lags = correlation_lags(len(window), len(template_data), mode='same')
        

        # Find the peak in the correlation to determine the appropriate window size
        max_index = np.argmax(correlation_values)
        peak_value = correlation_values[max_index]

        print("Peak value of correlation:", peak_value)
        print("Index of peak:", max_index)

        # Visualize the correlation result
        fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1, figsize=(12 , 6))
        ax_orig.plot(window)
        ax_orig.set_title('Original signal')
        ax_orig.set_xlabel('Sample Number')
        ax_template.plot(template_data)
        ax_template.set_title('Template')
        ax_template.set_xlabel('Sample Number')
        ax_corr.plot(lags, correlation_values, label = 'scipy function')
        ax_corr.set_title('Cross-correlated signal')
        ax_corr.set_xlabel('Lag')
        ax_orig.margins(0, 0.1)
        ax_template.margins(0, 0.1)
        ax_corr.margins(0, 0.1)
        fig.tight_layout()
        plt.legend()
        plt.show()