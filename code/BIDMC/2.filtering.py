csv_path = "CSV_Data_125Hz\\"
filtered_path = "filtered_125Hz\\"
path = "CSV_Data_125Hz"

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os

def butterworth_bp_filter(data_file, order, fs, lowcut, highcut, filtered_data_file):
    df = pd.read_csv(data_file, header=None)
    data = df.values.flatten()
    t = np.linspace(0, len(data) - 1, len(data))
    
    # Calculate Nyquist frequency
    nyquist = 0.5 * fs
    # Normalize cutoff frequencies
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design a Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to the data using filtfilt for zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    
    # Create a DataFrame for the filtered data
    filtered_df = pd.DataFrame({'Filtered_Signal': filtered_data})

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(filtered_data_file , index=False, header=False)

    # Plot original and filtered data
    plt.figure()
    plt.plot(t, data, 'b-', label='Original Data')
    plt.plot(t, filtered_data, 'r-', linewidth = 2, label='Filtered Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(data_file[len(path) - 1: ] + ": " + str(order) + ' Order Butterworth Bandpass Filter')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Get the list of all files and directories
dir_list = os.listdir(path)
print(dir_list)

for i in range(len(dir_list)):
    butterworth_bp_filter(csv_path + dir_list[i], 4, 125, 0.2, 4, filtered_path + dir_list[i])

