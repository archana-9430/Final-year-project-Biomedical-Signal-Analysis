csv_data = "CSV_Data_125Hz\\bidmc03m.csv"

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def butterworth_bp_filter(data_file, order, fs, lowcut, highcut):
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
    filtered_df.to_csv("filtered_125Hz\\bidmc03m.csv", index=False, header=False)

    # Plot original and filtered data
    plt.figure()
    plt.plot(t, data, 'b-', label='Original Data')
    plt.plot(t, filtered_data, 'r-', linewidth = 2, label='Filtered Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(str(order) + ' Order Butterworth Bandpass Filter')
    plt.legend()
    plt.grid(True)
    plt.show()
    
butterworth_bp_filter(csv_data, 4, 125, 0.2, 4)

