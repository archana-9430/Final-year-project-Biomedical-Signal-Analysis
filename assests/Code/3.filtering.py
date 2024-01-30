'''
Takes the 10min segments and then process them using a Band pass Butterworth filter
and then puts all the files in a single folder specified by global variable "output_folder"
'''

#~~~~~~~ SOME VARIABLES TO CONSIDER BFORE RUNNING THE SCRIPT~~~~~~~~~~~~~
# folder paths
import imported_files.paths_n_vars as pNv
input_folder = pNv.ten_min_csv_fol
output_folder = pNv.filtered_folder

# filter specifications
filter_order = 4
lower_cutoff = 0.2
higher_cutoff = 4
sampling_freq = pNv.sampling_frequency # in Hertz

# debug / enhancing the processing speed
plot_sig = False # make this false if you don't want to see the original and filtered signal
save = False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pprint import pprint
import os

def butterworth_bp_filter(data_file_path, order, fs, lowcut, highcut, filtered_data_file_path):
    df = pd.read_csv(data_file_path)
    data = df.values.flatten()
    t = np.linspace(0, df.shape[0] - 1, df.shape[0])
    
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
    if save:
        filtered_df.to_csv(filtered_data_file_path , index = False)

    # Plot original and filtered data
    if plot_sig:
        plt.figure()
        plt.plot(t, data, 'b-', label = 'Original Data')
        plt.plot(t, filtered_data, 'r-', linewidth = 2, label = 'Filtered Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(data_file_path[len(input_folder) - 1: ] + ": " + str(order) + ' Order Butterworth Bandpass Filter')
        plt.legend()
        plt.grid(True)
        plt.show()
    
# Get the list of all files and directories
csv_list = os.listdir(input_folder)
print(f"{csv_list}, {len(csv_list)}")

for csv_file in csv_list:
    butterworth_bp_filter(f"{input_folder}\\{csv_file}",
                          filter_order, 
                          sampling_freq, 
                          lower_cutoff, 
                          higher_cutoff, 
                          f"{output_folder}\\{csv_file}")

