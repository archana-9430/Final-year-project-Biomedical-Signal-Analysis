'''
Takes the 10min segments and then process them using a Band pass Butterworth filter
and then puts all the files in a single folder specified by global variable "output_folder"
'''

#~~~~~~~ SOME VARIABLES TO CONSIDER BFORE RUNNING THE SCRIPT~~~~~~~~~~~~~
# folder paths
from imported_files.paths_n_vars import  ten_min_csv_fol, filtered_folder, sampling_frequency

input_folder = ten_min_csv_fol
output_folder = filtered_folder

# filter specifications
filter_order = 4
lower_cutoff = 0.2
higher_cutoff = 4
sampling_freq = sampling_frequency # in Hertz

# debug / enhancing the processing speed
plot_sig = False # make this false if you don't want to see the original and filtered signal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os

class Filter():
    def __init__(self ,order, fs, lowcut, highcut) -> None:
        # Calculate Nyquist frequency
        nyquist = 0.5 * fs
        # Normalize cutoff frequencies
        low = lowcut / nyquist
        high = highcut / nyquist

        # Design a Butterworth bandpass filter
        self.b, self.a = butter(order, [low, high], btype='band')

    def butterworth_bp_filter(self, data_file_path, filtered_data_file_path):
        df = pd.read_csv(data_file_path)
        data = df.values.flatten()
        t = np.linspace(0, df.shape[0] - 1, df.shape[0])
        
        # Apply the filter to the data using filtfilt for zero-phase filtering
        filtered_data = filtfilt(self.b, self.a, data)
        
        # Create a DataFrame for the filtered data
        filtered_df = pd.DataFrame({'Filtered_Signal': filtered_data})

        # Save the filtered data to a new CSV file
        filtered_df.to_csv(filtered_data_file_path , index = False)

        # Plot original and filtered data
        if plot_sig:
            plt.figure()
            plt.plot(t, data, 'b-', label = 'Original Data')
            plt.plot(t, filtered_data, 'r-', linewidth = 2, label = 'Filtered Data')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(data_file_path[len(input_folder) - 1: ] + ": " + str(self.order) + ' Order Butterworth Bandpass Filter')
            plt.legend()
            plt.grid(True)
            plt.show()
    

def _main_filtering():
    # Get the list of all files and directories
    dir_list = os.listdir(input_folder)
    
    # process only the csv files
    csv_list = [file for file in dir_list if file.endswith('.csv')]
    print(f"{csv_list}, {len(csv_list)}")

    ppg_filter = Filter(filter_order, 
                            sampling_freq, 
                            lower_cutoff, 
                            higher_cutoff)
    for csv_file in csv_list:
        ppg_filter.butterworth_bp_filter(f"{input_folder}\\{csv_file}", 
                            f"{output_folder}\\{csv_file}")


if __name__ == "__main__":
    import time
    s = time.perf_counter()
    
    _main_filtering()
    
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

