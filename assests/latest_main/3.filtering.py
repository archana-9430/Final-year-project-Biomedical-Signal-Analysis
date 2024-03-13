'''
Takes the 10min segments and then process them using a Band pass Butterworth filter
and then puts all the files in a single folder specified by global variable "output_folder"
'''

#~~~~~~~ SOME VARIABLES TO CONSIDER BFORE RUNNING THE SCRIPT~~~~~~~~~~~~~
# folder paths
from imported_files.paths_n_vars import  filtered_merged, annotated_merged, sampling_frequency

input_path = annotated_merged
output_folder = filtered_merged

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
        annotation = df.iloc[0]
        ppg_values = df.values[1:]
        filtered_data = filtfilt(self.b, self.a, ppg_values , axis = 0)
        anno_filt = np.insert(filtered_data,0,values=annotation.values , axis = 0)
        merged_df = pd.DataFrame(data = anno_filt , columns = df.columns)
        merged_df.to_csv(filtered_data_file_path , index = False)
               

def _main_filtering():

    ppg_filter = Filter(filter_order, 
                            sampling_freq, 
                            lower_cutoff, 
                            higher_cutoff)
    ppg_filter.butterworth_bp_filter(input_path, output_folder)


if __name__ == "__main__":
    import time
    s = time.perf_counter()
    
    _main_filtering()
    
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

