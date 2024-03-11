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
        print(df)
        annotation = df.iloc[0]
        print(annotation)
        # df.drop(index=0,inplace=True, axis=0)
        # data = df.values.flatten()
        ppg_values = df.values[0:]
        print(ppg_values)
        filtered_data = filtfilt(self.b, self.a, ppg_values , axis = 0)
        print("shape of filtered data = ",filtered_data.shape)
        anno_filt = np.insert(filtered_data,0,values=annotation.values , axis = 1)
        merged_df = pd.DataFrame(data = anno_filt , columns = df.columns)
        merged_df.to_csv(filtered_data_file_path , index = False)
        
        
        
    # def butterworth_bp_filter(self, data_file_path, filtered_data_file_path):
    #     df = pd.read_csv(data_file_path)
    #     annotation = df.iloc[0]
    #     # df.drop(index=0,inplace=True, axis=0)
    #     # data = df.values.flatten()
    #     merged_df = pd.DataFrame()
        
    #     for _,col in df.items():
    #         # Apply the filter to the data using filtfilt for zero-phase filtering
    #         filtered_data = filtfilt(self.b, self.a, col.values[0:])
            
    #         # annotated_filtered_data = np.insert(filtered_data,col.values[0],0)
    #         # Create a DataFrame for the filtered data
    #         filtered_df = pd.DataFrame({col.name : filtered_data})
            
    #         merged_df = pd.concat([merged_df, filtered_df], axis=1)
    #         # # Plot original and filtered data
    #         # if plot_sig:
    #         #     plt.figure()
    #         #     t = np.linspace(0, df.shape[0] - 1, df.shape[0])        
    #         #     plt.plot(t, col.values, 'b-', label = 'Original Data')
    #         #     plt.plot(t, filtered_data, 'r-', linewidth = 2, label = 'Filtered Data')
    #         #     plt.xlabel('Time (s)')
    #         #     plt.ylabel('Amplitude')
    #         #     plt.title(data_file_path[len(input_path) - 1: ] + ": " + str(self.order) + ' Order Butterworth Bandpass Filter')
    #         #     plt.legend()
    #         #     plt.grid(True)
    #         #     plt.show()
 
    #     insert_index = 0
    #         # Save the filtered data to a new CSV file
    #     df_before = merged_df.iloc[:insert_index]
    #     df_after = merged_df.iloc[insert_index:]

    #     # Concatenate the two parts with the new row in between
    #     df = pd.concat([df_before, pd.DataFrame([annotation.values]), df_after], ignore_index=True)
    #     df.iloc[0] = annotation.values
        
    #     df.to_csv(filtered_data_file_path , index = False)

           
        

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

