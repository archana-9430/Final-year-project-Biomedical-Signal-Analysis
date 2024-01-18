'''This file should make all the csv file to contain 8min long datam i.e, it should have 60000 PPG signal values.'''

csv_data_fol = "assests\Code\Csv_data"
uniform_csv_fol = "assests\Code\\uniform_csv_data"

'''
The script takes csv files from "Csv_data" folder and then makes the signal of 10 min 
and save the output csv to folder "uniformed_csv_data".
For that purpose it assumes sampling frequency as specified in the variable "sampling_freq"
'''

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
sampling_freq = 125 # sampling freq in HERTZ

window_len_sec = 10*60 # in seconds
stride_len_sec = window_len_sec
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_signal(x : list ,y : list , x_label = None , y_label = None , title = None):
    plt.grid(True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

def check_n_uniform():
    # extract the names of all files in the given folder
    file_list = os.listdir(csv_data_fol)

    for csv in file_list:
        csv_path = f"{csv_data_fol}\\{csv}"
        # open each one by one
        data_file = pd.read_csv(csv_path , low_memory = False , dtype = float)
        if(data_file.shape[0] > sampling_freq * window_len_sec):
            # over 10 min length 
            uniformer(data_file , csv , uniform_csv_fol)
        else:
            # already 10 min length so simply save them there
            data_file.to_csv(path_or_buf = f"{uniform_csv_fol}\\Uniformed_{csv}" , index = False , header = None)


def uniformer(ppg_df , csv_name , uniform_csv_folder):
    # extract original signal from the csv file and plot it
    ppg_data = ppg_df.iloc[ : , 0].tolist()
    time_stamp_original_signal = [ t/sampling_freq for t in range(ppg_df.shape[0]) ] # time in seconds

    # calculate some values
    signal_len = ppg_df.shape[0]
    print(f"\nFile name = {csv_name}")
    print(f"Signal Len = {signal_len}")
    stride_samples = stride_len_sec * sampling_freq
    samples_per_window = int(window_len_sec * sampling_freq)
    num_segments = ( signal_len -  samples_per_window + stride_samples ) / stride_samples
    print("Number of segments = {}".format(num_segments))

    int_num_segments = int(num_segments)

    for i in range(int_num_segments):
        # take out the segment from the total signal
        current_segment = ppg_data[i * stride_samples : i * stride_samples + samples_per_window]
        plot_signal(range(len(current_segment)) , current_segment , "Samples", "PPG Signal" , f"{csv_name}: Segment {i + 1}")
        uniform_df = pd.DataFrame(data = current_segment , columns = [f"Segment {i + 1}"])
        uniform_df.to_feather(f"{uniform_csv_folder}\\Uniformed_{csv_name[ : -4]}_Segment_{i}.fthr")
        
check_n_uniform()
