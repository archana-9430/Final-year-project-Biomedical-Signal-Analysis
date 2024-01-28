'''This file should make all the csv file to contain 8min long datam i.e, it should have 60000 PPG signal values.'''

csv_data_fol = "Csv_data"
ten_min_csv_fol = "10min_csv_data"

'''
The script takes csv files from "Csv_data" folder and then makes the signal of 10 min 
and save the output csv to folder "10min_csv_data".
For that purpose it assumes sampling frequency as specified in the variable "sampling_freq"
'''

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
sampling_freq = 125 # sampling freq in HERTZ

window_len_sec = 10*60 # in seconds
stride_len_sec = window_len_sec

# debug only...
save = True
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import datatable as dt
import numpy as np
import os
from imported_files.snr import snr_csv
import matplotlib.pyplot as plt

def plot_signal(fig_num:int, x : list ,y : list , x_label:str = None , y_label:str = None , title:str = None) -> None:
    plt.figure(fig_num)
    plt.plot(x,y)
    plt.grid(True)
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
        data_dt = dt.fread(csv_path)
        data_file = data_dt.to_pandas()
        data = data_file.dropna()
        if(data.shape[0] > sampling_freq * window_len_sec + 1):
            # over 10 min length 
            uniformer(data , f"{ten_min_csv_fol}\\{csv}")
        else:
            # already 10 min length or less so simply save them
            if save:
                data.to_csv(path_or_buf = f"{ten_min_csv_fol}\\{csv}" , index = False)


def uniformer(ppg_df:pd.DataFrame , save_path:str) -> None:
    
    # calculate some values
    signal_len = len(ppg_df)
    print(f"\nFile name = {save_path}")
    print(f"Signal Len = {signal_len}")
    stride_samples = stride_len_sec * sampling_freq
    samples_per_window = int(window_len_sec * sampling_freq) + 1

    # calculate number of overlapping segments
    num_segments = ( signal_len -  samples_per_window + stride_samples ) / stride_samples
    int_num_segments = int(num_segments)
    print(f"Number of segments = {num_segments}")
    ppg_list = ppg_df.values

    for i in range(int_num_segments):
        # take out the segment from the total signal
        current_segment = ppg_list[i * stride_samples : i * stride_samples + samples_per_window]

        # if you want to suppress the plotting then comment out the following line
        plot_signal(i + 1 , range(samples_per_window) , current_segment , "Samples", "PPG Signal" , f"{save_path}: Segment {i + 1}")
        
        uniform_df = pd.DataFrame(data = current_segment , columns = [f"Segment {i + 1}"])
        if save:
            save_pathh = f"{save_path[ : -4]}_Segment_{i + 1}.csv"
            uniform_df.to_csv( save_pathh, index = False)
            print(f"   SNR of Segment {i + 1} = {snr_csv(save_pathh , sampling_freq)}")

        
def plot_csv_data(csv_file:str, fig_num):
    # Read the CSV file
    df_dt = dt.fread(csv_file)
    df = df_dt.to_pandas()
    # Extract the single column
    data_column = df.values
    print(range(len(df)))

    # Plot the data
    plt.figure(fig_num)
    plt.plot(range(len(df)), data_column)
    plt.title(csv_file)
    plt.xlabel("time")
    plt.ylabel("PPG Signal")
    plt.show()
    
def plot_csv(csv_path:str):
    dir_list = os.listdir(csv_path)
    print(len(dir_list))

    fig_num = 0
    for csv_file in dir_list:
        fig_num += 1
        plot_csv_data(f"{csv_path}\\{csv_file}", fig_num)

# check_n_uniform()
plot_csv(ten_min_csv_fol)