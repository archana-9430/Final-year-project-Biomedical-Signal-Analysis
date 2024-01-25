'''This file should make all the csv file to contain 8min long datam i.e, it should have 60000 PPG signal values.'''

csv_data_fol = "Csv_data"
uniform_csv_fol = "uniform_csv_data"

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
        data_file = np.loadtxt(csv_path , delimiter = ',' , skiprows = 1 , encoding = None)
        df = pd.DataFrame(data_file)
        if(len(data_file) > sampling_freq * window_len_sec + 1):
            # over 10 min length 
            uniformer(data_file , f"{uniform_csv_fol}\\{csv}")
        else:
            # already 10 min length or less so simply save them
            df.to_csv(path_or_buf = f"{uniform_csv_fol}\\{csv}" , index = False, header=False)


def uniformer(ppg_np , save_path):
    # extract original signal from the csv file and plot it
    # ppg_data = ppg_df.iloc[ : , 0].tolist()

    # calculate some values
    signal_len = len(ppg_np)
    print(f"\nFile name = {save_path}")
    print(f"Signal Len = {signal_len}")
    stride_samples = stride_len_sec * sampling_freq
    samples_per_window = int(window_len_sec * sampling_freq) + 1
    num_segments = ( signal_len -  samples_per_window + stride_samples ) / stride_samples
    print("Number of segments = {}".format(num_segments))

    int_num_segments = int(num_segments)

    for i in range(int_num_segments):
        # take out the segment from the total signal
        current_segment = ppg_np[i * stride_samples : i * stride_samples + samples_per_window]
        # plot_signal(range(len(current_segment)) , current_segment , "Samples", "PPG Signal" , f"{csv_name}: Segment {i + 1}")
        uniform_df = pd.DataFrame(data = current_segment , columns = [f"Segment {i + 1}"])
        uniform_df.to_csv(f"{save_path[ : -4]}_Segment_{i + 1}.csv" , index = False)
        
def plot_csv_data(csv_file, fig_num):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Extract the single column
    data_column = df.iloc[:, 0]

    # Plot the data
    plt.figure(fig_num)
    plt.plot(df.index, data_column, label=df.columns[0])
    plt.title(csv_file)
    plt.xlabel("time")
    plt.ylabel("PPG Signal")
    plt.show()
    
def plot_csv(csv_path):
    dir_list = os.listdir(csv_path)
    print(len(dir_list))

    fig_num = 0
    for csv_file in dir_list:
        fig_num += 1
        plot_csv_data(f"{csv_path}\\{csv_file}", fig_num)

check_n_uniform()
plot_csv(uniform_csv_fol)