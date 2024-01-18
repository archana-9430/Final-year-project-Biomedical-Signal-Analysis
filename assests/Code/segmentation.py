filtered_csv_path = "code\BIDMC\\filtered_125Hz\\bidmc03m.csv"
segmented_csv_path = "code\BIDMC\segmented_125Hz\\bidmc03m.csv"

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
sampling_freq = 125 # sampling freq in HERTZ

window_len_sec = 10
stride_len_sec = 6
# in seconds
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_signal(x : list ,y : list , x_label = None , y_label = None , title = None):
    plt.grid(True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

def segmentator():
    # extract original signal from the csv file and plot it
    ppg_df = pd.read_csv(filtered_csv_path , header = None)
    ppg_data = ppg_df.iloc[ : , 0].tolist()
    time_stamp_original_signal = [ t/sampling_freq for t in range(ppg_df.shape[0]) ] # time in seconds

    # plot the original signal
    plot_signal(time_stamp_original_signal , ppg_data, "Samples", "PPG Signal" , "Original Signal")

    # calculate some values
    signal_len = ppg_df.shape[0]
    print(f"Signal Len = {signal_len}")
    stride_samples = stride_len_sec * sampling_freq
    samples_per_window = int(window_len_sec * sampling_freq)
    num_segments = ( signal_len -  samples_per_window + stride_samples ) / stride_samples
    print(f"\nNumber of segments = {num_segments}")

    int_num_segments = int(num_segments)
    print(f"\nInteger Number segment = {int_num_segments}")
    segmented_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(int_num_segments):
        # take out the segment from the total signal
        current_segment = ppg_data[i * stride_samples : i * stride_samples + samples_per_window]
        # plot_signal(range(len(current_segment)) , current_segment , "Samples", "PPG Signal" , "Segment {}".format(i))
        segmented_df[f"Segment {i + 1}"] = current_segment
    
    segmented_df.to_csv(path_or_buf = segmented_csv_path , index = False , header = None) #save the file
    
segmentator()
