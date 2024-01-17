filtered_csv_path = "code\BIDMC\\filtered_125Hz\\bidmc03m.csv"
segmented_csv_path = "code\BIDMC\segmented_125Hz\\bidmc03m.csv"

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
sampling_freq = 125 # sampling freq in HERTZ

window_len_sec = 10
stride_len_sec = 6
# in seconds

# # global trimming variables
# trim_ask = True
# global_trim = False # set it to True if you want discard, matters if discard_ask is False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import matplotlib.pyplot as plt

def plot_signal(x : list ,y : list , x_label = None , y_label = None , title = None):
    plt.grid(True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

# # For cutting out first and last 0.5 s
# def trim_signal(original_signal:list , ask = False , time_stamp_original: list = NULL): 
# 
#     if(ask == True):
#         should_discard = input("\n Need to discard first and last 0.5 sec?(Y/N , Default: N) ") # discards if this variable is set to True
#     else:
#         should_discard = 'N'
# 
#     if((should_discard == ('y' or 'Y')) or (global_trim == True) and time_stamp_original != NULL):
#         discard_interval = 0.5
#         sample_discard = int(discard_interval * sampling_freq)
#         signal = original_signal[sample_discard : ]
#         time_stamp = time_stamp_original[sample_discard : ]
#         signal_len = len(signal)
#         signal = signal[ : signal_len - sample_discard]
#         time_stamp = time_stamp[ : signal_len - sample_discard]
#   
#     elif((should_discard == ('y' or 'Y')) or (global_discard == True) and time_stamp_original == NULL):
#         discard_interval = 0.5
#         sample_discard = int(discard_interval * sampling_freq)
#         signal = original_signal[sample_discard : ]
#         signal_len = len(signal)
#         signal = signal[ : signal_len - sample_discard]
# 
#     else:
#         signal = original_signal
#         time_stamp = time_stamp_original
# 
#     return time_stamp , signal
    

def segmentator():
    # extract original signal from the csv file and plot it
    ppg_df = pd.read_csv(filtered_csv_path , header = None)
    ppg_data = ppg_df.iloc[ : , 0].tolist()
    time_stamp_original_signal = [ t/sampling_freq for t in range(ppg_df.shape[0]) ] # time in seconds

    # plot the original signal
    plot_signal(time_stamp_original_signal , ppg_data, "Samples", "PPG Signal" , "Original Signal")

    # calculate some values
    signal_len = ppg_df.shape[0]
    stride_samples = stride_len_sec * sampling_freq
    samples_per_window = int(window_len_sec * sampling_freq)
    num_segments = ( signal_len -  samples_per_window + stride_samples ) / stride_samples
    print("\nNumber of segments = {}".format(num_segments))

    int_num_segments = int(num_segments)
    print("\nInteger Number segment = {}".format(int_num_segments))
    segmented_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(int_num_segments):
        # take out the segment from the total signal
        current_segment = ppg_data[i * stride_samples : i * stride_samples + samples_per_window]
        # plot_signal(range(len(current_segment)) , current_segment , "Samples", "PPG Signal" , "Segment {}".format(i))
        segmented_df["Segment {}".format(i)] = current_segment
    
    # if(num_segments % 1):
    #     # calculate the remaining sample number
    #     remaining_sample_num = signal_len - (samples_per_window + i * stride_samples)

    #     i += 1
    #     # take out the segment from the total signal and plot it
    #     current_segment = ppg_data[i * stride_samples : i * stride_samples + remaining_sample_num]
    #     plot_signal(range(len(current_segment)) , current_segment , "Samples", "PPG Signal" , "Last Segment {}".format(i))

    #     discard_seg = input("\nDiscard the last segment(y/n , Default: n)? ")
    #     if(discard_seg == 'n'):
    #         zero_padd = [ 0 for i in range(samples_per_window - remaining_sample_num) ]
    #         current_segment += zero_padd
    #         segmented_df["Segment {}".format(i)] = current_segment
    
    segmented_df.to_csv(path_or_buf = segmented_csv_path , index = False , header = None) #save the file

segmentator()
