filtered_csv_path = "code\BIDMC\\filtered_125Hz\\bidmc03m.csv"
segmented_csv_path = "code\BIDMC\Semgmented_125Hz\\bidmc03m.csv"

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
sampling_freq = 125 # sampling freq in HERTZ
window_len_sec = 10

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
    ppg_df = pd.read_csv(filtered_file_path)
    time_stamp_original_signal = [ t/sampling_freq for t in range(original_signal.shape[0]) ] # time in seconds

    # plot the original signal
    plot_signal(time_stamp_original_signal , ppg_df[1], "Time(s)", "PPG Signal" , "Original Signal")

    # calculate some values
    signal_len = ppg_df.shape[0]
    samples_per_window = int(window_len_sec * sampling_freq)
    num_segments = signal_len / samples_per_window
    print("\nNumber of segments = {}".format(num_segments))

    int_num_segment = int(num_segments)
    segmented_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(int_num_segment):
        # take out the segment from the total signal
        current_segment = signal.iloc[i * samples_per_window : (i + 1) * samples_per_window]
        segmented_df["Segment {}".format(i)] = current_segment
    
    if(num_segments % 1):
        i += 1
        # calculate the remaining sample number
        remaining_sample_num = signal_len - i * samples_per_window

        # take out the segment from the total signal and plot it
        current_segment = signal[i * samples_per_window : i * samples_per_window + remaining_sample_num]
        plot_signal(range(signal_len) , current_segment , "Time(s)", "PPG Signal" , "Segment {}".format(i))
        segmented_df["Segment {}".format(i)] = current_segment

        discard_seg = input("\nDiscard the last segment(y/n , Default: n)? ")
        if(discard_seg == 'n'):
            annot = take_annotation(i)
            current_segment.insert(0 , annot)
            zero_padd = [ 0 for i in range(0 , samples_per_window - remaining_sample_num)]
            current_segment += zero_padd
            segmented_df["Segment {}".format(i)] = current_segment

    
    segmented_df.to_csv(path_or_buf = segmented_csv_path , index = False) #save the file

segmentator()
