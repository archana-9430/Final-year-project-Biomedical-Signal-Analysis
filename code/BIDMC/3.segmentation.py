filtered_csv_path = "filtered_125Hz\\bidmc03m.csv"
annotated_csv_path = "Annotated_125Hz\\bidmc03m.csv"
#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
# save annotations settings ;;; for testing the running code protects from accidental overwrite
# of the csv file
save_anno = False

# sampling freq in HERTZ
sampling_freq = 125 
window_len_sec = 10

# global discard variables
discard_ask = False
global_discard = False # set it to True if you want discard, matters if discard_ask is False

# transpose settings
transpose_required = False

# class list
class_list = ["1" , "2"] # good segn = 1 , corrupted signal = 2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import matplotlib.pyplot as plt

def plot_signal(x : list ,y : list , x_label = None , y_label = None , title = None):
    plt.grid()
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

def pre_segmentation(filtered_file_path):
    ppg_df = pd.read_csv(filtered_file_path)
    
    # transpose if asked to do so
    if(transpose_required == True):
        ppg_df = ppg_df.T

    # Plotting the signal obtained from file
    original_signal = ( ppg_df.iloc[ : , 0] ).tolist()
    time_stamp_original_signal = [ t/sampling_freq for t in range(0 , len(original_signal)) ]
    plot_signal(time_stamp_original_signal , original_signal, "Time(s)", "PPG Signal" , "Original Signal")

    # discard samples if required
    time_stamp_trimmed , signal_trimmed = discard_samples(time_stamp_original_signal , original_signal , discard_ask)

    return time_stamp_trimmed , signal_trimmed

def discard_samples(time_stamp_original: list , original_signal:list , ask = False):
    
    if(ask == True):
        should_discard = input("\n Need to discard first and last 0.5 sec?(Y/N , Default: N) ") # discards if this variable is set to True
    else:
        should_discard = 'N'

    if((should_discard == ('y' or 'Y')) or (global_discard == True)):
        discard_interval = 0.5
        sample_discard = int(discard_interval * sampling_freq)
        signal = original_signal[sample_discard:]
        time_stamp = time_stamp_original[sample_discard:]
        signal_len = len(signal)
        signal = signal[:signal_len - sample_discard]
        time_stamp = time_stamp[:signal_len - sample_discard]
    else:
        signal = original_signal
        time_stamp = time_stamp_original

    return time_stamp , signal

def take_annotation(segment_num:int):# safe annotation accept
    while True:
        temp = input("Segment-{} annot = ".format(segment_num))
        if temp in class_list:
            return int(temp)
        else:
            print("\n!!Enter a valid number please!!\n")

def annotator(filtered_file_path , annotated_file_path):
    # extract time stamp , original signal from the csv file and pre-process e.g. discard 
    time_stamp , signal = pre_segmentation(filtered_file_path)

    # Ask the window length 
    # window_len_sec = float(input("Window length in sec? "))

    # calculate some values
    samples_per_window = int(window_len_sec * sampling_freq)
    num_segments = len(signal) / samples_per_window
    print("\nNumber of segments = {}".format(num_segments))

    # annotation starts
    print("\n~~~~~ANNOTATION Starts~~~~~~\n")
    print("Good Signal = Class 1\nCorrupted Signal = Class 2\n ")

    int_num_segment = int(num_segments)
    annotation_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(0 , int_num_segment):
        # take out the segment from the total signal
        current_segment = signal[i * samples_per_window : (i+1) * samples_per_window]
        time = time_stamp[i * samples_per_window : (i+1) * samples_per_window]

        # plot the signal
        plot_signal(time , current_segment , "Time(s)", "PPG Signal" , "Segment {}".format(i))

        # ask for annotation
        annot = take_annotation(i)

        #save the annotation 
        current_segment.insert(0,annot)
        annotation_df["Segment {}".format(i)] = current_segment
        
        # save on each segment
        if(save_anno == True):
            annotation_df.to_csv(path_or_buf = annotated_file_path , index = False) #save on each segment

    if(num_segments%1):
        i += 1
        # calculate the remaining sample number
        remaining_sample_num = len(signal) - i * samples_per_window

        # take out the segment from the total signal and plot it
        current_segment = signal[i * samples_per_window : i * samples_per_window + remaining_sample_num]
        time = time_stamp[i * samples_per_window : i * samples_per_window + remaining_sample_num]
        plot_signal(time , current_segment , "Time(s)", "PPG Signal" , "Segment {}".format(i))

        discard_seg = input("\nDiscard it(y/n , Default: n)? ")
        if(discard_seg == 'y'):
            annot = take_annotation(i)
            current_segment.insert(0,annot)
            zero_padd = [ 0 for i in range(0 , samples_per_window - remaining_sample_num)]
            current_segment += zero_padd
            annotation_df["Segment {}".format(i)] = current_segment

    if(save_anno == True):
        annotation_df.to_csv(path_or_buf = annotated_file_path , index = False) #save the file

annotator(filtered_csv_path , annotated_csv_path)
