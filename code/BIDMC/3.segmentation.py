filtered_csv_path = "filtered_125Hz\\bidmc03m.csv"
annotated_csv_path = "Annotated_125Hz\\bidmc03m.csv"
sampling_freq = 125
transpose_required = False

import pandas as pd
import matplotlib.pyplot as plt

def transpose(csv_file):
    df = pd.read_csv(csv_file)
    transposed = df.T
    return transposed

def plot_signal(x : list ,y : list , x_label = None , y_label = None , title = None):
    plt.grid()
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

def read_plot_original_signal(Filtered_Csv_Path):
    ppg_df = pd.read_csv(filtered_csv_path)
    
    # Plotting the signal obtained from file
    original_signal = ( ppg_df.iloc[ : , 0] ).tolist()
    time_stamp_original_signal = [ t/sampling_freq for t in range(0 , len(original_signal)) ]
    plot_signal(time_stamp_original_signal , original_signal, "Time(s)", "PPG Signal" , "Original Signal")

    return time_stamp_original_signal , original_signal

def discard_samples(Time_original_signal: list , Original_signal:list):
    should_discard = input("\n Need to discard first and last 0.5 sec?(Y/N , Default: N) ") # discards if this variable is set to True
    if(should_discard == ('y' or 'Y')):
        discard_interval = 0.5
        sample_discard = int(discard_interval * sampling_freq)
        signal = Original_signal[sample_discard:]
        time_stamp = Time_original_signal[sample_discard:]
        signal_len = len(signal)
        signal = signal[:signal_len - sample_discard]
        time_stamp = Time_original_signal[:signal_len - sample_discard]
    else:
        signal = original_signal
        time_stamp = Time_original_signal

    return time_stamp , signal

def annotator( Time_stamp:list, Signal:list):
    
    # Ask the window length 
    window_len_sec = float(input(" Window length in sec? "))
    samples_per_window = int(window_len_sec * sampling_freq)
    num_segments = len(Signal) / samples_per_window
    print("\n Number of segments = {}".format(num_segments))

    print("\n~~~~~ANNOTATION Starts~~~~~~\n")
    print("Clean Signal = Class 1\n\
    Partly Clean = Class 2\n ")

    int_num_segment = int(num_segments)
    Annotation_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(0 , int_num_segment):
        current_segment = Signal[i * samples_per_window : (i+1) * samples_per_window]
        time = Time_stamp[i * samples_per_window : (i+1) * samples_per_window]
        plot_signal(time , current_segment , "Time(s)", "PPG Signal" , "Segment {}".format(i))
        annot = int(input("Segment-{} annot = ".format(i)))
        current_segment.insert(0,annot)
        Annotation_df["Segment {}".format(i)] = current_segment
        Annotation_df.to_csv(path_or_buf = annotated_csv_path , index = False)

    if(sum_segments%1):
        i += 1
        remaining_sample_num = len(Signal) - i * samples_per_window
        current_segment = Signal[i * samples_per_window : i * samples_per_window + remaining_sample_num]
        time = Time_stamp[i * samples_per_window : i * samples_per_window + remaining_sample_num]
        plot_signal(time , current_segment , "Time(s)", "PPG Signal" , "Segment {}".format(i))

        discard_seg = input("\n Discard it(y/n)? ")
        if(discard_seg == 'n'):
            annot = int(input("Segment-{} annot = ".format(i)))
            current_segment.insert(0,annot)
            zero_padd = [ 0 for i in range(0 , samples_per_window - remaining_sample_num)]
            current_segment += zero_padd
            Annotation_df["Segment {}".format(i)] = current_segment

    return Annotation_df

time_stamp_original_signal , original_signal = read_plot_original_signal(filtered_csv_path)

time_stamp , signal = discard_samples(time_stamp_original_signal , original_signal)

annot_df = annotator(time_stamp , signal)

# save the Annotated data on csv file
annot_df.to_csv(path_or_buf = annotated_csv_path , index = False)