import pandas as pd
import matplotlib.pyplot as plt


wfdb_folder_path = "/PPG_3900370_0005m_Part1.txt"
saved_segments_path = "PPG_3900370_0005m_Part1.csv"
annotated_data_path = "PPG_3900370_0005m_Part1_annotated.csv"

def plot_signal(x,y, x_label = None , y_label = None , title = None):
    plt.grid()
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def annotator( Signal , Sampling_freq , Samples_per_window , Num_segments , Discard_on):

    if(Discard_on == True):
        time_signal = [ t/Sampling_freq + 0.5 for t in range(0 , len(Signal)) ]
    else:
        time_signal = [ t/Sampling_freq for t in range(0 , len(Signal)) ]

    int_num_segment = int(Num_segments)
    Annotation_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(0 , int_num_segment):
        current_segment = Signal[i * Samples_per_window : (i+1) * Samples_per_window]
        time = time_signal[i * Samples_per_window : (i+1) * Samples_per_window]
        plot_signal(time , current_segment , "Time(s)", "PPG Signal" , "Segment {}".format(i))
        annot = int(input("Segment-{} annot = ".format(i)))
        current_segment.insert(0,annot)
        Annotation_df["Segment {}".format(i)] = current_segment

    if(Num_segments%1):
        i += 1
        remaining_sample_num = len(Signal) - i * Samples_per_window
        current_segment = Signal[i * Samples_per_window : i * Samples_per_window + remaining_sample_num]
        time = time_signal[i * Samples_per_window : i * Samples_per_window + remaining_sample_num]
        plot_signal(time , current_segment , "Time(s)", "PPG Signal" , "Segment {}".format(i))

        discard_seg = input("\n Discard it(y/n)? ")
        if(discard_seg == 'n'):
            annot = int(input("Segment-{} annot = ".format(i)))
            current_segment.insert(0,annot)
            zero_padd = [ 0 for i in range(0 , Samples_per_window - remaining_sample_num)]
            current_segment += zero_padd
            Annotation_df["Segment {}".format(i)] = current_segment

    return Annotation_df

# Read the file containing the signal
# record = wfdb.rdrecord(wfdb_folder_path)

# Convert the wfdb record to a pandas dataframe
direct_data_frame = pd.read_table(wfdb_folder_path , delimiter = '\t')
print(direct_data_frame)

# Ask user if any transposition is needed
transpose_needed = input("~~~~\nData is row wise ? (y/n) ")

# Transpose if required and Save the data frame into a csv file with same name as the source signal file
if(transpose_needed == 'y'):
    transposed_direct_df = direct_data_frame.T
    print("~~~~~~~~\n The Transposed data is {}".format(transposed_direct_df))
    transposed_direct_df.to_csv(path_or_buf = saved_segments_path , index = False)
else:
    direct_data_frame.to_csv(path_or_buf = saved_segments_path , index = False)

# Now again read the data
ppg_df = pd.read_csv(saved_segments_path)

# Ask for sampling frequency in Hz
sampling_freq = float(input("~~~~~~\n Sampling frequency (Hz)? "))

# Plotting the signal obtained from file
original_signal = ( ppg_df.iloc[ : , 0] ).tolist()
time_original_signal = [ t/sampling_freq for t in range(0 , len(original_signal)) ]
plot_signal(time_original_signal , original_signal, "Time(s)", "PPG Signal" , "Original Signal")

# discard first 0.5 and last 0.5 seconds of the recording if asked so
should_discard = input("\n Need to discard first and last 0.5 sec?(y/n) ") # discards if this variable is set to True
if(should_discard == 'y'):
    should_discard = True
    discard_interval = 0.5
    sample_discard = int(discard_interval * sampling_freq)
    signal = original_signal[sample_discard:]
    original_signal_len = len(signal)
    signal = signal[:original_signal_len - sample_discard]
else:
    should_discard = False
    signal = original_signal

# Ask and window length to chop data into segments of given length
window_len_sec = float(input(" Window length in sec? "))
samples_per_window = int(window_len_sec * sampling_freq)
num_segments = len(signal) / samples_per_window
print("\n Number of segments = {}".format(num_segments))

print("\n~~~~~ANNOTATION Starts~~~~~~\n")
print(" Clean Signal = Class 1\n \
Partly Clean = Class 2\n \
Corrupted    = Class 3\n")

annot_df = annotator(signal , sampling_freq , samples_per_window , num_segments , should_discard)

# save the Annotated data on csv file
annot_df.to_csv(path_or_buf = annotated_data_path , index = False)