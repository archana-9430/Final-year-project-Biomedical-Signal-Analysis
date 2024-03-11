from imported_files.paths_n_vars import csv_data_fol, segmented_folder \
                                        , sampling_frequency, window_len_seconds,\
                                         shift_len_seconds

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
# for filtered segments
input_folder = csv_data_fol
output_folder = segmented_folder

# # for unfiltered segments
# input_folder = "2.10min_csv_data"
# output_folder = "4.Ten_sec_segmented_unfiltered"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import os
import matplotlib.pyplot as plt
from pprint import pprint

def plot_signal(x : list ,y : list , x_label = None , y_label = None , title = None):
    plt.grid(True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

def segmentator(filtered_csv_path , segmented_csv_path):
    # extract original signal from the csv file and plot it
    ppg_df = pd.read_csv(filtered_csv_path)
    ppg_data = ppg_df.iloc[ : , 0].tolist()
    # time_stamp_original_signal = [ t/sampling_frequency for t in range(ppg_df.shape[0]) ] # time in seconds

    # plot the original signal
    # plot_signal(time_stamp_original_signal , ppg_data, "Samples", "PPG Signal" , "Original Signal")

    # calculate some values
    signal_len = ppg_df.shape[0]
    print(f"Signal Len = {signal_len}")
    stride_samples = shift_len_seconds * sampling_frequency
    samples_per_window = int(window_len_seconds * sampling_frequency)
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
    
    
    segmented_df.to_csv(path_or_buf = f"{segmented_csv_path}" , index = False) #save the file

def rename(processed_folder : str):
  csv_list = os.listdir(processed_folder)
  csv_list.sort()
  name_string = ""

  for i in range(len(csv_list)):
    original_path = f"{processed_folder}\\{csv_list[i]}"
    renamed_path = f"{processed_folder}\\patient_{i}.csv"
    print(original_path)
    print(renamed_path)

    # rename the file
    os.rename(original_path , renamed_path)

    # add it the string
    name_string += "{} -> {}\n".format(original_path.split("\\")[1] , renamed_path.split("\\")[1]) 

  pprint(name_string)

  # save to text 
  txt_file_name = f"{processed_folder}\\mapping.txt"
  with open( txt_file_name , "a") as txt_file:
    txt_file.write(name_string)
  

def _main_segmn_10sec():
    csv_list = os.listdir(input_folder)

    for csv_file in csv_list:
        segmentator(f"{input_folder}//{csv_file}" , f"{output_folder}//{csv_file}")

    rename(output_folder)

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    
    _main_segmn_10sec()
    
    elapsed = time.perf_counter() - start
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
    
