from imported_files.merge import merge_csv
from imported_files.paths_n_vars import segmented_folder, annotated_data_fol, annotated_merged

# for filtered files
input_folder = segmented_folder
# output_folder = annotation_folder #~~~ for previous annotation
#~~~ new annotation
output_folder = annotated_data_fol

# for unfiltered files
# input_folder = "4.Ten_sec_segmented_unfiltered"
# output_folder = "5.Annotation_unfiltered"

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
# class list
class_list = ["0" , "1" , "2"] # good segn = 0 , partly Corrupted signal = 1 , corrupted = 2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
from pprint import pprint
import pandas as pd
from imported_files.plot import plot_signal_interactive



def take_annotation(segment_num:int ,c_seg : list):# safe annotation accept
    # plot the signal
    plot_signal_interactive(range(len(c_seg)) , c_seg ,'b' , "Sample Number", "PPG Signal" , "Segment {}".format(segment_num))

    while True:
        temp = input("Segment-{} annot = ".format(segment_num))
        if temp in class_list:
            return int(temp)
        elif temp == ';': # input ';' if u want to replot the current segment
            plot_signal_interactive(range(len(c_seg)) , c_seg ,'b' , "Sample Number", "PPG Signal" , "AGAIN\nSegment {}".format(segment_num))
        else:
            print("\n!!Enter a valid number please!!\n")

def annotator(segmented_file_path , annotated_file_path):
    # extract time stamp , original signal from the csv file and pre-process e.g. discard 
    ppg_df = pd.read_csv(segmented_file_path)

    # annotation starts
    print("~~~~~ANNOTATION Starts~~~~~~\n")
    print("Good Signal = Class 0\nPartly Corrupted Signal = Class 1\nCorrupted Signal = Class 2")

    num_segments = ppg_df.shape[1]
    annotation_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(1 , num_segments + 1):
        # take out the segment from the total signal
        current_segment = ppg_df[f"Segment {i}"].to_list()

        # ask for annotation
        annot = take_annotation(i, current_segment)

        # save the annotation 
        current_segment.insert(0,annot)
        annotation_df["Segment {}".format(i)] = current_segment
        
        # save on each annotation
        annotation_df.to_csv(path_or_buf = annotated_file_path , index = False) # save on each segment

    annotation_df.to_csv(path_or_buf = annotated_file_path , index = False) #save the file

def _main_annotation():
    # Get the list of all files and directories
    csv_list = os.listdir(input_folder)
    pprint(f"{csv_list}, {len(csv_list)}")
    # for csv_file in csv_list:
    #     print(f"{segmented_folder}\\{csv_file}")

    for csv_file in csv_list:
        if csv_file.split('.')[-1] == 'csv':
            print(f"\nFILE NAME = {csv_file}")
            annotator(f"{input_folder}\\{csv_file}", 
                    f"{output_folder}\\{csv_file}"
                    )

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    
    # _main_annotation()
    
    # elapsed = time.perf_counter() - start
    # print(f"{__file__} executed in {elapsed:0.2f} seconds.")
    
    merge_csv("3.annotated_data", annotated_merged)