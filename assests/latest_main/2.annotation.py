from imported_files.merge import merge_csv
from imported_files.paths_n_vars import segmented_folder, annotated_data_fol, annotated_merged


#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
input_folder = segmented_folder
output_folder = annotated_data_fol
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
        response = input("Segment-{} annot = ".format(segment_num))
        if response in class_list:
            return int(response)
        if response == ';': # input ';' if u want to replot the current segment
            plot_signal_interactive(range(len(c_seg)) , c_seg ,'b' , "Sample Number", "PPG Signal" , "AGAIN\nSegment {}".format(segment_num))
        else:
            print("\n!!Enter a valid number please!!\n")

def annotator(segmented_file_path , annotated_file_path):
    ppg_df = pd.read_csv(segmented_file_path)
    
    # assuming one segment in each column
    num_segments = ppg_df.shape[1] 
    i = 1 # index for annotatted segments initialize it to the first one

    annotation_df = pd.DataFrame() # Create an empty dataframe to contain annotated data
    
    # check whether the patient is already annotated to some segmnets
    if os.path.exists(annotated_file_path):
        annotation_df = pd.read_csv(annotated_file_path)

        # if fully annotated then skip the file
        if annotation_df.shape[1] == num_segments: 
            print(f'{segmented_file_path.split("/")[-1]} completely annotated')
            return
        
        i = annotation_df.shape[1] + 1

    # annotation starts
    print("~~~~~ANNOTATION Starts~~~~~~\n")
    print("Good Signal = Class 0\nPartly Corrupted Signal = Class 1\nCorrupted Signal = Class 2")

    while i < num_segments + 1:
        current_segment = ppg_df[f"Segment {i}"].to_list()

        # ask for annotation
        annot = take_annotation(i, current_segment)

        # save the annotation 
        current_segment.insert(0,annot)
        annotation_df["Segment {}".format(i)] = current_segment
        
        # save on each annotation
        annotation_df.to_csv(path_or_buf = annotated_file_path , index = False)

        # next segment
        i += 1
def annotate(local_input_folder,local_output_folder):
     # Get the list of all files and directories
    file_list = os.listdir(local_input_folder)
    pprint(f"{file_list}, {len(file_list)}")
    # for csv_file in file_list:
    #     print(f"{segmented_folder}\\{csv_file}")

    for csv_file in file_list:
        if csv_file.split('.')[-1] == 'csv':
            print(f"\nFILE NAME = {csv_file}")
            annotator(f"{local_input_folder}/{csv_file}",
                    f"{local_output_folder}/{csv_file}"
                    )
            
def _main_annotation():
    annotate(input_folder , output_folder)
    merge_csv("3.annotated_data", annotated_merged)


if __name__ == "__main__":

    _main_annotation()

    