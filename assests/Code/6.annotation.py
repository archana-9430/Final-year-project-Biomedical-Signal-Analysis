from imported_files.paths_n_vars import segmented_folder , annotated_folder

input_folder = segmented_folder
output_folder = annotated_folder

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
# class list
class_list = ["0" , "1" , "2"] # good segn = 0 , partly clean signal = 1 , corrupted = 2

#for debugging only
save_anno = True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from imported_files.plot import plot_signal_interactive
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import os


def take_annotation(segment_num:int ,c_seg : list):# safe annotation accept
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
    print("\n~~~~~ANNOTATION Starts~~~~~~\n")
    print("Good Signal = Class 0\nPartly Clean Signal = Class 1\nCorrupted Signal = Class 2")

    num_segments = ppg_df.shape[1]
    annotation_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(1 , num_segments + 1):
        # take out the segment from the total signal
        current_segment = ppg_df[f"Segment {i}"].to_list()
        
        # plot the signal
        plot_signal_interactive(range(len(current_segment)) , current_segment ,'b' , "Sample Number", "PPG Signal" , "Segment {}".format(i))

        # ask for annotation
        annot = take_annotation(i, current_segment)

        # save the annotation 
        current_segment.insert(0,annot)
        annotation_df["Segment {}".format(i)] = current_segment
        
        # save on each annotation
        if(save_anno == True):
            annotation_df.to_csv(path_or_buf = annotated_file_path , index = False) # save on each segment

    if(save_anno == True):
        annotation_df.to_csv(path_or_buf = annotated_file_path , index = False) #save the file


# Get the list of all files and directories
csv_list = os.listdir(input_folder)
pprint(f"{csv_list}, {len(csv_list)}")
# for csv_file in csv_list:
#     print(f"{segmented_folder}\\{csv_file}")

for csv_file in csv_list:
    if csv_file.split('.')[-1] == 'csv':
        print(f"FILE NAME = {csv_file}")
        annotator(f"{input_folder}\\{csv_file}", 
                  f"{output_folder}\\{csv_file}"
                )
