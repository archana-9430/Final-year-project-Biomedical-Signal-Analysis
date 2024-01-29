segmented_folder= "10sec_segmented_data"
annotated_folder = "10sec_annotated_data"

#~~~~~~~~~~~~~~~~~~~~~~Check these before running~~~~~~~~~~~~~~~~~~~~~~~~~
# sampling_freq = 125 # sampling freq in HERTZ
# window_len_sec = 10

# class list
class_list = ["0" , "1" , "2"] # good segn = 0 , partly corrupted signal = 1 , corrupted = 2

#for debugging only
save_anno = True
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_signal(x : list ,y : list , x_label = None , y_label = None , title = None):
    plt.grid(True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

def take_annotation(segment_num:int):# safe annotation accept
    while True:
        temp = input("Segment-{} annot = ".format(segment_num))
        if temp in class_list:
            return int(temp)
        else:
            print("\n!!Enter a valid number please!!\n")

def annotator(segmented_file_path , annotated_file_path):
    # extract time stamp , original signal from the csv file and pre-process e.g. discard 
    ppg_df = pd.read_csv(segmented_file_path)

    # annotation starts
    print("\n~~~~~ANNOTATION Starts~~~~~~\n")
    print("Good Signal = Class 0\nCorrupted Signal = Class 1\n ")

    num_segments = ppg_df.shape[1]
    annotation_df = pd.DataFrame() # Create an empty dataframe to contain annotated data

    for i in range(1 , num_segments + 1):
        # take out the segment from the total signal
        current_segment = ppg_df[i]
        
        # plot the signal
        plot_signal(range(len(current_segment)) , current_segment , "Time(s)", "PPG Signal" , "Segment {}".format(i))

        # ask for annotation
        annot = take_annotation(i)

        # save the annotation 
        current_segment.insert(0,annot)
        annotation_df["Segment {}".format(i)] = current_segment
        
        # save on each annotation
        if(save_anno == True):
            annotation_df.to_csv(path_or_buf = annotated_file_path , index = False) # save on each segment

    if(save_anno == True):
        annotation_df.to_csv(path_or_buf = annotated_file_path , index = False) #save the file
    annotation_df.close()
    ppg_df.close()


# Get the list of all files and directories
csv_list = os.listdir(segmented_folder)
print(f"{csv_list}, {len(csv_list)}")
# for csv_file in csv_list:
#     print(f"{segmented_folder}\\{csv_file}")

for csv_file in csv_list:
    annotator(f"{segmented_folder}\\{csv_file}", 
              f"{annotated_folder}\\{csv_file}"
             )
