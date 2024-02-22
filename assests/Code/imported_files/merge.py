'''
* Script implements a function to merge all files in given folder
* Either saves them in the given folder location 
* Or returns the merged dataframe
'''

import os
import pandas as pd

# input_folder = annotated_folder
# merge_path = annotated_folder + "\\" + "combined_annotated.csv"

def merge_csv(input_folder, merge_path , save : bool = True):
    '''
    * A function to merge all files in "input_folder"
    * if "save" is true saves them in "merge_path" location 
    * else returns the merged dataframe
    '''
    if os.path.exists(merge_path):
        print(f"MERGED FILE ({merge_path}) ALREADY EXISTS::SO REMOVING IT FIRST")
        os.remove(merge_path)

    dir_list = os.listdir(input_folder)
    merged_df = pd.DataFrame()

    csv_files = [file for file in dir_list if file.endswith('.csv')]
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        filename_parts = os.path.splitext(file)[0].split('_')  # Split filename using "_"
        file_number = filename_parts[1]  # Extract prefix
        df = pd.read_csv(file_path, header=None)
        first_row = df.iloc[0].values
        for i in range(len(first_row)):
            second_part = first_row[i].split(" ")[1]  
            first_row[i] = file_number + "_" + second_part
        df[:0] = first_row
        # print(df[0])
        merged_df = pd.concat([merged_df, df], axis=1)

    if save:
        merged_df.to_csv(merge_path, index=False, header=False)

    