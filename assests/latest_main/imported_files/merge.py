'''
* Script implements a function to merge all files in given folder
* Either saves them in the given folder location 
* Or returns the merged dataframe
'''

import os
import pandas as pd
import re

# input_folder = annotated_folder
# merge_path = annotated_folder + "\\" + "combined_annotated.csv"

def merge_csv(input_folder, merge_path , save : bool = True):
    '''
    * A function to merge all files in "input_folder"
    * if "save" is true saves them in "merge_path" location 
    * always returns the merged dataframe 
    '''
    if os.path.exists(merge_path):
        print(f"MERGED FILE ({merge_path}) ALREADY EXISTS::SO REMOVING IT FIRST")
        os.remove(merge_path)
    dir_list = os.listdir(input_folder)
    dir_list.sort()
    merged_df = pd.DataFrame()

    # process only the csv files
    csv_files = [file for file in dir_list if file.endswith('.csv')]

    for csv in csv_files:
        file_path = os.path.join(input_folder, csv)

        # getting numbers from string 
        patient_number = str(re.findall(r'\d+', csv)[0]) # Extract prefix / patient number

        df = pd.read_csv(file_path, header=None)
        first_row = df.iloc[0]
        # print("type of first_row : " , type(first_row))

        for i in range(len(first_row)):
            second_part = str(re.findall(r'\d+', first_row[i])[0])
            first_row[i] = patient_number + "_" + second_part

        df[:0] = first_row
        # print(df[0])
        merged_df = pd.concat([merged_df, df], axis=1)

    if save:
        merged_df.to_csv(merge_path, index=False, header=False)
        print("file saved at: ", merge_path)

    return merged_df

    