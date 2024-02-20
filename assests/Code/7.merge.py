from imported_files.paths_n_vars import annotated_folder

import os
import pandas as pd

input_folder = annotated_folder
merge_path = annotated_folder + "\\" + "combined_annotated.csv"

dir_list = os.listdir(input_folder)
merged_df = pd.DataFrame()

csv_files = [file for file in dir_list if file.endswith('.csv')]

for file in csv_files:
    file_path = os.path.join(input_folder, file)
    # filename_parts = os.path.splitext(file)[0].split('_')  # Split filename using "_"
    # file_number = filename_parts[1]  # Extract prefix
    df = pd.read_csv(file_path, header=None)
    # first_row = df.iloc[0].values
    # for i in range(len(first_row)):
    #     second_part = first_row[i].split(" ")[1]  
    #     first_row[i] = file_number + "-" + second_part
    # df.iloc[0] = first_row
    # print(df.iloc[0])
    merged_df = pd.concat([merged_df, df], axis=1)
    
merged_df.to_csv(merge_path, index=False, header=None)

    
    
    