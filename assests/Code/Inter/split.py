import sys
 
sys.path.insert(0, 'C:/Users/MYPC/Documents/Final-year-project-Biomedical-Signal-Analysis/assests/Code/imported_files')

import os
import shutil

from paths_n_vars import annotated_folder, inter_train, inter_test
from merge import merge_csv
 
# print(annotated_folder)
# gather all files
src_folder = "..\\" + annotated_folder
allfiles = os.listdir(src_folder)
 
def split():
    for i in range(len(allfiles)):
        if i % 2:
            print(f"source = {src_folder + "\\" + allfiles[i]}")
            shutil.copy2(src_folder + "\\" + allfiles[i], inter_train)
            
        else:
            print(f"source = {src_folder + "\\" + allfiles[i]}")
            shutil.copy2(src_folder + "\\" + allfiles[i], inter_test)
        
def merge():
    merge_csv(inter_train, inter_train + "\\" + "combined_annotated_train.csv")
    merge_csv(inter_test, inter_test + "\\" + "combined_annotated_test.csv")

merge()
        
