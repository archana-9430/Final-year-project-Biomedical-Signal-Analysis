'''
Script splits the annotated data to Train and Test parts in about 50 - 50 ratio
'''
import sys
 
sys.path.insert(0, 'F:/Shwashwat/B_Tech_ECE/project/Github folder/Final-year-project-Biomedical-Signal-Analysis/assests/Code/imported_files')

import os
import shutil

from paths_n_vars import annotated_folder, inter_train, inter_test
from merge import merge_csv


# gather all files
src_folder = "..\\" + annotated_folder
allfiles = os.listdir(src_folder)
 
def split():
    '''
    Does the splitting
    '''
    for i in range(len(allfiles)):
        if allfiles[i].split(".")[-1] == "csv": # check for csv files
            if i % 2:
                # files with even index in train 
                shutil.copy2(src_folder + "\\" + allfiles[i], inter_train)
                
            else:
                # files with odd index in train
                shutil.copy2(src_folder + "\\" + allfiles[i], inter_test)
        
def merge():
    '''
    Merges train data and test data
    '''
    merge_csv(inter_test, inter_test + "\\" + "combined_annotated_test.csv")
    merge_csv(inter_train, inter_train + "\\" + "combined_annotated_train.csv")

split()
merge()
        
