'''
Script shuffles then splits the annotated data to Train and Test parts in about 50 - 50 ratio
'''
import sys
 
sys.path.insert(0, 'F:/Shwashwat/B_Tech_ECE/project/Github folder/Final-year-project-Biomedical-Signal-Analysis/assests/Code/imported_files')
import os
import shutil
from paths_n_vars import annotated_folder
from merge import merge_csv

src_folder = "../" + annotated_folder
        
def merge():
    merge_csv(src_folder, "combined_annotated.csv" , save = True)

merged_df_intra = merge()

from icecream import ic 
  
ic.configureOutput(includeContext=True)
