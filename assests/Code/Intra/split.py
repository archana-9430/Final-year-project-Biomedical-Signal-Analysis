import sys
 
sys.path.insert(0, 'C:/Users/MYPC/Documents/Final-year-project-Biomedical-Signal-Analysis/assests/Code/imported_files')

import os
import shutil

from paths_n_vars import annotated_folder
from merge import merge_csv

src_folder = "..\\" + annotated_folder
        
def merge():
    merge_csv(src_folder, "combined_annotated.csv")

merge()
        
