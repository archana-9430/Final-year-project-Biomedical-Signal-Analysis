# from ...imported_files.paths_n_vars import annotated_folder, inter_train, inter_test
 
# importing sys
import sys
 
# adding Folder_2/subfolder to the system path
sys.path.insert(0, 'C:/Users/MYPC/Documents/Final-year-project-Biomedical-Signal-Analysis/assests/Code/imported_files')
import os
import shutil
from paths_n_vars import annotated_folder, inter_train, inter_test
 
# print(annotated_folder)
# gather all files
target_folder = "..\\" + annotated_folder
allfiles = os.listdir(target_folder)
 
for i in range(len(allfiles)):
    if i % 2:
        print(f"source = {target_folder + "\\" + allfiles[i]}")
        shutil.copy2(target_folder + "\\" + allfiles[i], "Train")
    else:
        print(f"source = {target_folder + "\\" + allfiles[i]}")
        shutil.copy2(target_folder + "\\" + allfiles[i], "Test")
        
