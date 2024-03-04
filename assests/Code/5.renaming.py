from imported_files.paths_n_vars import ten_min_csv_fol
# folder paths
# for filtered
# input_folder = ten_min_csv_fol

# for unfiltered
input_folder = "4.Ten_sec_segmented_unfiltered"

import os
from pprint import pprint

def rename(processed_folder : str):
  csv_list = os.listdir(processed_folder)
  name_string = ""

  for i in range(len(csv_list)):
    original_path = f"{processed_folder}\\{csv_list[i]}"
    renamed_path = f"{processed_folder}\\patient_{i}.csv"

    # rename the file
    os.rename(original_path , renamed_path)

    # add it the string
    name_string += "{} -> {}\n".format(original_path.split("\\")[1] , renamed_path.split("\\")[1]) 

  pprint(name_string)

  # save to text 
  txt_file_name = f"{processed_folder}\\mapping.txt"
  with open( txt_file_name , "a") as txt_file:
    txt_file.write(name_string)
  
rename(input_folder)