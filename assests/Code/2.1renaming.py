# folder paths
segmented_folder = "10sec_segmented_data"

import os
from pprint import pprint

def rename():
  csv_list = os.listdir(segmented_folder)
  name_string = ""

  for i in range(len(csv_list)):
    original_path = f"{segmented_folder}\\{csv_list[i]}"
    renamed_path = f"{segmented_folder}\\patient_{i}.csv"

    # rename the file
    os.rename(original_path , renamed_path)

    # add it the string
    name_string += f"{original_path} -> {renamed_path}\n" 

  pprint(name_string)

  # save to text 
  txt_file_name = f"{segmented_folder}\\mapping.txt"
  with open( txt_file_name , "a") as txt_file:
    txt_file.write(name_string)
  
rename()