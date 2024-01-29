# folder paths
segmented_folder = "10sec_segmented_data"

import os
import pandas as pd
from pprint import pprint

def rename():
  csv_list = os.listdir(segmented_folder)
  name_dict = {}
  for i in range(len(csv_list)):
    original_path = f"{ten_sec_folder}\\{csv_list[i]}"
    renamed_path = f"{ten_sec_folder}\\patient {i}.csv"
    os.rename(original_path , renamed_path)
    name_dict[original path] = renamed_path

  df = pd.DataFrame(name_dict)
  df.to_csv(path_or_buf = f"{ten_sec_folder}\\mapping.csv" , index = False)
  pprint(name_dict)
  
    
