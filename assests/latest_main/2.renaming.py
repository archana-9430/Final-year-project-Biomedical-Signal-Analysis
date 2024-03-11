from imported_files.paths_n_vars import csv_data_fol
# folder paths
# for filtered
input_folder = csv_data_fol

# for unfiltered
# input_folder = "4.Ten_sec_segmented_unfiltered"

import os



if __name__ == "__main__":
    import time
    start = time.perf_counter()
    
    rename(input_folder)
    
    elapsed = time.perf_counter() - start
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")