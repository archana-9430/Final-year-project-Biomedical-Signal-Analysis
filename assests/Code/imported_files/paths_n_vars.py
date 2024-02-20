'''
Contains crucial folder (relative) paths and various variables for entire workflow
'''

# folder paths
csv_data_fol = "1.Csv_data"
csv_noise_fol = "1.Csv_noise_DaLiA"
ten_min_csv_fol = "2.10min_csv_data"
filtered_folder = "3.filtered_csv_data"
segmented_folder = "4.Ten_sec_segmented_data"
annotated_folder = "5.Ten_sec_annotated_data"
inter_folder = "Inter"
intra_folder = "Intra"
inter_train = "Train"
inter_test = "Test"

# variables
sampling_frequency = 125 # in Hertz

# segment length per patient in 
# seconds
seg_len_patient = 600 

# window and shift length in 
# seconds
window_len_seconds = 10
shift_len_seconds = 6
