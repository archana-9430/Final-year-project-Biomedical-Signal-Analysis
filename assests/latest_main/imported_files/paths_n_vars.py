'''
Contains crucial (relative) folder paths and various variables for entire workflow
'''

# folder paths
csv_data_fol = '1.10min_csv_data'
segmented_folder = '2.segmented_data'
annotated_data_fol = "3.annotated_data"
annotated_merged = "annotated_merged.csv"
fitered_folder = "4.filtered_data"
filtered_merged = 'filtered_merged.csv'
stats_features = 'stats_features.csv'
AE_features = 'AE_features.csv'
feature_merged = 'feature_merged.csv'

encoder_save = 'AE_model/encoder_model.keras'
decoder_save = 'AE_model/decoder_model.keras'
encoder_derivative_save = 'AE_model/encoder_derivative_model.keras'
decoder_derivative_save = 'AE_model/decoder_derivative_model.keras'

ae_derivative_features_file = '6.Features_extracted/AE_derivative_features.csv'

# variables
sampling_frequency = 125 # in Hertz

# segment length per patient in 
# seconds
seg_len_patient = 600 

# window and shift length in 
# seconds
window_len_seconds = 10
shift_len_seconds = 6
