'''
Contains crucial (relative) folder paths and various variables for entire workflow
'''

# folder paths
csv_data_fol = '1.10min_csv_data'
csv_noise_fol = '1.Csv_noise_DaLiA'
ten_min_csv_fol = '2.10min_csv_data'
filtered_folder = '3.filtered_csv_data'
segmented_folder = '2.segmented_data'
annotation_folder = '5.Ten_sec_annotated_data'
new_annotation_folder = '5.New_annotated_data'
features_folder = '6.Features_extracted'
features_file = '6.Features_extracted/features.csv'
ae_features_file = '6.Features_extracted/AE_features.csv'
ae_derivative_features_file = '6.Features_extracted/AE_derivative_features.csv'
all_features_file = '6.Features_extracted/all_features.csv'
min_max_features_file = '6.Features_extracted/all_features_minmax.csv'
zscore_features_file = '6.Features_extracted/all_features_zscore.csv'
inter_folder = 'Inter'
intra_folder = 'Intra'
inter_train = 'Train'
inter_test = 'Test'
inter_test_file = 'Inter/Test/combined_annotated_test.csv'
inter_train_file = 'Inter/Train/combined_annotated_train.csv'
intra_annotated_file = 'Intra/combined_annotated.csv'
encoder_save = 'AE_model/encoder_model.keras'
decoder_save = 'AE_model/decoder_model.keras'
encoder_derivative_save = 'AE_model/encoder_derivative_model.keras'
decoder_derivative_save = 'AE_model/decoder_derivative_model.keras'

# variables
sampling_frequency = 125 # in Hertz

# segment length per patient in 
# seconds
seg_len_patient = 600 

# window and shift length in 
# seconds
window_len_seconds = 10
shift_len_seconds = 6
