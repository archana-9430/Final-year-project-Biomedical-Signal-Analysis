from imported_files.paths_n_vars import  filtered_merged, stats_features, AE_features, feature_merged
from imported_files.statistical_feature import statistical
# from imported_files.merge import merge_csv

import pandas as pd
# import numpy as np
import os

input_file = filtered_merged
output_file = stats_features

def store_features(input_train_file, local_features_file):
    df = pd.read_csv(input_train_file, skiprows=2, header=None) 

    # calculate the featurse
    extracted_features = {col: statistical(df[col].to_numpy(copy = True)) for col in df}
    features_df = pd.DataFrame(extracted_features)
    features_df = features_df.T

    # check for nan values
    if features_df.isnull().values.any():
        print("ERROR::STORE FEATURES::RETURNS NAN VALUES")

    # add precalculated values for ApEn    
    if os.path.exists(local_features_file):
        temp_df = pd.read_csv(local_features_file)
        try:
            features_df['Approx_entropy_EN'] = temp_df['Approx_entropy_EN']
            features_df['dwt_Approx_entropy_EN'] = temp_df['dwt_Approx_entropy_EN']
        except:
            print('EXCEPTION :: ApEn NOT FOUND ON AXIS')
            pass
    
    # save the files
    features_df.to_csv(local_features_file, index = False)
    print(features_df)

def merge_all_features(_features_file , _ae_features_file , _all_features_file):
    '''
    Merges all the features into a single csv file
    '''
    features_df = pd.read_csv(_features_file)
    AE_features_df = pd.read_csv(_ae_features_file)

    if 'annotation' in features_df.columns and 'PatientID' in features_df.columns:
        features_df.drop(columns=['annotation','PatientID'],inplace=True)
    if 'annotation' in AE_features_df.columns and 'PatientID' in AE_features_df.columns:
        AE_features_df.drop(columns=['annotation','PatientID'],inplace=True)
        
    all_features_df = pd.concat([features_df , AE_features_df] , axis = 1)
    print(all_features_df)
    all_features_df.to_csv(_all_features_file , index=False)
    
    
def add_patientID_annotation(annotation_file, target_file):
    '''
    ADD PATIENT ID AND ANNOTATION TO FINAL target_file IN COLUMN_0 and COLUMN_1 respectively
    '''
    annotation_row = pd.read_csv(annotation_file, nrows=2, header=None)
    target_df = pd.read_csv(target_file)
    if 'annotation' not in target_df.columns and 'PatientID' not in target_df.columns:
        target_df.insert(0, 'annotation', annotation_row.iloc[1])
        target_df.insert(0, 'PatientID', annotation_row.iloc[0])
        target_df.to_csv(target_file, index=False)

def _main_feature_ext(calculate_features: bool = True ):
    if calculate_features:
        store_features(input_file, output_file)
    merge_all_features(output_file , AE_features , feature_merged)
    add_patientID_annotation(input_file, AE_features)
    add_patientID_annotation(input_file, output_file)
    add_patientID_annotation(input_file, feature_merged)

if __name__ == "__main__":
    import time
    start = time.perf_counter()

    _main_feature_ext(True)

    print(f"{__file__} took {time.perf_counter() - start : 0.6f} seconds" )

