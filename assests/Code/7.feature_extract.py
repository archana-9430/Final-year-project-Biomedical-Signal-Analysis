from imported_files.paths_n_vars import inter_train_file, inter_test_file, intra_annotated_file, features_file , ae_features_file , all_features_file
from imported_files.statistical_feature import statistical

import pandas as pd
import numpy as np

def store_features(local_features_file, input_train_file):
    df = pd.read_csv(input_train_file, skiprows=2, header=None) 

    extracted_features = {col: statistical(df[col].values) for col in df}
    features_df = pd.DataFrame(extracted_features)
    features_df = features_df.T
    features_df.to_csv(local_features_file, index = False)
    
    print(features_df)

def merge_all_features(_features_file , _ae_features_file , _all_features_file):
    '''
    Merges all the features into a single csv file
    '''
    features_df = pd.read_csv(_features_file)
    AE_features = pd.read_csv(_ae_features_file)
    all_features_df = pd.concat([features_df , AE_features] , axis = 1)
    print(all_features_df)
    all_features_df.to_csv(_all_features_file , index=False)
    
def add_annotation(annotation_file, feature_file):
    '''
    ADD ANNOTATION TO FINAL FEATURES FILE IN COLUMN_1
    '''
    annotation_row = pd.read_csv(annotation_file, skiprows=1, nrows=1, header=None)
    target_df = pd.read_csv(feature_file)
    target_df.insert(0, 'annotation_row', annotation_row.iloc[0])
    target_df.to_csv(feature_file, index=False)

# store_features(features_file, intra_annotated_file)
merge_all_features(features_file , ae_features_file , all_features_file)
add_annotation(intra_annotated_file, all_features_file)
