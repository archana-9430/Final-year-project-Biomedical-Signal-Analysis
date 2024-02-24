from imported_files.paths_n_vars import inter_train_file, inter_test_file, intra_annotated_file, features_file
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

store_features(features_file, intra_annotated_file)