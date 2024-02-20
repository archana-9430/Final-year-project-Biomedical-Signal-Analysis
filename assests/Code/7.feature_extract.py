annotated_file = "Annotated_125Hz\\bidmc03m.csv"
features_file = "features_125Hz\\bidmc03m.csv"

import pandas as pd
import math
import numpy as np
from scipy.stats import entropy, skew, kurtosis

def shannon_entropy(segment):
    # Calculate the probability distribution
    p = np.histogram(segment)[0]
    p_normalized = p / float(np.sum(p))
    p_normalized = p_normalized[np.nonzero(p_normalized)]
    
    # Compute the Shannon entropy
    H = entropy(p_normalized, base=2)
    
    return H

def extract_features(segment):
    features = {}
    # value_counts = segment.value_counts(normalize=True)
 
    # features['entropy'] = entropy(value_counts, base=2)
    features['Shannon entropy'] = shannon_entropy(segment)
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    
    return features

def store_features(features_file, annotated_file):
    df = pd.read_csv(annotated_file, skiprows=2, header=None) 
    extracted_features = {col: extract_features(df[col]) for col in df}
    features_df = pd.DataFrame(extracted_features)
    features_df = features_df.T
    features_df.to_csv(features_file, index = False)
    
    print(features_df)
    
store_features(features_file, annotated_file)

#debugging
# for col in df.columns:
#     unique_values = df[col].value_counts()
#     print(f"Column {col}: \n{unique_values}\n")