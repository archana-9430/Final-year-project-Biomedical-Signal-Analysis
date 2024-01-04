annotated_file = "Annotated_125Hz\\bidmc03m.csv"

import pandas as pd
import math
from scipy.stats import entropy, skew, kurtosis

def shanon_entropy(segment) -> int: 
    entr = 0
    for i in range(0, len(segment)):
        entr += segment[i]*math.log(segment[i], 2)
    return entr

def extract_features(segment):
    features = {}
    value_counts = segment.value_counts(normalize=True)
 
    features['entropy'] = entropy(value_counts, base=2)
    
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    
    return features

df = pd.read_csv(annotated_file, skiprows=2, header=None) 
#debugging
for col in df.columns:
    unique_values = df[col].value_counts()
    print(f"Column {col}: \n{unique_values}\n")
    
# Extract features from each segment (each column in the DataFrame)
extracted_features = {col: extract_features(df[col]) for col in df}

features_df = pd.DataFrame(extracted_features)
print(features_df.T)


