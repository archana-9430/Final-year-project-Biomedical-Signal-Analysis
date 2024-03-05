import pandas as pd
from imported_files.paths_n_vars import all_features_file, zscore_features_file, min_max_features_file

def MyMinMaxScaler(dataframe : pd.DataFrame):
    return (dataframe - dataframe.min())/(dataframe.max() - dataframe.min())

def MyZScoreScaler(dataframe : pd.DataFrame):
    return (dataframe - dataframe.mean())/dataframe.std()
