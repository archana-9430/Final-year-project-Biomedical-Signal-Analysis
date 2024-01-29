'''
This file plots any feather file you give in "path_to_feather"
'''

path_to_feather = "assests\Code\Csv_data\S11_125Hz.fthr"

import pandas as pd 
import matplotlib.pyplot as plt

def plot_feather(path):
    feather_df = pd.read_feather(path_to_feather)
    plt.plot(feather_df.iloc[ : , 0])
    plt.xlabel("Samples")
    plt.ylabel("PPG Signal")
    plt.title(path_to_feather)
    plt.grid(True)
    plt.show()

plot_feather(path_to_feather)