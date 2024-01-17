''''
This file plots all the csv data. This is done to get a picture of the overall signal. 
It will help identify the desired 10min segments from larger datasets.
'''

csv_path = 'Csv_data'

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_data(csv_path, fig_num):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract the single column
    data_column = df.iloc[:, 0]

    # Plot the data
    plt.figure(fig_num)
    plt.plot(data_column)
    plt.title(csv_path)
    plt.xlabel("time")
    plt.ylabel("PPG Signal")
    plt.show()

dir_list = os.listdir(csv_path)
print(dir_list)

fig_num = 0
for csv_file in dir_list:
    fig_num += 1
    plot_csv_data(f"{csv_path}\\{csv_file}", fig_num)