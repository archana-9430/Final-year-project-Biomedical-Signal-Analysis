''''
This file plots all the csv data. This is done to get a picture of the overall signal. 
It will help identify the desired 10min segments from larger datasets.
'''
import sys
 
sys.path.insert(0, 'F:/Shwashwat/B_Tech_ECE/project/Github folder/Final-year-project-Biomedical-Signal-Analysis/assests/latest_main/imported_files')

csv_path = 'F:/Shwashwat/B_Tech_ECE/project/Github folder/Final-year-project-Biomedical-Signal-Analysis/assests/latest_main/4.missclassifications'

import os
import pandas as pd
import matplotlib.pyplot as plt
# from imported_files.plot import plot_signal_interactive
'''
Provides customised interactive matplotlib functionality for annotation
'''
import matplotlib.pyplot as plt

def on_press(event):
    '''
    Callback function to handle keypress events
    '''
    # print('press', event.key)
    if event.key == 'escape':
        plt.close()

    elif event.key == 'm':
        # print("Zoom!!")
        plt.get_current_fig_manager().window.state('zoomed')



def plot_signal_interactive(x : range|list , y : list , style : str = 'b' , x_label : str = "" , y_label : str = "" , title : str = ""):
    '''
    Function plots 2D graphs with limited interactive functionalities
    '''
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax.plot(x , y , style)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_csv_data(csv_path, fig_num):
    # Read the CSV file
    df = pd.read_csv(csv_path,skiprows=1)

    # Extract the single column
    # data_column = df.iloc[:, 0]

    # # Plot the data
    # plt.figure(fig_num)
    # plt.plot(data_column)
    # plt.title(csv_path)
    # plt.xlabel("time")
    # plt.ylabel("PPG Signal")
    # plt.show()
    for _ , col in df.items():
        plot_signal_interactive(range(len(col.values)) , col.values , title = csv_path.split('/')[-1] + col.name)

dir_list = os.listdir(csv_path)
print(dir_list)

fig_num = 0
for csv_file in dir_list:
    fig_num += 1
    plot_csv_data(f"{csv_path}/{csv_file}", fig_num)