''''
This file plots all the csv data. This is done to get a picture of the overall signal. 
It will help identify the desired 10min segments from larger datasets.
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
'''
Provides customised interactive matplotlib functionality for annotation
'''
import matplotlib.pyplot as plt

csv_path = os.getcwd()

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

def plot_csv_data(file_path, fig_num):
    # Read the CSV file
    df = pd.read_csv(file_path)
    df.drop(index = 0 , inplace=True)
    # Extract the single column
    # data_column = df.iloc[:, 0]

    # # Plot the data
    # plt.figure(fig_num)
    # plt.plot(data_column)
    # plt.title(file_path)
    # plt.xlabel("time")
    # plt.ylabel("PPG Signal")
    # plt.show()
    for col_name , col_data in df.items():
        plot_signal_interactive(range(len(col_data.values)) , col_data.values , title = file_path.split('/')[-1].split('.')[0] + ' ' + col_name)

dir_list = [x for x in os.listdir(csv_path) if x.split('.')[-1] == 'csv']
print(dir_list)

fig_num = 0
for csv_file in dir_list:
    fig_num += 1
    plot_csv_data(f"{csv_path}/{csv_file}", fig_num)