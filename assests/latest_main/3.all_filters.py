from imported_files.paths_n_vars import  filtered_merged, annotated_merged, sampling_frequency, annotated_data_fol

from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt
import numpy as np
import pandas as pd
import os

input_path = annotated_merged
output_path = filtered_merged

input_folder = annotated_data_fol
output_folder = "5.all_filter_merged"

# filter specifications
filter_order = 4
lower_cutoff = 0.2
higher_cutoff = 4
sampling_freq = sampling_frequency 

nyq = 0.5 * sampling_freq
low = lower_cutoff / nyq
high = higher_cutoff / nyq

def butter_bandpass(order=5):
    b, a = butter(order, [low, high], btype='band')
    return b, a

def cheby1_bandpass(order=5):
    b, a = cheby1(order, 5, [low, high], btype='band')
    return b, a

def cheby2_bandpass(order=5):
    b, a = cheby2(order, 40, [low, high], btype='band')
    return b, a

def ellip_bandpass(order=5):
    b, a = ellip(order, 3, 40, [low, high], btype='band')
    return b, a

def bandpass_filter(data, order=5, ftype='butter'):
    if ftype == 'butter':
        b, a = butter_bandpass(order=order)
    elif ftype == 'cheby1':
        b, a = cheby1_bandpass(order=order)
    elif ftype == 'cheby2':
        b, a = cheby2_bandpass(order=order)
    elif ftype == 'ellip':
        b, a = ellip_bandpass(order=order)
    else:
        raise ValueError("Invalid filter type. Choose from 'butter', 'cheby1', 'cheby2', 'ellip'.")
    filtered_data = filtfilt(b, a, data)
    return filtered_data


file_list = os.listdir(input_folder)
print(f"{file_list}, {len(file_list)}")

for csv_file in file_list:
    if csv_file.split('.')[-1] == 'csv':
        df = pd.read_csv(input_folder + "/" + csv_file)
        annotation = df.iloc[0]
        ppg_values = df.values[1:]

        filtered_data1 = bandpass_filter(ppg_values, 4, "butter")
        filtered_data2 = bandpass_filter(ppg_values, 4, "cheby1")
        filtered_data3 = bandpass_filter(ppg_values, 4, "cheby2")
        filtered_data4 = bandpass_filter(ppg_values, 4, "ellip")

        anno_filt1 = np.insert(filtered_data1,0,values=annotation.values , axis = 0)
        anno_filt2 = np.insert(filtered_data2,0,values=annotation.values , axis = 0)
        anno_filt3 = np.insert(filtered_data3,0,values=annotation.values , axis = 0)
        anno_filt4= np.insert(filtered_data4,0,values=annotation.values , axis = 0)
        
        
        merged_df1 = pd.DataFrame(data = anno_filt1 , columns = df.columns)
        merged_df1.to_csv(output_folder + "/" + "butter/" + csv_file , index = False)
        merged_df2 = pd.DataFrame(data = anno_filt1 , columns = df.columns)
        merged_df2.to_csv(output_folder + "/" + "cheby1/" + csv_file , index = False)
        merged_df3 = pd.DataFrame(data = anno_filt1 , columns = df.columns)
        merged_df3.to_csv(output_folder + "/cheby2/" + csv_file , index = False)
        merged_df4 = pd.DataFrame(data = anno_filt1 , columns = df.columns)
        merged_df4.to_csv(output_folder + "/ellip/" + csv_file , index = False)
