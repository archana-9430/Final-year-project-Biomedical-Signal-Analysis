from imported_files.paths_n_vars import  filtered_merged, annotated_merged, sampling_frequency, annotated_data_fol

from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt
import numpy as np
import pandas as pd
import os

# input_path = annotated_merged
# output_path = filtered_merged

# ##SNR
# input_path = "SNR_data/clean_merged.csv"
# output_folder = "SNR_data"
# noise = "SNR_data/noise_seg_merged.csv"

input_folder = annotated_data_fol
output_folder = "5.all_filter_merged"

# filter specifications
order = 4
lower_cutoff = 0.2
higher_cutoff = 4
sampling_freq = sampling_frequency 

nyq = 0.5 * sampling_freq
low = lower_cutoff / nyq
high = higher_cutoff / nyq

def butter_bandpass():
    b, a = butter(order, [low, high], btype='band')
    return b, a

def cheby1_bandpass():
    b, a = cheby1(order, 5, [low, high], btype='band')
    return b, a

def cheby2_bandpass():
    b, a = cheby2(order, 40, [low, high], btype='band')
    return b, a

def ellip_bandpass():
    b, a = ellip(order, 3, 40, [low, high], btype='band')
    return b, a

def bandpass_filter(data, ftype='butter'):
    if ftype == 'butter':
        b, a = butter_bandpass()
    elif ftype == 'cheby1':
        b, a = cheby1_bandpass()
    elif ftype == 'cheby2':
        b, a = cheby2_bandpass()
    elif ftype == 'ellip':
        b, a = ellip_bandpass()
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

        filtered_data1 = bandpass_filter(ppg_values, "butter")
        filtered_data2 = bandpass_filter(ppg_values, "cheby1")
        filtered_data3 = bandpass_filter(ppg_values, "cheby2")
        filtered_data4 = bandpass_filter(ppg_values, "ellip")

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

#For SNR
# clean_signal = pd.read_csv(input_path, skiprows=1)
# noise_data = pd.read_csv(noise)

# def add_noise(clean_signal, noise_data):
#     clean_signal_array = clean_signal.to_numpy()
#     noise_data_array = noise_data.to_numpy()
#     num_segments = min(clean_signal_array.shape[1], noise_data_array.shape[1])
#     print("num_segments", num_segments)
#     noisy_signal = np.zeros(clean_signal_array.shape)
#     for i in range(num_segments):
#         noisy_signal[:, i] = clean_signal_array[:, i] + noise_data_array[:, i]
#     return noisy_signal

# def save_to_csv(filename, data):
#     df = pd.DataFrame(data)
#     df.to_csv(filename, index=False)
    
# noisy_signal = add_noise(clean_signal, noise_data)
# save_to_csv('SNR_data/clean_noise.csv', noisy_signal)

# ppg_values = pd.read_csv("SNR_data/clean_noise.csv")

# print("ppg_values.shape", ppg_values.shape)

# filtered_data1 = bandpass_filter(ppg_values, "butter")
# filtered_data2 = bandpass_filter(ppg_values, "cheby1")
# filtered_data3 = bandpass_filter(ppg_values, "cheby2")
# filtered_data4 = bandpass_filter(ppg_values, "ellip")

# print("filtered_data1.shape", filtered_data1.shape)

# save_to_csv("SNR_data/butter.csv", filtered_data1)
# save_to_csv("SNR_data/cheby1.csv", filtered_data2)
# save_to_csv("SNR_data/cheby2.csv", filtered_data3)
# save_to_csv("SNR_data/ellip.csv", filtered_data4)