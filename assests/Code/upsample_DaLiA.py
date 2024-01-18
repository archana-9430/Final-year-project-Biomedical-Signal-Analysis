'''
Takes 64Hz (specified by variable "original_samp_freq") DaLiA dataset 
and upsamples it to 125Hz (specified by variable "target_samp_freq")

Also upsamples the 32 Hz accelerometer signals to 125Hz

It also outputs a txt file (named "processed_files.txt") which have a list of csv files processed
by the script in "report_folder"
'''

csv_path = 'assests\Code\Csv_data'
csv_noise_path = "assests\Code\Csv_noise_DaLiA"
report_folder = "assests\Code"

# sampling frequencies in Hz
original_acc_samp_freq = 32
original_samp_freq = 64
target_samp_freq =125

# libraries
import matplotlib.pyplot as plt
from scipy.signal import resample
import pandas as pd
import numpy as np
import os

# keeps track of which csv files are processed
processed_files = []

def list_to_string(x : list):
    temp = ""
    for element in x:
        temp = f"{temp}\n{element}"
    return temp

def BVP_upsample(original_path , upsampled_path):

    org_data = pd.read_csv(original_path)

    data_list = org_data.iloc[ : , 0]  

    num_samples = int(len(data_list) * target_samp_freq / original_samp_freq)
    resampled_data_list = resample(data_list , num_samples)

    data_df = pd.DataFrame(resampled_data_list)
    data_df.to_feather(upsampled_path)
    processed_files.append(f"{original_path}")
    return

def ACC_upsample(original_acc_path , upsampled_path):

    acc_org_data = pd.read_csv(original_acc_path)

    data_list_x = np.array(acc_org_data.iloc[ : , 0])
    data_list_y = np.array(acc_org_data.iloc[ : , 1])
    data_list_z = np.array(acc_org_data.iloc[ : , 2])
    data_list_acc = np.sqrt(data_list_x**2 + data_list_y**2 + data_list_z**2)

    num_samples_acc = int(len(data_list_acc) * target_samp_freq / original_acc_samp_freq)
    resampled_data_list_acc = resample(data_list_acc , num_samples_acc)

    data_acc_df = pd.DataFrame(resampled_data_list_acc)
    data_acc_df.to_feather(upsampled_path)
    processed_files.append(f"{original_acc_path}")
    return

def recursive(object_path , entry_name):
    obj = os.scandir(object_path)
    for entry in obj:

        if entry.is_file() and len(entry.name.split('.')) > 1:
            if  entry.name == "BVP.csv":
                BVP_upsample(f"{object_path}\\{entry.name}" , f"{csv_path}\\{entry_name}_125Hz.fthr")
            
            if entry.name == "ACC.csv":
                ACC_upsample(f"{object_path}\\{entry.name}" , f"{csv_noise_path}\\{entry_name}_Noise_125Hz.fthr")

        elif entry.is_dir():
            recursive(f"{object_path}\\{entry.name}" , f"{entry.name}")

def write_csv_names():
    string_data = list_to_string(processed_files)
    print(string_data)
    csv_txt_path = f"{report_folder}\\processed_files.txt"

    if os.path.exists(csv_txt_path):
        with open(csv_txt_path , mode = "w") as txt_file:
            txt_file.write(string_data)
    else:   
        with open(csv_txt_path , mode = "x") as txt_file:
            txt_file.write(string_data)

# code
            
def DaLiA(dalia_org_fol):
    recursive(dalia_org_fol , "")
    write_csv_names()