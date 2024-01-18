''''
This file converts the MIMIC III Waveform dataset (.dat and .hea files) to csv files. 
It extracts the column for PLETH, removes the NaN values and stores it to the csv file.
It also extracts PPG data from BIDMC and MIMIC PERForm AF dataset and stores it to a csv file.
All the csv files generated is passed to a common directory.
'''
csv_path = 'Csv_data\\'
#MIMIC
mimic_path = '..\\Mixed_dataset\\MIMIC'
mimic_10Min = '..\\Mixed_dataset\\MIMIC_10MIn_125Hz'

#BIDMC
bidmc_csv_path = "..\\Mixed_dataset\\BIDMC\\bidmc_data_all_125Hz_8Min.csv"

#CSL Benchmark
csl_csv_path = "..\\Mixed_dataset\\csl_benchmark_data_125Hz_1hr\\csl_benchmark_data_125Hz_1hr.csv"

#MIMIC PERForm AF
perform_dir = "..\\Mixed_dataset\\mimic_perform_af_csv"

#DaLiA
dalia_path = "assests\\Mixed_dataset\\DaLiA Dataset"

import os
import wfdb
import pandas as pd
from upsample_DaLiA import DaLiA

def dat_to_csv(record_name, output_csv_path):
    # Read the record
    record = wfdb.rdrecord(record_name)

    # Extract the "PLETH" column
    pleth_column_index = record.sig_name.index("PLETH")
    pleth_data = record.p_signal[:, pleth_column_index]

    # Create a DataFrame with "PLETH" column
    df = pd.DataFrame({"PLETH": pleth_data})

    # Drop rows with NaN values in the "PLETH" column
    df = df.dropna(subset=['PLETH'])

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
     
def mimic():
    # Get the list of all .dat and .hea files
    dir_list = os.listdir(mimic_path)
    print(dir_list)
    print(len(dir_list))

    for i in range(len(dir_list)):
        if i%2 == 0:
            dat_to_csv(f"{mimic_path}\\{dir_list[i][:-4]}", csv_path + dir_list[i][:-3] + "csv")
            
def txt_2_csv(input_txt_path, output_csv_path):
    # Read the CSV file
    df = pd.read_csv(input_txt_path)
    df.to_csv(output_csv_path, index=False)
    
def mimic_10_min():
    # Get the list of all .dat and .hea files
    dir_list = os.listdir(mimic_10Min)
    print(dir_list)
    print(len(dir_list))
    
    for i in range(len(dir_list)):    
        txt_2_csv(f"{mimic_10Min}\\{dir_list[i]}", csv_path + dir_list[i][:-3] + "csv")
              
def split_csv_columns(input_csv_path, csv_path):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Iterate over columns and create separate CSV files
    for column_name in df.columns:
        column_data = df[column_name]
        output_csv_path = csv_path + f"{column_name}.csv"
        column_data.to_csv(output_csv_path, index=False)
        
def bidmc():
    split_csv_columns(bidmc_csv_path, csv_path)
    
def csl():
    split_csv_columns(csl_csv_path, csv_path)
    
def extract_ppg_col(input_csv_path, output_csv_path):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Extract the "PPG" column
    ppg_column = df["PPG"]
    # Create a new DataFrame with only the "PPG" column
    ppg_df = pd.DataFrame({"PPG": ppg_column})

    # Save the new DataFrame to a CSV file
    ppg_df.to_csv(output_csv_path, index=False)
    
def mimic_perform_af():
    dir_list = os.listdir(perform_dir)
    print(dir_list)
    print(len(dir_list))
    # Filter only CSV files
    csv_files = [file for file in dir_list if file.lower().endswith('.csv')]
    for csv_input in csv_files:
        output_csv_path = csv_path + f"{csv_input}"
        extract_ppg_col(f"{perform_dir}\\{csv_input}", output_csv_path)


##Uncomment this code for running this code
# mimic()
# mimic_10_min()
# bidmc()
# mimic_perform_af()
# csl()
# DaLiA(dalia_path)