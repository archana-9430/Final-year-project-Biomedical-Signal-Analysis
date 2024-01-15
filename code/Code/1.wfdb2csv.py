''''
This file converts the MIMIC III Waveform dataset (.dat and .hea files) to csv files. 
It extracts the column for PLETH, removes the NaN values and stores it to the csv file.
'''

path = '..\\Mixed_dataset\\MIMIC'
file_dir = '..\\Mixed_dataset\\MIMIC\\'
csv_path = 'Csv_data\\'


import os
import wfdb
import pandas as pd

def convert_to_csv(record_name, output_csv_path):
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
    

# Get the list of all .dat and .hea files
dir_list = os.listdir(path)
print(dir_list)
print(len(dir_list))

for i in range(len(dir_list)):
    if i%2 == 0:
        convert_to_csv(file_dir + dir_list[i][:-4], csv_path + dir_list[i][:-3] + "csv")

