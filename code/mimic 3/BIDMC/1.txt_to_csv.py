textfile_path = "Data_125Hz\\bidmc03m.txt" 
csv_path = "CSV_Data_125Hz\\bidmc03m.csv"

import pandas as pd

def convert_file(textfile_path):
    with open(textfile_path, 'r') as input_file:
        data_with_commas = input_file.read()

    # Splitting the string by commas and joining with newline character '\n'
    data_with_enter = "\n".join(data_with_commas.split(','))

    # Writing the separated data to a new file
    with open(textfile_path, 'w') as output_file:
        output_file.write(data_with_enter)
        
def text_to_csv(input_file, output_file):
    df = pd.read_csv(input_file, delimiter = ',',header=None) 
    df.to_csv(output_file, index=False)
    
def get_rows_columns(csv_file):
    df = pd.read_csv(csv_file)
    num_rows = len(df)
    num_columns = len(df.columns)
    return num_rows, num_columns

convert_file(textfile_path)
text_to_csv(textfile_path, csv_path)
print(get_rows_columns(csv_path))