import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = 'rf'
template_fol = 'templates'
scavenge_from = ['model_prediction/partly_corrupted','model_prediction/corrupted']
scavenge_save_fol = 'scavenged_data'
shift = int(0.2*125)

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

def Pearson_correl(sig1, sig2):
    return np.sum((sig1 - np.mean(sig1))*(sig2 - np.mean(sig2)))/(np.sqrt(np.var(sig1)*np.var(sig2))*len(sig1))

def visualize(actual_data, reference_data):
    fig, (ax_orig, ax_template) = plt.subplots(2, 1, figsize=(12 , 6))
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax_orig.plot(actual_data)
    ax_orig.set_title('Original signal')
    ax_orig.set_xlabel('Sample Number')
    ax_template.plot(reference_data)
    ax_template.set_title('Template')
    ax_template.set_xlabel('Sample Number')
    ax_orig.margins(0, 0.1)
    ax_template.margins(0, 0.1)
    fig.tight_layout()
    plt.legend()
    plt.show()

def scavenge_data(template_path , scavenging_path):
    # Load the template and PPG data from CSV files
    template_df = pd.read_csv(template_path)
    try:
        corrupt_ppg_df = pd.read_csv(scavenging_path).drop(index = 0)
    except:
        return pd.DataFrame()
    patient_num = re.findall(r'\d+', template_path)[0]

    # Assuming the template and PPG data are in 'value' column of the DataFrame, adjust as per your actual data structure
    template_data = template_df.values[:,0]
    ppg_data = corrupt_ppg_df.values

    # # Visualize the template and PPG data
    # visualize(ppg_data,template_data)

    # Calculate the cross-correlation between the template and a portion of the PPG data
    # You may adjust the portion of data used here for your analysis
    window_size = len(template_data)
    num_windows = int(corrupt_ppg_df.shape[0]/window_size)
    print('_'*100)
    print(f'PATIENT {patient_num}:')
    print('Number of windows = ',num_windows)

    count = 0
    scavenged_data_df = pd.DataFrame()
    for col_name, col_data in corrupt_ppg_df.items():
        for i in range(num_windows):
            window = col_data.values[i * shift : i * shift + window_size]
            pcc = Pearson_correl(window,template_data)
            # print('shape of window = ', window.shape)
            # print('shape of template_data = ', template_data.shape)
            # print("Pearson's correlation coefficient: ", pcc)

            # Visualize the correlation result
            if pcc > 0.86:
                count += 1
                scavenged_data_df[f'{col_name}_beat{count}'] = window

    if 'partly_corrupted' in scavenging_path.split('/'):
        print(f'scavenged beats from patient_{patient_num} (PC Segments) : {count}')
    if 'corrupted' in scavenging_path.split('/'):
        print(f'scavenged beats from patient_{patient_num} (C Segments) : {count}')

    return scavenged_data_df

def main_template_matching_():
    # calculate paths
    template_dir = f'{model}/{template_fol}'
    template_csv_list = [x for x in os.listdir(template_dir) if x.split('.')[-1] == 'csv']
    scavenge_dir =[ f'{model}/{x}' for x in scavenge_from]
    scavenged_save_dir = f'{model}/{scavenge_save_fol}'

    # now scavenge like hell
    for csv in template_csv_list:
        template_csv_path = f'{template_dir}/{csv}'
        scvng_pc_csv_path = f'{scavenge_dir[0]}/{csv}'
        save_path = f'{scavenged_save_dir}/{csv}'
        pc_scvngd_df = scavenge_data(template_csv_path, scvng_pc_csv_path)
        scvng_c_csv_path = f'{scavenge_dir[1]}/{csv}'
        c_scvngd_df = scavenge_data(template_csv_path, scvng_c_csv_path)
        scvngd_df = pd.concat([pc_scvngd_df , c_scvngd_df])
        scvngd_df.to_csv(save_path , index = False)


if __name__ == "__main__":
    
    import time
    start = time.perf_counter()

    main_template_matching_()

    
    elapsed = time.perf_counter() - start
    print(f'Time taken by {__file__} is {elapsed} seconds')


