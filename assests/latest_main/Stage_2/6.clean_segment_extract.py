import re 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed()

model = 'rf'
predicted_clean_fol = 'model_prediction/clean'
save_clean_fol = 'clean segments'
template_fol = 'templates'

def separate_patientwise(segment_list : list):
    '''
    Segregates segments belonging to one patient from others, returs a list of lists
    Each sub list contains segments of only one patients 
    Assumes segment numbering as '0_9' where 0 is patientID and 9 is segment number
    '''
    # extract patient ID from segment ID and find unique patients
    patients = [element.split('_')[0] for element in segment_list]
    unique_patients = np.unique(patients)

    # segregate segments
    out = []
    for patient in unique_patients:
        temp = []
        for x in segment_list:
            if x.split('_')[0] == patient:
                temp.append(x)
        out.append(temp)
    return out

def save_clean_seg(source_path, save_path):
    # load data and drop annotation
    ppg_df = pd.read_csv(source_path)
    ppg_df.drop(index = 0 , inplace=True) 

    # separate patientwise
    segments_list = separate_patientwise(ppg_df.columns)

    # randomly choose any one good segment of each patient and save it in csv
    for patient_segms in segments_list:
        chosen = int(random.random()*len(patient_segms))
        print("chosen segment : ",chosen)
        segment = ppg_df[f'{patient_segms[chosen]}']
        segment.to_csv(save_path,index=False)

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

# beat extractor
def beat_select(clean_path : list):

    segment_df = pd.read_csv(clean_path)
    ppg = segment_df.values
    vpg = np.gradient(ppg, axis = 0)
    apg = np.gradient(vpg, axis = 0)
    jpg = np.gradient(apg, axis = 0)

    # identify onset points
    onset = [x for x in range(1,len(ppg)-1) if vpg[x-1] < 0  and vpg[x] > 0 and apg[x] > 0 and jpg[x] >= 0]
    # onset0 = [x for x in range(1,len(ppg)-1) if vpg[x-1] < 0 and vpg[x+1] > 0 and vpg[x] < 0 and jpg[x] >= 0]
    # onset1 = [x for x in range(1,len(ppg)-1) if vpg[x-1] < 0 and vpg[x+1] > 0 and vpg[x] < 0 and apg[x] > 0]
    # onset2 = [x for x in range(1,len(ppg)-1) if vpg[x-1] < 0 and vpg[x+1] > 0 and apg[x] > 0]
    # onset3 = [x for x in range(1,len(ppg)-1) if vpg[x-1] < 0 and vpg[x+1] > 0 ]
    # onset4 = [x for x in range(1,len(ppg)-1) if vpg[x-1] < 0 and vpg[x+1] > 0 and vpg[x] < 0 ]
    
    onset_values = [ppg[x] for x in onset]
    # onset_values0 = [ppg[x] for x in onset0]
    # onset_values1 = [ppg[x] for x in onset1]
    # onset_values2 = [ppg[x] for x in onset2]
    # onset_values3 = [ppg[x] for x in onset3]
    # onset_values4 = [ppg[x] for x in onset4]
    print('_'*50)
    pID = re.findall(r'\d+', clean_path)[0]
    print(f"PATIENT {pID}")

    # # visulaize the onset points
    # fig,ax = plt.subplots()
    # ax.plot(ppg)
    # ax.set_xlabel('Samples')
    # ax.set_ylabel('PPG signal')
    # ax.set_title(f'Onset points of {csv}')
    # ax.scatter(onset, onset_values, c = 'k', marker='x' , label = 'vpg,apg>0,jpg>=0')
    # # ax.scatter(onset, onset_values0, c = 'k', marker='x' , label = 'vpg,<0,jpg')
    # # ax.scatter(onset3, onset_values3,c = 'r', marker='|' , label = 'only vpg')
    # # ax.scatter(onset1, onset_values1,c = 'b', marker='+' , label = 'vpg,<0,apg')
    # # ax.scatter(onset2, onset_values2, c = 'g', marker='_' , label = 'vpg,apg')
    # # ax.scatter(onset4, onset_values4, c = 'm', marker=6 , label = 'vpg,<0')
    # plt.legend()
    # plt.show()

    # cut first 5 beats
    beat_intervals = [onset[i] - onset[i-1] for i in range(1,len(onset))]
    median_beat = np.median(beat_intervals)
    tolerance = np.std(beat_intervals)
    acceptable_beat_time = np.array([median_beat + tolerance , median_beat - tolerance] , dtype=int)

    print('beat intervals : ' , beat_intervals)
    print(median_beat)
    print(tolerance)
    print('acceptable_beat_time : ' , acceptable_beat_time)

    num_beat = 0
    i = 0
    beat_df = pd.DataFrame()
    while i < (len(beat_intervals)) and num_beat < 1:
        if beat_intervals[i] >= acceptable_beat_time[1] and beat_intervals[i] <= acceptable_beat_time[0]:
            num_beat += 1
            beat = ppg[onset[i] : onset[i+1] ].flatten()
            # print('shape of padded beat: ', beat)
            beat_df[f'beat {i}'] = beat
        i += 1

    return beat_df
    

def main_template_extract_():
    # calculate paths
    source_dir = model + '/' + predicted_clean_fol
    source_csv_list = [x for x in os.listdir(source_dir) if x.split('.')[-1] == 'csv']
    template_dir = model + '/' + template_fol
    
    save_dir = model + '/' + save_clean_fol
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(template_dir):
        os.mkdir(template_dir)

    # now create templates
    for csv in source_csv_list:
        clean_csv = source_dir + '/' + csv
        save_path = save_dir + '/' + csv
        template_path = template_dir + '/' + csv
        # beat wise segmentation and good beat selection
        save_clean_seg(clean_csv,save_path)
        beat_select(save_path).to_csv(template_path, index=False)

if __name__ == "__main__":
    
    import time
    start = time.perf_counter()

    main_template_extract_()

    
    elapsed = time.perf_counter() - start
    print(f'Time taken by {__file__} is {elapsed} seconds')



