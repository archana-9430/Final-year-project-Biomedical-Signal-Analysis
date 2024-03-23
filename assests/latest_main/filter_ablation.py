import pandas as pd
import os
import re
# import random 
# random.seed()

import matplotlib.pyplot as plt

from matplotlib.widgets import Button, Slider
from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt
from imported_files.paths_n_vars import sampling_frequency

input_fol = '3.annotated_data'
f_low_range = [0.01 , 0.5]
f_high_range = [3 , 20]
f_low0 = 0.1
f_high0 = 3


# filter specifications
filter_order0 = 4
ord_range = [1 , 8]
lower_cutoff = f_low0
higher_cutoff = f_high0
filterType = 'butter'

def get_good_segments(fol_path : str) -> list:
    csv_files = [x for x in os.listdir(fol_path) if x.split('.')[-1] == 'csv']
    
    good_list = []
    
    for csv in csv_files:
        path = f'{fol_path}/{csv}'
        subject = str(re.findall(r'\d+', csv)[0])
        annotated_df = pd.read_csv(path)
        count = 0

        for col_name , col_series in annotated_df.items():
            if count >= 3:
                break
            if col_series.iloc[0] == 0:
                good_list.append((subject + '_' + col_name.split(' ')[1] , col_series.values[1:]))
                count += 1

    return good_list

def bandpass_filter(data, fs = 125, order:int=5, cutoffFreqHertz : list=[0.1 , 4], ftype:str='butter'):
    cutoffFreq = [2 * x / fs for x in cutoffFreqHertz]
    if ftype == 'butter':
        b, a = butter(order, cutoffFreq, btype='band')
    elif ftype == 'cheby1':
        b, a = cheby1(order, 5, cutoffFreq, btype='band')
    elif ftype == 'cheby2':
        b, a = cheby2(order, 40, cutoffFreq, btype='band')
    elif ftype == 'ellip':
        b, a = ellip(order, 3, 40, cutoffFreq, btype='band')
    else:
        raise ValueError("Invalid filter type. Choose from 'butter', 'cheby1', 'cheby2', 'ellip'.")
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def show_comparison(segment_name , segment_data):

    def update(val):
        f_order = s_order.val
        f_low = s_low_cut.val
        f_high = s_high_cut.val
        l.set_ydata(bandpass_filter(segment_data, sampling_frequency, f_order,
                                     [f_low , f_high], filterType))
        fig.canvas.draw_idle()

    def reset(event):
        s_order.reset()
        s_low_cut.reset()
        s_high_cut.reset()

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    ax.plot(segment_data,label='original')
    l, = ax.plot(bandpass_filter(segment_data, sampling_frequency, filter_order0,
                                     [f_low0 , f_high0], filterType),
                'r',lw=2, label='Filtered')
    ax.set_title(segment_name)

    ax_high_cut = fig.add_axes([0.1, 0.1, 0.65, 0.03])
    ax_low_cut = fig.add_axes([0.1, 0.15, 0.65, 0.03])
    ax_order = fig.add_axes([0.9, 0.1, 0.03, 0.65])

    # create the sliders
    s_order = Slider(
        ax_order, "Order", ord_range[0] , ord_range[1],
        orientation = 'vertical',
        # slidermin=f_low_range[0] , slidermax=f_low_range[1],
        valinit=filter_order0, valstep=1,
        color="green"
    )

    s_low_cut = Slider(
        ax_low_cut, "LowCut", f_low_range[0] , f_low_range[1],
        # slidermin=f_low_range[0] , slidermax=f_low_range[1],
        valinit=f_low0, valstep=[x/10 for x in range(6)],
        color="red"
    )

    s_high_cut = Slider(
        ax_high_cut, "HighCut", f_high_range[0], f_high_range[1],
        # slidermin=f_high_range[0], slidermax=f_high_range[1],
        valinit=f_high0, valstep=1,
        color='blue'
    )
    

    s_order.on_changed(update)
    s_low_cut.on_changed(update)
    s_high_cut.on_changed(update)
    ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(ax_reset, 'Reset', hovercolor='0.975')
    button.on_clicked(reset)
    plt.show()

print('script started')
good_segmn = get_good_segments(input_fol)
print(good_segmn)
print(f'{len(good_segmn) = }')
for x in good_segmn:
    print(f'{type(x) =}')
    print(f'{x =}')
    show_comparison(x[0] , x[1])
