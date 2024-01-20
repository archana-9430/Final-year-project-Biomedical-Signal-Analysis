import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.fft import fft , fftshift , fftfreq
import os

uniform_data_path = "uniform_csv_data"
sampling_freq = 125
noise_cutoff = 20

def plot_signal(x : list ,y : list , x_label = None , y_label = None , title = None):
    plt.grid(True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()


def snr_csv(csv_path , sampling_frequency:int = sampling_freq) -> float:
    if "csv" not in csv_path.split('.'):
        return None
    data = np.loadtxt(csv_path , delimiter = ',' , skiprows = 1)
    data_fft = fft(data)
    len_fft = len(data_fft)

    # # for plotting
    # freq_list = [ f * sampling_frequency/len_fft for f in range(len_fft) ]
    # plot_signal( range(len_fft) , data , None , None , f"{csv_path}")
    # plot_signal( freq_list , np.abs(data_fft) , None , None , f"FFT of {csv_path} before shift")

    noise_cut_samples = int(np.ceil(len_fft * (noise_cutoff / sampling_frequency )))

    # according to Nyquist the maximum frequency is sampling frequency / 2 
    max_freq_samp = int(len_fft / 2 )

    noise_samples = data_fft[ noise_cut_samples : max_freq_samp ]
    signal_samples = data_fft[ : noise_cut_samples ]

    noise_pwr = np.sum( (np.abs(noise_samples)) ** 2)
    signal_pwr = np.sum( (np.abs(signal_samples)) ** 2)

    return 10*np.log10(signal_pwr/noise_pwr)

def print_snr(files_list):
    snr = 0

    print("\t\t File \t\t\t\t\t SNR (dB)")

    for f in files_list:
        file_path = f"{uniform_data_path}\\{f}"
        snr = snr_csv(file_path , sampling_freq)
        print(f"{file_path}\t\t{snr}")

print_snr(os.listdir(uniform_data_path))