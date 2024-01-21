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


def snr_csv(csv_path: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None,
            sampling_frequency:int = sampling_freq) -> float:

    # if file is not csv then skip
    if "csv" not in csv_path.split('.'):
        return None
        
    data = np.loadtxt(csv_path , delimiter = ',' , skiprows = 1)
    data_fft = fft(data)
    len_fft = len(data_fft)

    # # for plotting
    # rad_freq_list = [ f * 2 * np.pi/len_fft for f in range(len_fft) ]
    # plot_signal( range(len_fft) , data , None , None , f"{csv_path}")
    # plot_signal( rad_freq_list , np.abs(data_fft) , None , None , f"FFT of {csv_path} before shift")

    noise_cut_samples = int(len_fft * (noise_cutoff / sampling_frequency ))

    # according to Nyquist the maximum frequency is sampling frequency / 2 
    # max_freq = sampling_frequency / 2
    # max_freq_samp = int(len_fft * (max_freq / sampling_frequency))
    max_freq_samp = int(len_fft / 2 )

    noise_samples = data_fft[ noise_cut_samples : max_freq_samp ]
    signal_samples = data_fft[ : noise_cut_samples ]

    noise_enrg = np.sum( noise_samples.real ** 2 + noise_samples.imag ** 2)
    signal_enrg = np.sum( signal_samples.real ** 2 + signal_samples.imag **2)

    return 10*np.log10(signal_enrg/noise_enrg)

def print_snr(files_list : list ,
              sampling_frequency : int = sampling_freq,
              print_results : bool = True) -> dict:
    snr = {}
    
    for f in files_list:
        file_path = f"{uniform_data_path}\\{f}"
        snr[file_path] = snr_csv(file_path , sampling_frequency)
    if print_results:
        print(snr)

print_snr(os.listdir(uniform_data_path))
