import matplotlib.pyplot as plt
from pprint import pprint
from numpy import abs , sum , log10 , loadtxt , ndarray
import datatable as dt
from numpy.fft import fft , fftshift , fftfreq
import os

csv_fol = "10sec_segmented_data"
sampling_freq = 125
noise_cutoff = 20

def plot_signal(x : list , y : list , x_label = None , y_label = None , title = None):
    plt.grid(True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

def snr(data : ndarray , fSamp : float , csv_path : str = None):
    # print(f"{type(data)}    {data.shape}")
    data_fft = fft(data)
    len_fft = len(data_fft)
    noise_cut_samples = int(len_fft * (noise_cutoff / fSamp ))

    # # for plotting
    freq_list = [ f * fSamp/len_fft for f in range(len_fft) ]
    plot_signal( range(len_fft) , data , None , None , f"{csv_path}")
    plot_signal( freq_list , abs(data_fft) , None , None , f"FFT of {csv_path}")

    # according to Nyquist the maximum frequency is sampling frequency / 2 
    # max_freq = sampling_frequency / 2
    # max_freq_samp = int(len_fft * (max_freq / sampling_frequency))
    max_freq_samp = int(len_fft / 2 )

    noise_samples = data_fft[ noise_cut_samples : max_freq_samp ]
    signal_samples = data_fft[ : noise_cut_samples ]

    noise_enrg = sum( noise_samples.real ** 2 + noise_samples.imag ** 2)
    signal_enrg = sum( signal_samples.real ** 2 + signal_samples.imag **2)
    # print(f"Noise samps = {noise_cut_samples}")
    # print(f"Noise = {noise_enrg} , Signal = {signal_enrg}")

    return 10*log10(signal_enrg / noise_enrg)

def snr_csv(csv_path: str | None,
            sampling_frequency : float = sampling_freq) -> float:

    # if file is not csv then skip
    if "csv" not in csv_path.split('.'):
        return None
        
    data_numpy = loadtxt(csv_path , skiprows = 1 , delimiter = ",")
    # print(f"{csv_path} : {data_numpy}")

    return snr(data_numpy , sampling_frequency , csv_path)

def print_snr(folder_path : str ,
              sampling_frequency : int = sampling_freq,
              print_results : bool = True) -> dict:
    snr = {}
    files_list = os.listdir(folder_path)
    for f in files_list:
        file_path = f"{folder_path}\\{f}"
        snr[file_path] = snr_csv(file_path , sampling_frequency)
    if print_results:
        pprint(snr)

print_snr(csv_fol)
