import functools
import time
def time_custom(function):
    @functools.wraps(function)
    def time_custom_wrapper(*arg , **kwargs):
        start = time.perf_counter()
        function_output = function(*arg , **kwargs)
        print(f"{function.__name__} took {time.perf_counter() - start : 0.6f} seconds" )
        return function_output
    
    return time_custom_wrapper

import numpy as np
from scipy.stats import  skew, kurtosis
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft
from ordpy import permutation_entropy
import pywt
from EntropyHub import  PermEn ,ApEn , SampEn, K2En, DistEn, DispEn

NAN_SUBSTITUTE = 0

PermEn = time_custom(PermEn)
permutation_entropy = time_custom(permutation_entropy)
ApEn = time_custom(ApEn)

#TIME DOMAIN
def shannon_entropy(segment):
    segment_sqred = segment**2
    return np.sum(segment_sqred * np.log2(segment_sqred))

# def symbolize(data, num_levels):
#     min_val = np.min(data)
#     max_val = np.max(data)
#     step_size = (max_val - min_val) / num_levels
    
#     symbols = np.floor((data - min_val) / step_size).astype(int)
#     return symbols

# def permutation_entropy(data, order, num_levels):
#     symbols = symbolize(data, num_levels)
#     patterns = [tuple(symbols[i:i+order]) for i in range(len(symbols) - order + 1)]
#     _, counts = np.unique(patterns, return_counts=True, axis=0)
#     probabilities = counts / len(patterns)
#     entropy_val = -np.sum(probabilities * np.log2(probabilities))
#     return entropy_val


def svd_entropy(data):
    # Convert pandas Series to numpy array
    # data_array = data.values
    data_array = data.reshape(-1, 1)
    # print(f"svd_entropy :: data_array {data_array}")
    # print(f"svd_entropy :: data_array shape {data_array.shape}")
    s = np.linalg.svd(data_array ,  compute_uv = False)
    # print(f"svd_entropy :: s {s}")
    norm_s = s / np.sum(s)
    entropy_val = -np.sum(norm_s * np.log(norm_s))
    # print(f"svd_entropy :: norm_s  {norm_s }\n")
    
    return entropy_val

def first_derivative_std(segment):
    first_derivative = np.gradient(segment)
    std_derivative = np.std(first_derivative)
    return std_derivative   
 
def Hjorth_parameters(segment):
    first_derivative = np.gradient(segment)
    signal_hjorth_mobility = np.sqrt(np.var(first_derivative)/np.var(segment))
    derivative_hjorth_mobility = np.sqrt(np.var(np.gradient(first_derivative))/np.var(first_derivative))
    hjorth_complexity = derivative_hjorth_mobility/signal_hjorth_mobility
    return signal_hjorth_mobility,hjorth_complexity    
    
def zero_crossing_rate(segment):
    zero_crossings = 0
    segment = segment - np.mean(segment)
    for i in range(1, len(segment)):
        # Check if the segment changes sign
        if (segment[i] >= 0 and segment[i - 1] < 0) or (segment[i] < 0 and segment[i - 1] >= 0):
            zero_crossings += 1
    # Return the zero crossing rate
    return zero_crossings / (len(segment) - 1)   

def mean_absolute_power(segment):
    abs_signal = np.abs(segment)
    map_signal = np.mean(abs_signal ** 2)
    return map_signal

def rms(segment):
    value = np.sqrt(np.mean(segment**2))
    return value

def interquartile_range(segment):
    q3, q1 = np.percentile(segment , [75 ,25])
    return (q3 - q1)
 
def extract_rr_intervals(ppg_segment, sampling_rate):
    peaks, _ = find_peaks(ppg_segment, distance=sampling_rate/2)
    # Compute the time differences between consecutive peaks to obtain RR intervals
    rr_intervals = np.diff(peaks) / sampling_rate
    return rr_intervals   

# RMSSD (Root Mean Square of Successive Differences) is a measure of heart rate variability
def rmssd(segment):
    rr_intervals = extract_rr_intervals(segment, 125)
    successive_diff = np.diff(rr_intervals)

    if len(successive_diff) == 0: # avoids nan output
        successive_diff = NAN_SUBSTITUTE

    squared_diff = successive_diff ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmssd_value = np.sqrt(mean_squared_diff)
    return rmssd_value
    
#FREQUENCY DOMAIN
def mean_psd(segment, fs):
    f, psd = signal.welch(segment, fs=fs)
    '''Welch's method: Divide the signal into overlapping segments,
    computing the Fourier transform of each segment,
    and averaging the squared magnitudes of the resulting spectra to obtain the PSD estimate.
    '''
    std_psd = np.std(psd)
    mean_psd = np.mean(psd)
    #accessing the maximum frequency from the index of the maximum power value in the PSD array
    dominant_frequency = f[np.argmax(psd)]
    # if dominant_frequency == 0:

    normalized_psd = psd / np.sum(psd)
    spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd))
    return std_psd, dominant_frequency , spectral_entropy, mean_psd

def boibs(segment):
    # Compute the FFT
    fft_result = np.fft.fft(segment)

    peaks, _ = find_peaks(np.abs(fft_result))
    amplitudes = np.abs(fft_result[peaks])
    frequencies = np.fft.fftfreq(len(signal))[peaks]
    return amplitudes, frequencies

def fourier_kurtosis(signal):
    fourier_transform = fft(signal)
    magnitude_spectrum = np.abs(fourier_transform)
    kurt = kurtosis(magnitude_spectrum)
    return kurt

def statistical(segment : np.ndarray):
    '''
    Ensure the argument is a numpy array
    '''
    features = {}
    
    ''' TIME DOMAIN: '''
    # print(type(segment))
    features['population_std'] = np.std(segment)
    features['sample_std'] = np.std(segment,  ddof=1)
    features['variance'] = np.std(segment) ** 2
    features['mean'] = np.mean(segment)
    features['median'] = np.percentile(segment, 50)
    features['q1'] = np.percentile(segment, 25)
    features['q3'] = np.percentile(segment, 75)
    features['max'] = max(segment)
    features['min'] = min(segment)
    features['range'] = max(segment) - min(segment)
    features['cov'] = np.std(segment) / np.mean(segment)
    features['mean_abs_dev'] = np.mean(np.abs(segment - np.mean(segment)))
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['first_derivative_std'] = first_derivative_std(segment)
    features['Hjorth mobility'] , features['Hjorth complexity'] = Hjorth_parameters(segment)
    features['zero_crossing_rate'] = zero_crossing_rate(segment)#~~
    features['interquartile_range'] = interquartile_range(segment)
    features['mean_absolute_power'] = mean_absolute_power(segment)
    features['rms'] = rms(segment)
    features['rmssd'] = rmssd(segment)
    
    ''' ENTROPY FEATURES '''
    # features['permutation_entropy'] = permutation_entropy(segment, 3, 10)
    # _ , temp, _ = PermEn(segment, m = 5, tau = 1)
    # features['permutation_entropy_EN'] = temp[-1]
    features['permutation_entropy_EN'] = permutation_entropy(segment , dx = 5, normalized = True)
    
    temp, _ = ApEn(segment, m = 5, tau = 1)
    features['Approx_entropy_EN'] = temp[-1]

    # temp , _ , _ = SampEn(segment, m = 5, tau = 1)
    # features['Sample_entropy_EN'] = temp[-1]

    # temp , _ = K2En(segment, m = 2, tau = 1)
    # features['Kolmogorov_entropy_EN'] = temp[-1]

    # features['Distribution_entropy_EN'] , _ = DistEn(segment, m = 3, tau = 1)1

    # features['svd_entropy'] = svd_entropy(segment)
    
    features['Shannon entropy'] = shannon_entropy(segment)#~~

    ''' FREQUENCY DOMAIN '''
    features['std_psd'], features['dominant_freq'] , features['spectral_entropy'], features['mean_psd'] = mean_psd(segment, 125)
    features['fourier_kurtosis'] = fourier_kurtosis(segment)
    features['fft_amplitude'], features['fft_frequency'] = boibs(segment)
    
    '''DWT features'''
    cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(segment, 'db1', mode='symmetric', level=4, axis=-1)
    features['dwt_kurtosis'] = kurtosis(cA4)
    features['dwt_skew'] = skew(cA4)    
    features['dwt_mean_absolute_power'] = mean_absolute_power(cA4)    
    features['dwt_permutation_entropy_EN'] = permutation_entropy(cA4 , dx = 5, normalized = True)
    temp, _ = ApEn(cA4, m = 5, tau = 1)
    features['dwt_Approx_entropy_EN'] = temp[-1]
    features['dwt_Hjorth_mobility'] , features['dwt_Hjorth_complexity'] = Hjorth_parameters(cA4)
    features['dwt_variance'] = np.var(cA4)
    return features
