import pandas as pd
import numpy as np
import math
# import nolds
NAN_SUBSTITUTE = 0
from scipy.stats import entropy, skew, kurtosis
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft

# decorator for chacking for nan return values
import functools
def nan_check(function):
    @functools.wraps(function)
    def nan_check_wrapper(*arg , **kwargs):
        function_output = function(*arg , **kwargs)
        assert not np.any(pd.isna(function_output)) , f"ERROR::{function.__name__} returns nan at seg {i%16}"
     
        return function_output
    
    return nan_check_wrapper

#TIME DOMAIN
def shannon_entropy(segment):
    # Calculate the probability distribution
    p = np.histogram(segment)[0]
    p_normalized = p / float(np.sum(p))
    p_normalized = p_normalized[np.nonzero(p_normalized)]
    
    # Compute the Shannon entropy
    H = entropy(p_normalized, base=2)
    return H

def symbolize(data, num_levels):
    min_val = np.min(data)
    max_val = np.max(data)
    step_size = (max_val - min_val) / num_levels
    symbols = np.floor((data - min_val) / step_size).astype(int)
    return symbols

def permutation_entropy(data, order, num_levels):
    symbols = symbolize(data, num_levels)
    patterns = [tuple(symbols[i:i+order]) for i in range(len(symbols) - order + 1)]
    _, counts = np.unique(patterns, return_counts=True, axis=0)
    probabilities = counts / len(patterns)
    entropy_val = -np.sum(probabilities * np.log2(probabilities))
    return entropy_val

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

def first_derivative_std(signal):
    first_derivative = np.gradient(signal)
    std_derivative = np.std(first_derivative)
    return std_derivative    
    
def zero_crossing_rate(signal):
    zero_crossings = 0
    for i in range(1, len(signal)):
        # Check if the signal changes sign
        if (signal[i] >= 0 and signal[i - 1] < 0) or (signal[i] < 0 and signal[i - 1] >= 0):
            zero_crossings += 1
    # Return the zero crossing rate
    return zero_crossings / (len(signal) - 1)    

def mean_absolute_power(signal):
    abs_signal = np.abs(signal)
    map_signal = np.mean(abs_signal ** 2)
    return map_signal

def rms(segment):
    rms = np.sqrt(np.mean(segment**2))
    return rms

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
    squared_diff = successive_diff ** 2
    mean_squared_diff = np.mean(squared_diff)
    if np.isnan(mean_squared_diff):
        mean_squared_diff = NAN_SUBSTITUTE
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
    # mean_psd = np.mean(psd)
    #accessing the maximum frequency from the index of the maximum power value in the PSD array
    dominant_frequency = f[np.argmax(psd)]
    normalized_psd = psd / np.sum(psd)
    spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd))
    return std_psd, dominant_frequency, spectral_entropy

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
    #Time domain:
    # print(type(segment))
    features['population_std'] = np.std(segment)
    features['sample_std'] = np.std(segment,  ddof=1)
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['permutation_entropy'] = permutation_entropy(segment, 3, 10)
    # features['svd_entropy'] = svd_entropy(segment)
    features['Shannon entropy'] = shannon_entropy(segment)
    features['first_derivative_std'] = first_derivative_std(segment)
    features['zero_crossing_rate'] = zero_crossing_rate(segment)
    features['interquartile_range'] = interquartile_range(segment)
    features['mean_absolute_power'] = mean_absolute_power(segment)
    features['mean'] = np.mean(segment)
    features['rms'] = rms(segment)
    features['rmssd'] = rmssd(segment)

    #Frequency domain
    features['std_psd'], features['dominant_freq'], features['spectral_entropy'] = mean_psd(segment, 125)
    features['fourier_kurtosis'] = fourier_kurtosis(segment)
    return features