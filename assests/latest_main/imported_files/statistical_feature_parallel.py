import pandas as pd
import numpy as np
import math
# import nolds
NAN_SUBSTITUTE = 0
from scipy.stats import entropy, skew, kurtosis
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft
import statistics
from EntropyHub import PermEn, ApEn , SampEn, K2En, DistEn, DispEn

# decorator for chacking for nan return values
import functools
def nan_check(function):
    @functools.wraps(function)
    def nan_check_wrapper(*arg , **kwargs):
        function_output = function(*arg , **kwargs)
        assert not np.any(pd.isna(function_output)) , f"ERROR::{function.__name__} returns nan at seg {i%16}"
     
        return function_output
    
    return nan_check_wrapper

import time
def time_custom(function):
    @functools.wraps(function)
    def time_custom_wrapper(*arg , **kwargs):
        start = time.perf_counter()
        function_output = function(*arg , **kwargs)
        print(f"{function.__name__} took {time.perf_counter() - start : 0.6f} seconds" )
        return function_output
    
    return time_custom_wrapper

#TIME DOMAIN
# @time_custom
def shannon_entropy(segment):
    # Calculate the probability distribution
    p = np.histogram(segment)[0]
    p_normalized = p / float(np.sum(p))
    p_normalized = p_normalized[np.nonzero(p_normalized)]
    
    # Compute the Shannon entropy
    H = entropy(p_normalized, base=2)
    return H

@time_custom
def symbolize(data, num_levels):
    min_val = np.min(data)
    max_val = np.max(data)
    step_size = (max_val - min_val) / num_levels
    
    symbols = np.floor((data - min_val) / step_size).astype(int)
    return symbols

@time_custom
def permutation_entropy(data, order, num_levels):
    symbols = symbolize(data, num_levels)
    patterns = [tuple(symbols[i:i+order]) for i in range(len(symbols) - order + 1)]
    _, counts = np.unique(patterns, return_counts=True, axis=0)
    probabilities = counts / len(patterns)
    entropy_val = -np.sum(probabilities * np.log2(probabilities))
    return entropy_val


@time_custom
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

@time_custom
def first_derivative_std(signal):
    first_derivative = np.gradient(signal , axis = 0)
    std_derivative = np.std(first_derivative , axis = 0)
    return std_derivative    
    
# @time_custom
def zero_crossing_rate(signal):
    zero_crossings = 0
    for i in range(1, len(signal)):
        # Check if the signal changes sign
        if (signal[i] >= 0 and signal[i - 1] < 0) or (signal[i] < 0 and signal[i - 1] >= 0):
            zero_crossings += 1
    # Return the zero crossing rate
    return zero_crossings / (len(signal) - 1)   

@time_custom
def mean_absolute_power(signal):
    abs_signal = np.abs(signal)
    map_signal = np.mean(abs_signal ** 2  , axis = 0)
    return map_signal

@time_custom
def rms(segment):
    rms = np.sqrt(np.mean(segment**2 ,axis = 0 ))
    return rms

@time_custom
def interquartile_range(segment):
    q3, q1 = np.percentile(segment , axis = 0 , q = [75 ,25])
    return (q3 - q1)
 
 
# @time_custom
def extract_rr_intervals(ppg_segment, sampling_rate):
    peaks, _ = find_peaks(ppg_segment, distance=sampling_rate/2)
    # Compute the time differences between consecutive peaks to obtain RR intervals
    rr_intervals = np.diff(peaks) / sampling_rate
    return rr_intervals   

# RMSSD (Root Mean Square of Successive Differences) is a measure of heart rate variability
# @time_custom
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
@time_custom
def mean_psd(segment, fs):
    f, psd = signal.welch(segment, fs=fs , axis = 0)
    '''Welch's method: Divide the signal into overlapping segments,
    computing the Fourier transform of each segment,
    and averaging the squared magnitudes of the resulting spectra to obtain the PSD estimate.
    '''
    std_psd = np.std(psd , axis = 0)
    # mean_psd = np.mean(psd)
    #accessing the maximum frequency from the index of the maximum power value in the PSD array
    dominant_frequency = f[np.argmax(psd , axis = 0)]
    # if dominant_frequency == 0:

    normalized_psd = psd / np.sum(psd , axis = 0)
    spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd) , axis = 0)
    return std_psd, dominant_frequency , spectral_entropy

@time_custom
def fourier_kurtosis(signal):
    fourier_transform = fft(signal , axis = 0)
    magnitude_spectrum = np.abs(fourier_transform)
    kurt = kurtosis(magnitude_spectrum , axis = 0 )
    return kurt

def statistical_parallel(segment : np.ndarray):
    '''
    Ensure the argument is a numpy array
    '''

    features = {}
    #Time domain:
    # print(type(segment))
    features['population_std'] = np.std(segment , axis = 0)
    features['sample_std'] = np.std(segment , axis = 0,  ddof=1)
    features['variance'] = np.std(segment , axis = 0) ** 2
    features['mean'] = np.mean(segment , axis = 0 )
    features['median'] = np.percentile(segment, axis = 0 , q = 50)
    features['q1'] = np.percentile(segment, axis = 0 , q = 25)
    features['q3'] = np.percentile(segment, axis = 0 , q = 75)
    features['min'] = np.min(segment , axis = 0)
    features['max'] = np.max(segment , axis = 0)
    features['range'] = np.max(segment , axis = 0) - np.min(segment , axis = 0)
    features['cov'] = np.std(segment , axis = 0) / np.mean(segment , axis = 0 )
    features['mean_abs_dev'] = np.mean(segment - np.mean(segment , axis = 0 ) , axis = 0)
    features['skewness'] = skew(segment , axis = 0)
    features['kurtosis'] = kurtosis(segment , axis = 0)
    # ENTROPY FEATURES

    # # features['permutation_entropy'] = permutation_entropy(segment, 3, 10)
    # PermEn1 = time_custom(PermEn)
    # ApEn1 = time_custom(ApEn)
    PermEnList = []
    ApEnList = []
    RMSSDList = []
    ShanEnList = []
    ZCRList = []
    for i in range(segment.shape[1]):
        _ , temp, _ = PermEn(segment[ : , i], m = 5, tau = 1)
        PermEnList.append(temp[-1])

        temp, _ = ApEn(segment[ : , i], m = 5, tau = 1)
        ApEnList.append( temp[-1])

        RMSSDList.append(rmssd(segment[ : , i]))
        ShanEnList.append(shannon_entropy(segment[ : , i]))
        ZCRList.append(zero_crossing_rate(segment[ : , i]))

    features['permutation_entropy_EN'] = np.array(PermEnList)
    features['Approx_entropy_EN'] = np.array(ApEnList)
    features['Shannon entropy'] = np.array(ShanEnList)

    # temp , _ , _ = SampEn(segment, m = 5, tau = 1)
    # features['Sample_entropy_EN'] = temp[-1]

    # temp , _ = K2En(segment, m = 2, tau = 1)
    # features['Kolmogorov_entropy_EN'] = temp[-1]

    # features['Distribution_entropy_EN'] , _ = DistEn(segment, m = 3, tau = 1)1

    # features['svd_entropy'] = svd_entropy(segment)
    
    # features['Shannon entropy'] = shannon_entropy(segment)
    features['first_derivative_std'] = first_derivative_std(segment)
    features['zero_crossing_rate'] = np.array(ZCRList)
    features['interquartile_range'] = interquartile_range(segment)
    features['mean_absolute_power'] = mean_absolute_power(segment)

    features['rms'] = rms(segment)
    features['rmssd'] = np.array(RMSSDList)
    # features['rmssd'] = rmssd(segment)

    #Frequency domain
    features['std_psd'], features['dominant_freq'] , features['spectral_entropy'] = mean_psd(segment, 125)
    features['fourier_kurtosis'] = fourier_kurtosis(segment)

    return features