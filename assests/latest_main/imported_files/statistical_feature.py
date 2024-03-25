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
sampling_rate = 125

PermEn = time_custom(PermEn)
permutation_entropy = time_custom(permutation_entropy)
ApEn = time_custom(ApEn)

#TIME DOMAIN
def shannon_entropy(segment):
#     segment_sqred = segment**2
#     return np.sum(segment_sqred * np.log2(segment_sqred))
    p_signal = np.abs(signal) / np.sum(np.abs(signal))  # Calculate probability distribution
    sh_entropy = -np.sum(p_signal * np.log2(p_signal))  # Calculate Shannon entropy
    return sh_entropy

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

def calculate_bpm(ppg_signal, sampling_rate):
    peaks, _ = find_peaks(ppg_signal, height=0)  # Find peaks in the PPG signal
    bpm = len(peaks) / len(ppg_signal) * sampling_rate * 60  # Calculate beats per minute
    return bpm

def calculate_IPI(ppg_signal, sampling_rate):
    peaks, _ = find_peaks(ppg_signal, height=0)  # Find peaks in the PPG signal
    IPI = np.diff(peaks) / sampling_rate  # Calculate inter-pulse interval
    return IPI

def statistical(segment : np.ndarray):
    '''
    Ensure the argument is a numpy array
    '''
    features = {}
    
    ''' TIME DOMAIN: '''
    # print(type(segment))
    features['mean'] = np.mean(segment)
    features['median'] = np.percentile(segment, 50)
    features['population_std'] = np.std(segment)
    features['sample_std'] = np.std(segment,  ddof=1)
    features['variance'] = np.std(segment) ** 2
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['max'] = max(segment)
    features['min'] = min(segment)
    features['range'] = max(segment) - min(segment)
    features['rms'] = rms(segment)
    features['zero_crossing_rate'] = zero_crossing_rate(segment)
    features['mean_abs_dev'] = np.mean(np.abs(segment - np.mean(segment)))
    features['interquartile_range'] = interquartile_range(segment)
    features['mean_absolute_power'] = mean_absolute_power(segment)
    features['mean_absolute_diff'] = np.mean(np.abs(np.diff(segment)))
    features['mean_diff_absolute_successive'] = np.mean(np.diff(np.abs(segment)))
    features['median_absolute_diff'] = np.median(np.abs(np.diff(signal)))
    features['std_absolute_diff'] = np.std(np.abs(np.diff(signal)))
    features['variance_absolute_diff'] = np.var(np.abs(np.diff(signal)))
    features['rms_successive_diff'] = np.sqrt(np.mean(np.diff(signal)**2))
    features['sum_absolute_diff'] = np.sum(np.abs(np.diff(signal)))
    features['maximal_deviation'] = np.max(np.abs(signal))
    
    features['q1'] = np.percentile(segment, 25)
    features['q3'] = np.percentile(segment, 75)
    features['first_derivative_std'] = first_derivative_std(segment)
    features['Hjorth mobility'] , features['Hjorth complexity'] = Hjorth_parameters(segment)
    features['rmssd'] = rmssd(segment)
   
    ''' FREQUENCY DOMAIN '''
    f, psd = signal.welch(segment, fs=125)
    '''Welch's method: Divide the signal into overlapping segments,
    computing the Fourier transform of each segment,
    and averaging the squared magnitudes of the resulting spectra to obtain the PSD estimate.
    '''
    geometric_mean = np.exp(np.mean(np.log(psd)))
    arithmetic_mean = np.mean(psd)
    
    features['std_psd'] = np.std(psd)
    features['mean_psd'] = np.mean(psd)
    features['dominant_frequency'] = f[np.argmax(psd)]
    features['total_power'] = np.sum(psd)
    features['normalized_psd'] = psd / np.sum(psd)
    features['spectral_entropy'] = -np.sum(psd / np.sum(psd) * np.log2(psd / np.sum(psd)))
    features['normalized_psd'] = psd / np.sum(psd)
    features['spectral_centroid'] = np.sum(f * psd) / np.sum(psd)
    features['spectral_flatness'] = geometric_mean / arithmetic_mean
    features['spectral_skewness'] = skew(psd)
    features['spectral_kurtosis'] = kurtosis(psd)
    features['spectral_kurtosis'] = kurtosis(psd)
    features['fourier_kurtosis'] = fourier_kurtosis(segment)
    # features['fft_amplitude'], features['fft_frequency'] = boibs(segment)
    

    ''' ENTROPY FEATURES '''
    temp , _ , _ = SampEn(segment, m = 5, tau = 1)
    features['Sample_entropy_EN'] = temp[-1]
    
    temp, _ = ApEn(segment, m = 5, tau = 1)
    features['Approx_entropy_EN'] = temp[-1]
    
    features['permutation_entropy'] = permutation_entropy(segment, 3, 10)
    _ , temp, _ = PermEn(segment, m = 5, tau = 1)
    
    # features['permutation_entropy_EN'] = temp[-1]
    # features['permutation_entropy_EN'] = permutation_entropy(segment , dx = 5, normalized = True)
    
    features['Shannon entropy'] = shannon_entropy(segment)
    
    # temp , _ = K2En(segment, m = 2, tau = 1)
    # features['Kolmogorov_entropy_EN'] = temp[-1]

    # features['Distribution_entropy_EN'] , _ = DistEn(segment, m = 3, tau = 1)1

    # features['svd_entropy'] = svd_entropy(segment)

    
    '''DWT features'''
    cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(segment, 'db1', mode='symmetric', level=4, axis=-1)
    features['cA4'] = cA4
    features['cD4'] = cD4
    features['cD3'] = cD3
    features['cD2'] = cD2
    features['dwt_kurtosis'] = kurtosis(cA4)
    features['dwt_skew'] = skew(cA4)    
    features['dwt_mean_absolute_power'] = mean_absolute_power(cA4)    
    features['dwt_permutation_entropy_EN'] = permutation_entropy(cA4 , dx = 5, normalized = True)
    temp, _ = ApEn(cA4, m = 5, tau = 1)
    features['dwt_Approx_entropy_EN'] = temp[-1]
    features['dwt_Hjorth_mobility'] , features['dwt_Hjorth_complexity'] = Hjorth_parameters(cA4)
    features['dwt_variance'] = np.var(cA4)
    features['population_std'] = np.std(cA4)
    features['sample_std'] = np.std(cA4,  ddof=1)
    features['variance'] = np.std(cA4) ** 2
    features['mean'] = np.mean(cA4)
    features['median'] = np.percentile(cA4, 50)
    features['q1'] = np.percentile(cA4, 25)
    features['q3'] = np.percentile(cA4, 75)
    features['max'] = max(cA4)
    features['min'] = min(cA4)
    features['range'] = max(cA4) - min(cA4)
    features['cov'] = np.std(cA4) / np.mean(cA4)
    features['mean_abs_dev'] = np.mean(np.abs(cA4 - np.mean(cA4)))
    features['skewness'] = skew(cA4)
    features['kurtosis'] = kurtosis(cA4)
    features['first_derivative_std'] = first_derivative_std(cA4)
    features['Hjorth mobility'] , features['Hjorth complexity'] = Hjorth_parameters(cA4)
    features['zero_crossing_rate'] = zero_crossing_rate(cA4)#~~
    features['interquartile_range'] = interquartile_range(cA4)
    features['mean_absolute_power'] = mean_absolute_power(cA4)
    features['rms'] = rms(cA4)
    features['rmssd'] = rmssd(cA4)
    
    
    #DWT of first derivative
    first_derivative = np.gradient(segment)
    cA4_d, cD4, cD3, cD2, cD1 = pywt.wavedec(first_derivative, 'db1', mode='symmetric', level=4, axis=-1)
    features['dwt_kurtosis_d'] = kurtosis(cA4_d)
    features['dwt_skew_d'] = skew(cA4_d)    
    features['dwt_mean_absolute_power_d'] = mean_absolute_power(cA4_d)    
    features['dwt_permutation_entropy_EN_d'] = permutation_entropy(cA4_d , dx = 5, normalized = True)
    temp, _ = ApEn(cA4_d, m = 5, tau = 1)
    features['dwt_Approx_entropy_EN_d'] = temp[-1]
    features['dwt_Hjorth_mobility_d'] , features['dwt_Hjorth_complexity_d'] = Hjorth_parameters(cA4_d)
    features['dwt_variance_d'] = np.var(cA4_d)
    features['population_std_d'] = np.std(cA4_d)
    features['sample_std_d'] = np.std(cA4_d,  ddof=1)
    features['variance_d'] = np.std(cA4_d) ** 2
    features['mean_d'] = np.mean(cA4_d)
    features['median_d'] = np.percentile(cA4_d, 50)
    features['q1_d'] = np.percentile(cA4_d, 25)
    features['q3_d'] = np.percentile(cA4_d, 75)
    features['max_d'] = max(cA4_d)
    features['min_d'] = min(cA4_d)
    features['range_d'] = max(cA4_d) - min(cA4_d)
    features['cov_d'] = np.std(cA4_d) / np.mean(cA4_d)
    features['mean_abs_dev_d'] = np.mean(np.abs(cA4_d - np.mean(cA4_d)))
    features['skewness_d'] = skew(cA4_d)
    features['kurtosis_d'] = kurtosis(cA4_d)
    features['first_derivative_std_d'] = first_derivative_std(cA4_d)
    features['Hjorth mobility_d'] , features['Hjorth complexity_d'] = Hjorth_parameters(cA4_d)
    features['zero_crossing_rate_d'] = zero_crossing_rate(cA4_d)#~~
    features['interquartile_range_d'] = interquartile_range(cA4_d)
    features['mean_absolute_power_d'] = mean_absolute_power(cA4_d)
    features['rms_d'] = rms(cA4_d)
    features['rmssd_d'] = rmssd(cA4_d)
    
    
    #DWT of second derivative
    second_derivative = np.gradient(segment)
    cA4_d2, cD4, cD3, cD2, cD1 = pywt.wavedec(second_derivative, 'db1', mode='symmetric', level=4, axis=-1)
    features['dwt_kurtosis_d2'] = kurtosis(cA4_d2)
    features['dwt_skew_d2'] = skew(cA4_d2)    
    features['dwt_mean_absolute_power_d2'] = mean_absolute_power(cA4_d2)    
    features['dwt_permutation_entropy_EN_d2'] = permutation_entropy(cA4_d2 , dx = 5, normalized = True)
    temp, _ = ApEn(cA4_d2, m = 5, tau = 1)
    features['dwt_Approx_entropy_EN_d2'] = temp[-1]
    features['dwt_Hjorth_mobility_d2'] , features['dwt_Hjorth_complexity_d2'] = Hjorth_parameters(cA4_d2)
    features['dwt_variance_d2'] = np.var(cA4_d2)
    features['population_std_d2'] = np.std(cA4_d2)
    features['sample_std_d2'] = np.std(cA4_d2,  ddof=1)
    features['variance_d2'] = np.std(cA4_d2) ** 2
    features['mean_d2'] = np.mean(cA4_d2)
    features['median_d2'] = np.percentile(cA4_d2, 50)
    features['q1_d2'] = np.percentile(cA4_d2, 25)
    features['q3_d2'] = np.percentile(cA4_d2, 75)
    features['max_d2'] = max(cA4_d2)
    features['min_d2'] = min(cA4_d2)
    features['range_d2'] = max(cA4_d2) - min(cA4_d2)
    features['cov_d2'] = np.std(cA4_d2) / np.mean(cA4_d2)
    features['mean_abs_dev_d2'] = np.mean(np.abs(cA4_d2 - np.mean(cA4_d2)))
    features['skewness_d2'] = skew(cA4_d2)
    features['kurtosis_d2'] = kurtosis(cA4_d2)
    features['first_derivative_std_d2'] = first_derivative_std(cA4_d2)
    features['Hjorth mobility_d2'] , features['Hjorth complexity_d2'] = Hjorth_parameters(cA4_d2)
    features['zero_crossing_rate_d2'] = zero_crossing_rate(cA4_d2)#~~
    features['interquartile_range_d2'] = interquartile_range(cA4_d2)
    features['mean_absolute_power_d2'] = mean_absolute_power(cA4_d2)
    features['rms_d2'] = rms(cA4_d2)
    features['rmssd_d2'] = rmssd(cA4_d2)
    
    
    '''STATISTICAL FEATURES'''
    features['cov'] = np.std(segment) / np.mean(segment)
    
    
    ''' FIDUCIAL FEATURES OR INTER-BEAT INTERVAL FEATURES'''
    bpm = calculate_bpm(segment, sampling_rate)
    IPI = calculate_IPI(segment, sampling_rate)
    rr_intervals_sorted = np.sort(IPI)
    hist, bins = np.histogram(IPI, bins='auto', density=True)
    
    features["bpm"] = bpm
    features["IPI"] = IPI
    features["std_IPI"] = np.std(IPI)
    features["std_diff_IPI"] = np.std(np.diff(IPI))
    features["rms_diff_IPI"] = np.sqrt(np.mean(np.diff(IPI)**2))
    features["mean_pulse_rate"] = np.mean(60 / IPI)
    features["mean_cycle_duration"] = np.mean(IPI)
    features["PNN20"] = np.mean(np.abs(np.diff(IPI)) > 0.02) * 100
    features["PNN50"] = np.mean(np.abs(np.diff(IPI)) > 0.05) * 100
    features["TINN"] = rr_intervals_sorted[-1] - rr_intervals_sorted[0]
    features["triangular_index"] = np.sum(hist) / np.max(hist)
    
    return features
