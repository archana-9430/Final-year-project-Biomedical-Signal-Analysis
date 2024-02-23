import pandas as pd
import numpy as np
import EntropyHub as EH

from scipy.stats import entropy, skew, kurtosis
import scipy.signal as signal

def shannon_entropy(segment):
    # Calculate the probability distribution
    p = np.histogram(segment)[0]
    p_normalized = p / float(np.sum(p))
    p_normalized = p_normalized[np.nonzero(p_normalized)]
    
    # Compute the Shannon entropy
    H = entropy(p_normalized, base=2)
    return H

def mean_psd(segment, fs):
    f, psd = signal.welch(segment, fs=fs)
    '''Welch's method: Divide the signal into overlapping segments,
    computing the Fourier transform of each segment,
    and averaging the squared magnitudes of the resulting spectra to obtain the PSD estimate.
    '''
    # mean_psd = np.mean(psd)
    std_psd = np.std(psd)
    #accessing the maximum frequency from the index of the maximum power value in the PSD array
    dominant_frequency = f[np.argmax(psd)]
     # Calculate spectral entropy
    normalized_psd = psd / np.sum(psd)  # Normalize PSD
    spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd))
    return std_psd, dominant_frequency, spectral_entropy

def sample_entropy(segment):
    Samp, _ = EH.SampEn(segment, m = 4)
    return Samp
    
def rms(segment):
    rms = np.sqrt(np.mean(segment**2))
    return rms

def interquartile_range(segment):
    q3, q1 = np.percentile(segment, [75 ,25])
    return (q3 - q1)
    

def statistical(segment):
    features = {}
    #Time domain:
    features['mean'] = np.mean(segment)
    features['population_std'] = np.std(segment)
    features['sample_std'] = np.std(segment,  ddof=1)
    features['rms'] = rms(segment)
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['interquartile_range'] = interquartile_range(segment)
    
    features['Shannon entropy'] = shannon_entropy(segment)
    # features['sample_entropy'] = sample_entropy(segment)

    #Frequency domain
    features['std_psd'], features['dominant_freq'], features['spectral_entropy'] = mean_psd(segment, 125)
    
    return features