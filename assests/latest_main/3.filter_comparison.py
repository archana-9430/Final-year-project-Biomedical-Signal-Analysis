'''
Signal-to-Noise Ratio (SNR)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Peak Signal-to-Noise Ratio (PSNR)
'''

clean_signal = "SNR_data/clean_merged.csv"
# filtered_signal = "SNR_data/butter.csv"
# filtered_signal = "SNR_data/cheby1.csv"
# filtered_signal = "SNR_data/cheby2.csv"
filtered_signal = "SNR_data/ellip.csv"
noise = "SNR_data/noise_seg_merged.csv"

import numpy as np
import pandas as pd

def snr_improvement(clean_segment, filtered_segment, noise):
    # print("clean_segment.shape", clean_segment.shape, " filtered_segment.shape ", filtered_segment.shape)
    assert clean_segment.shape == filtered_segment.shape, "Clean and filtered segments must have the same shape"
    snr_improvement = (np.mean(clean_segment**2) / ((np.mean(filtered_segment**2)) - (np.mean(clean_segment**2)))) - (np.mean(clean_segment**2) / np.mean(noise**2))
    return snr_improvement

def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_mse(signal, filtered_signal):
    N = len(signal)
    mse = np.mean((signal - filtered_signal)**2) / N
    return mse

def calculate_rmse(signal, filtered_signal):
    mse = calculate_mse(signal, filtered_signal)
    rmse = np.sqrt(mse)
    return rmse

def calculate_psnr(signal, filtered_signal):
    max_possible_signal_value = np.max(signal)
    rmse = calculate_rmse(signal, filtered_signal)
    psnr = 20 * np.log10(max_possible_signal_value / rmse)
    return psnr

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def calculate_ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets
    To check the correct implementation, the NCC of a sample with itself needs to return 1.0

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def calculate_sf(original_signal, filtered_signal):
    sd_filtered = np.std(filtered_signal)
    sd_original = np.std(original_signal)
    sf = sd_filtered / sd_original
    return sf


# Read the CSV files for signal and filtered signal
signal_data = pd.read_csv(clean_signal, skiprows=1)
filtered_signal_data = pd.read_csv(filtered_signal)
noise_data = pd.read_csv(noise)

# print("signal_data", signal_data)
# print("filtered_signal_data", filtered_signal_data)
# print("noise_data", noise_data)

# Assuming data is organized in columns and both CSV files have the same number of columns
num_columns = len(signal_data.columns)
snr_values = []
ncc_values = []
rmse_values = []
mse_values = []
psnr_values = []
sf_values = []

# noise_len = len(noise_data.columns)

for i in range(num_columns):
    # Extract data from corresponding columns in both files
    signal_segment = signal_data.iloc[:, i]
    filtered_signal_segment = filtered_signal_data.iloc[:, i]
    # noise_index = i % noise_len
    # noise_segment = noise_data.iloc[:, noise_index]
    noise_segment = noise_data.iloc[:, i]
    
    # print("i: ", i)
    # print("signal_segment", signal_segment)
    
    # Calculate SNR
    snr = snr_improvement(signal_segment, filtered_signal_segment, noise_segment)
    # print("snr", snr)
    snr_values.append(snr)
    
    # Calculate MSE
    mse = calculate_mse(signal_segment, filtered_signal_segment)
    mse_values.append(mse)
    
    # Calculate RMSE
    rmse = calculate_rmse(signal_segment, filtered_signal_segment)
    rmse_values.append(rmse)
    
    # Calculate PSNR
    psnr = calculate_psnr(signal_segment, filtered_signal_segment)
    psnr_values.append(psnr)
    
    # Calculate NCC
    ncc = calculate_ncc(signal_segment, filtered_signal_segment)
    ncc_values.append(ncc)
    
    # Calculate SF: Smoothness factor
    sf = calculate_sf(signal_segment, filtered_signal_segment)
    sf_values.append(sf)

mean_snr = np.mean(snr_values)
mean_ncc = np.mean(ncc_values)
mean_rmse = np.mean(rmse_values)
mean_mse = np.mean(mse_values)
mean_psnr = np.mean(psnr_values)
mean_sf = np.mean(sf_values)

print("Mean SNR:", mean_snr)
print("Mean NCC:", mean_ncc)
print("Mean RMSE:", mean_rmse)
print("Mean MSE:", mean_mse)
print("Mean PSNR:", mean_psnr)
print("Mean SF:", mean_sf)

# snr = calculate_snr(original_signal, noise)
# mse = calculate_mse(original_signal, filtered_signal)
# rmse = calculate_rmse(original_signal, filtered_signal)
# psnr = calculate_psnr(original_signal, filtered_signal)
# pcc = ncc(original_signal, filtered_signal)
# sf = calculate_sf(original_signal, filtered_signal)

