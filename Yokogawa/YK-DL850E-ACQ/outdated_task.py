import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss


basedir = r"Z:\Users\Soyeon\JulyQZS\250709_noDampers_0.381kg\nowpeakfinderop_202507101324_bestYet2Hz_ch1top_ch2bot_tf_3.0VSQUare"
filename = "tf_data_3.0Hz_10kS_n2002000_noDampers_0.381kg_20250712_133040.csv"


tf_data = pd.read_csv(os.path.join(basedir, filename), low_memory=False)

# Set your threshold value
threshold = 2e-9

# Apply filter: keep rows where both PSD_top and PSD_bot are above the threshold
filtered_df = tf_data[(tf_data['channel1'] > threshold) & (tf_data['channel2'] > threshold)]

# Optional: save the filtered results
filtered_df.to_csv("filtered_psd_data.csv", index=False)

# Optional: preview the result
print(filtered_df.head())


# --- Peak Detection ---
peaks_ch1_idx, _ = ss.find_peaks(psd_ch1, **peak_params)
peaks_ch2_idx, _ = ss.find_peaks(psd_ch2, **peak_params)

# --- Odd Overtone Filter ---
def filter_to_odd_harmonics(freqs, peak_indices, base_freq, max_freq=100, tol=0.1):
    """Keep only peaks near odd harmonics of base_freq (e.g. 3, 9, 15...)."""
    odd_harmonics = np.arange(1, int(max_freq // base_freq) + 1, 2) * base_freq
    filtered = []
    for i in peak_indices:
        if any(np.abs(freqs[i] - overtone) <= tol for overtone in odd_harmonics):
            filtered.append(i)
    return np.array(filtered)

# Driving frequency (e.g. from filename or manually set)
driving_freq = 3.0  # adjust or extract dynamically
harmonic_tol = 0.1  # Hz

# Filter detected peaks to only those near odd overtones
peaks_ch1_idx = filter_to_odd_harmonics(freq_ch1, peaks_ch1_idx, driving_freq, tol=harmonic_tol)
peaks_ch2_idx = filter_to_odd_harmonics(freq_ch2, peaks_ch2_idx, driving_freq, tol=harmonic_tol)

# --- Pad columns for CSV if peak counts differ ---
max_len = max(len(peaks_ch1_idx), len(peaks_ch2_idx))
ch1_freqs = np.pad(freq_ch1[peaks_ch1_idx], (0, max_len - len(peaks_ch1_idx)), constant_values=np.nan)
ch1_psd = np.pad(psd_ch1[peaks_ch1_idx], (0, max_len - len(peaks_ch1_idx)), constant_values=np.nan)
ch2_freqs = np.pad(freq_ch2[peaks_ch2_idx], (0, max_len - len(peaks_ch2_idx)), constant_values=np.nan)
ch2_psd = np.pad(psd_ch2[peaks_ch2_idx], (0, max_len - len(peaks_ch2_idx)), constant_values=np.nan)

# --- Construct peak table ---
peak_data = {
    'CH1_Freq_Hz': ch1_freqs,
    'CH1_PSD': ch1_psd,
    'CH2_Freq_Hz': ch2_freqs,
    'CH2_PSD': ch2_psd,
    'Peak_Params_Height': [peak_params.get('height')] * max_len,
    'Peak_Params_Distance': [peak_params.get('distance')] * max_len,
    'Peak_Params_Prominence': [peak_params.get('prominence')] * max_len,
    'Peak_Params_Width': [peak_params.get('width')] * max_len
}