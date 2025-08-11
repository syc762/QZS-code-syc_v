from scipy.signal import welch, find_peaks
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

# Function to compute PSD at a given frequency using adaptive nperseg
def compute_psd(file_path, sampling_rate, nperseg):
    data = pd.read_csv(file_path).iloc[:, 0].values  # Use the first (and only) column
    f, Pxx = welch(data, fs=sampling_rate, window='hann', nperseg=nperseg, noverlap=nperseg // 2, detrend='constant', scaling='density')
    peak_index = np.argmax(Pxx)
    return f[peak_index], Pxx[peak_index]

# Helper: Get (max PSD, corresponding freq) near target_freq within ±tol
def get_psd_at_freq(freq_array, psd_array, target_freq, tol=0.2):
    # Find all indices within the tolerance range
    mask = (freq_array >= target_freq - tol) & (freq_array <= target_freq + tol)
    
    if not np.any(mask):
        return np.nan, np.nan  # No valid bins within range

    # Extract values in the range
    freqs_in_range = freq_array[mask]
    psd_in_range = psd_array[mask]

    # Find max PSD and its corresponding frequency
    max_idx = np.argmax(psd_in_range)
    max_psd = psd_in_range[max_idx]
    max_freq = freqs_in_range[max_idx]

    return max_psd, max_freq


def optimal_nperseg(data, fs, freq_of_interest, waveform_length,
                    min_cycles=5, max_cycles=18, window='hamming', noverlap=None, verbose=False):
    """
    Return the nperseg that minimizes:
    sqrt( (bin_freq - freq_of_interest)^2 + (peak_freq - freq_of_interest)^2 )
    
    Optimization criteria:
    1. top_data's maximum psd occurs at freq_of_interest
    2. bin spacings are aligned with freq_of_interest (i.e. freq_of_interest falls into a bin)
    """
    best_nperseg = None
    best_k = None
    best_peak_freq = None
    best_combined_error = np.inf
    best_bin_error = None
    best_psd_error = None

    for n_cycles in range(min_cycles, max_cycles + 1):
        nperseg_cand = int(round(n_cycles * fs / freq_of_interest))
        nperseg_cand = min(max(nperseg_cand, 32), waveform_length)  # Clamp safely

        if noverlap is None:
            safe_noverlap = min(nperseg_cand - 1, nperseg_cand // 2)
        else:
            safe_noverlap = min(nperseg_cand - 1, noverlap)

        k = max(1, int(round(freq_of_interest * nperseg_cand / fs)))
        bin_freq = k * (fs / nperseg_cand)
        bin_error = abs(freq_of_interest - bin_freq)

        freqs, psd = welch(data, fs=fs, window=window,
                           nperseg=nperseg_cand, noverlap=safe_noverlap)

        peaks, _ = find_peaks(psd)
        if len(peaks) == 0:
            continue

        peak_freqs = freqs[peaks]
        nearest_idx = np.argmin(np.abs(peak_freqs - freq_of_interest))
        nearest_freq = peak_freqs[nearest_idx]
        psd_error = abs(freq_of_interest - nearest_freq)

        combined_error = np.sqrt(bin_error**2 + psd_error**2)

        if verbose:
            print(f"nperseg: {nperseg_cand}, k: {k}, bin_freq: {bin_freq:.2f}, "
                  f"peak_freq: {nearest_freq:.2f}, bin_error: {bin_error:.2f}, "
                  f"psd_error: {psd_error:.2f}, combined_error: {combined_error:.2f}")

        if combined_error < best_combined_error:
            best_combined_error = combined_error
            best_nperseg = nperseg_cand
            best_k = k
            best_peak_freq = nearest_freq
            best_bin_error = bin_error
            best_psd_error = psd_error

    if best_nperseg is None:
        raise RuntimeError("No valid nperseg found within the specified cycle range.")

    return best_nperseg, best_k, best_peak_freq, best_bin_error, best_psd_error



def optimal_nperseg_for_square(fs, freq, len_acc, max_k=1):
    """
    Choose nperseg such that FFT bins fall exactly on odd harmonics of `freq`.
    """
    # Trial1.
    # nperseg = int(max_k * fs / freq)

    # Trial2. Round to next power of 2 for efficient FFT
    #nperseg = 2 ** int(np.round(np.log2(nperseg)))
    #nperseg = min(nperseg, len_acc)

    # Trial3. Simpler approach
    nperseg = fs

    return nperseg


# --- Select only Odd Overtones in the found peaks ---
def filter_to_odd_harmonics(freqs, peak_indices, base_freq, max_freq=100, tol=0.1):
    """Keep only peaks near odd harmonics of base_freq (e.g. 3, 9, 15...)."""
    odd_harmonics = np.arange(1, int(max_freq // base_freq) + 1, 2) * base_freq
    filtered = []
    for i in peak_indices:
        if any(np.abs(freqs[i] - overtone) <= tol for overtone in odd_harmonics):
            filtered.append(i)
    return np.array(filtered)

# Improved version of finding peaks near odd harmonics
def find_peaks_at_odd_harmonics_summary(freqs, psd, base_freq, fs, tol=0.2, peak_params=None):
    if peak_params is None:
        peak_params = {}

    odd_harmonics = np.arange(1, int((fs / 2) // base_freq) + 1, 2) * base_freq
    
    summary_rows = []

    for overtone_num, target_freq in zip(np.arange(1, len(odd_harmonics)*2, 2), odd_harmonics):
        mask = np.where((freqs >= target_freq - tol) & (freqs <= target_freq + tol))[0]
        if len(mask) == 0:
            summary_rows.append({
                "Harmonic": f"{overtone_num}x",
                "Target_Freq_Hz": target_freq,
                "Peak_Freq_Hz": np.nan,
                "Peak_Height": np.nan,
                "Found": False
            })
            continue

        local_psd = psd[mask]
        local_peaks, _ = find_peaks(local_psd, **peak_params)

        if len(local_peaks) == 0:
            summary_rows.append({
                "Harmonic": f"{overtone_num}x",
                "Target_Freq_Hz": target_freq,
                "Peak_Freq_Hz": np.nan,
                "Peak_Height": np.nan,
                "Found": False
            })
        else:
            # Take the highest peak in this window
            best_idx = local_peaks[np.argmax(local_psd[local_peaks])]
            summary_rows.append({
                "Harmonic": f"{overtone_num}x",
                "Target_Freq_Hz": target_freq,
                "Peak_Freq_Hz": freqs[mask[best_idx]],
                "Peak_Height": local_psd[best_idx],
                "Found": True
            })

    return pd.DataFrame(summary_rows)


# # --- Restrict peak detection to odd harmonics, then find peaks ---
# def find_peaks_at_odd_harmonics(freqs, psd, base_freq, fs, tol=0.2, peak_params=None):
#     """Restrict search to odd harmonics of base_freq, apply find_peaks locally."""
#     if peak_params is None:
#         peak_params = {}

#     odd_harmonics = np.arange(1, int((fs / 2) // base_freq) + 1, 2) * base_freq
#     matched_freqs = []
#     matched_heights = []

#     for target_freq in odd_harmonics:
#         # Identify region within ±tol Hz around each harmonic
#         mask = np.where((freqs >= target_freq - tol) & (freqs <= target_freq + tol))[0]
#         if len(mask) == 0:
#             continue

#         local_psd = psd[mask]
#         local_peaks, props = find_peaks(local_psd, **peak_params)
#         for i in local_peaks:
#             matched_freqs.append(freqs[mask[i]])
#             matched_heights.append(local_psd[i])
#             matched_labels.append(f"{int(target_freq/base_freq)}x")

#     return np.array(matched_freqs), np.array(matched_heights)



# def run_welch_loop(example_waveform, fs_default=10000):
#     while True:
#         try:
#             fs = float(input(f"Enter sampling frequency [Hz] (default={fs_default}): ") or fs_default)
#             nperseg = int(input("Enter nperseg value (e.g., 1024): "))

#             freqs, psd = welch(example_waveform, fs=fs, nperseg=nperseg)
            
#             plt.figure(figsize=(5, 3))
#             plt.semilogy(freqs, psd, label=f"nperseg={nperseg}, fs={fs}")
#             plt.xlabel("Frequency [Hz]")
#             plt.ylabel("PSD [V²/Hz]")
#             plt.title("Welch PSD Estimate")
#             plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#             plt.legend()
#             plt.tight_layout()
#             plt.show()

#             confirm = input("Are you satisfied with these parameters? (y/n): ").lower()
#             if confirm == 'y':
#                 return fs, nperseg
#         except Exception as e:
#             print(f"Error: {e}. Please try again.")

def plot_acceleration_segment(acc_data, freq, fs, channel_label, start_time=0.0, end_time=2.0, save_dir='.'):
    """
    Plots a segment of the acceleration data for a specified time range.
    
    Parameters:
        acc_data (np.array): Acceleration data
        fs (float): Sampling frequency in Hz
        channel_label (str): Channel label (e.g., "CH1")
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        save_dir (str): Directory to save the plot
    """
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    t_segment = np.arange(start_idx, end_idx) / fs
    acc_segment = acc_data[start_idx:end_idx]   

    plt.figure(figsize=(10, 4))
    plt.plot(t_segment, acc_segment, label=f'{channel_label} Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'{channel_label} Acceleration ({start_time}–{end_time} s)')
    plt.grid(True)
    plt.ylim(acc_segment.min()*1.2, acc_segment.max()*1.2)  # Explicitly set y-axis to min/max
    plt.tight_layout()
    filename = f'{channel_label.lower()}_{freq}Hz_acceleration_{start_time:.1f}_{end_time:.1f}s.png'
    plt.savefig(os.path.join(save_dir, filename))
    

    # Example usage for CH1: 0 to 2 seconds
    # plot_acceleration_segment(acc_data=acc_ch1, fs=fs, channel_label="CH1", start_time=0.0, end_time=2.0, save_dir=save_dir)
def plot_psd(c, freq, psd_data, frequency_of_interest, save_dir, log=False):
    """
    Plot the Power Spectral Density (PSD) for a given channel.
    Highlights the peak value and saves the figure.
    """
    rounded_freq = round(frequency_of_interest, 2)

    # Ensure freq and psd_data are arrays and not empty
    if len(freq) == 0 or len(psd_data) == 0:
        print(f"[Warning] Empty PSD data for channel {c} — skipping plot.")
        return

    # peak_idx = np.argmax(psd_data)
    # peak_freq = freq[peak_idx]
    # peak_val = psd_data[peak_idx]

    plt.figure(figsize=(7,5))
    if log:
        plt.loglog(freq, psd_data, label=f"CH{c} PSD at {rounded_freq} Hz")
    else:
        plt.plot(freq, psd_data)
    
    plt.title(f"CH{c} PSD at driv freq {rounded_freq} Hz")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [m²/Hz]")
    plt.legend(fontsize=8)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.tight_layout()

    fname = f"CH{c}_psd_{rounded_freq}Hz.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close()






if __name__ == "__main__":

    
    # Sampling rate and number of cycles per segment
    sampling_rate = 10000  # Hz
    cycles_per_segment = 30
     # Max record length

    # Initialize lists for TF and validated frequency axis
    tf_values = []
    validated_freqs = []

    #top_dir = [f for f in os.listdir() if f.startswith("top")][0]  # Get the top directory  
    #bot_dir = [f for f in os.listdir() if f.startswith("bot")][0]  # Get the bottom directory
    bot_dir = r"Z:\Users\Soyeon\SoyeonChoi_fromQZSDesktop\QZS_2505201413_backup\202505201353_bot_bestYet2Hz_can_vol67_tf_100points_1.0V"
    save_path = os.path.join(bot_dir, f"bot_psd.csv")

    # Load both files
    # List matching CSV files
    # top_files = sorted(f for f in os.listdir(top_dir) if f.startswith("acceleration_data_"))
    bot_files = sorted(f for f in os.listdir(bot_dir) if f.startswith("acceleration_data_"))
    

    array_psd_bot = []

    # Get the waveform length from the first file
    waveform_length = len(bot_files[0])


    # Get the frequencies from the filenames
    frequencies = []
    for file in bot_files:
        match = re.search(r'(\d+\.\d+)Hz', file)
        if match:
            frequencies.append(float(match.group(1)))
    print(f"Frequencies found: {frequencies}")

    

    # Process all frequencies present in the transfer data
    for freq in frequencies:
        # Construct matching filenames
        freq_str = f"{freq}Hz"
        bot_file = os.path.join(bot_dir, f"acceleration_data_{freq_str}.csv")
        # top_file = os.path.join(top_dir, f"acceleration_data_{freq_str}.csv")

        # fs_top, nperseg_top = run_welch_loop(top_file, fs_default=sampling_rate)
        # fs_bot, nperseg_bot = run_welch_loop(bot_file, fs_default=sampling_rate)
        
        # Compute nperseg for each frequency
        # nseg = np.minimum((cycles_per_segment * sampling_rate / freq).astype(int), waveform_length)
        nseg = int(min(cycles_per_segment * sampling_rate / freq, waveform_length))


        # Compute peak PSD for both top and bottom
        try:
            f_bot_peak, psd_bot = compute_psd(bot_file, sampling_rate, nseg)
            #f_top_peak, psd_top = compute_psd(top_file, sampling_rate, nseg)
            
            # Show the computed PSD values and the normalized transmissibility
            #print(f"Frequency: {freq_str}, Top PSD: {psd_top:.2e}, Bottom PSD: {psd_bot:.2e}")
            print(f"Frequency: {freq_str}, Bottom PSD: {psd_bot:.2e}")

            # Ask the user if they want to keep the values, and if they don't want them, recompute with a different nseg value
            freq_str = f"{freq}Hz"

            #tf = psd_bot / psd_top
            #tf_values.append(tf)
            #validated_freqs.append(freq)
            array_psd_bot.append(psd_bot)

        except Exception as e:
            print(f"Skipping {freq_str}: {e}")
        
    # Save to CSV
    df = pd.DataFrame({
        "frequency": frequencies,
        "bot": array_psd_bot
    })
    
    df.to_csv(save_path, index=False)
    print(f"Saved PSD CSV for {freq}Hz to {save_path}")

    # Plot
    plt.loglog(frequencies, array_psd_bot, label=f"Bottom PSD")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("Bottom Signal PSDs")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(bot_dir, "bot_psd.png"), dpi=300)
    plt.show()