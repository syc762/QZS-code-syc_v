from psd_params import *
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
from scipy.signal import find_peaks

def load_acceleration_data(file_path):
    data = pd.read_csv(file_path, header=0).squeeze("columns")
    data = pd.to_numeric(data, errors="coerce").dropna().reset_index(drop=True)
    return data

def find_matching_files(freq_of_interest, top_dir, bot_dir):
    # Regex pattern to match frequencies with optional decimals
    pattern = re.compile(rf"{freq_of_interest:.6f}".rstrip("0").rstrip(".") + r"Hz", re.IGNORECASE)
    
    top_file = next((f for f in os.listdir(top_dir) if pattern.search(f)), None)
    bot_file = next((f for f in os.listdir(bot_dir) if pattern.search(f)), None)

    if not top_file or not bot_file:
        raise FileNotFoundError("Matching files not found for frequency:", freq_of_interest)
    
    return os.path.join(top_dir, top_file), os.path.join(bot_dir, bot_file)



def compute_optimal_fs_nperseg(
    data_len,
    freq_of_interest,
    osc_sample_rate=10000,
    fs_raw=10000,
    min_cycles=3,
    max_cycles=10,
    nominal_fs=None
):
    
    """
    Compute optimal sampling frequency and segment length for Welch's method.
    Parameters:
        data_len (int): Length of the data in samples.
        freq_of_interest (float): Frequency of interest in Hz.
        osc_sample_rate (int): Oscilloscope sample rate in Hz (default 10kHz).
        fs_raw (int): Raw sampling frequency in Hz (default 2kHz).
        min_cycles (int): Minimum number of cycles to consider.
        max_cycles (int): Maximum number of cycles to consider.
        nominal_fs (int, optional): User-provided sampling frequency to use instead of candidates.
    """
    
    
    print(f"Data length: {data_len} samples")
    duration = data_len / osc_sample_rate  # Duration in seconds

    # Use user-provided fs or scan candidates
    fs_candidates = [nominal_fs] # if nominal_fs else range(int(2 * freq_of_interest), fs_raw + 1)

    best_match = None
    smallest_bin_error = np.inf
    
    diagnostics = []

    for fs in fs_candidates:

        for n_cycles in range(min_cycles, max_cycles + 1):
            nominal_nperseg = int(round(n_cycles * fs / freq_of_interest))
            step = nominal_nperseg // 2
            if step == 0:
                continue

            # Check that data fits an integer number of 50% overlapped segments
            n_segments = (data_len - nominal_nperseg) / step + 1
            valid_fit = (data_len - nominal_nperseg) % step == 0

            diagnostics.append({
                "n_cycles": n_cycles,
                "nperseg": nominal_nperseg,
                "step": step,
                "n_segments": n_segments,
                "valid_fit": valid_fit
            })

            # Skip invalid configs
            # if not valid_fit:
            #     continue

        

            # Generate PSD bin frequencies to check alignment
            # In addition, check whether there is a peak close to the frequency of interest
            freqs = np.fft.rfftfreq(nominal_nperseg, d=1/fs)
            bin_diffs = np.abs(freqs - freq_of_interest)
            k_peak = np.argmin(bin_diffs)
            bin_error = bin_diffs[k_peak]
            

            if bin_error < smallest_bin_error:
                smallest_bin_error = bin_error
                best_match = {
                    "fs": fs,
                    "nperseg": nominal_nperseg,
                    "duration": duration,
                    "k_peak": k_peak,
                    "bin_error": bin_error,
                    "n_cycles": n_cycles
                }

            
    df_diagnostics = pd.DataFrame(diagnostics)
    print("Diagnostics:")
    print(df_diagnostics)

    if best_match:
        print(f"Selected fs: {best_match['fs']} Hz, nperseg: {best_match['nperseg']}, "
              f"n_cycles: {best_match['n_cycles']}, total duration: {best_match['duration']:.3f}s, "
              f"bin error: {best_match['bin_error']:.6f} Hz")
        return (
            best_match["fs"],
            best_match["nperseg"],
            best_match["k_peak"],
            best_match["bin_error"],
            best_match["n_cycles"],
            best_match["duration"]
        )
    else:
        raise ValueError("No valid fs and nperseg found that satisfies all constraints.")


def save_loglog_psd_plot(data, fs, nperseg, label, save_dir, window='hamming', overlap_frac=0.5, osc_sample_rate=10000):
    """
    Compute PSD using Welch method and save a log-log plot.
    """
    from scipy.signal import welch
    noverlap = int(nperseg * overlap_frac)
    f, Pxx = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)

    # Compute acquisition time from data length at raw sampling rate (10kHz)
    acq_time = len(data) / osc_sample_rate


    # File-safe name
    fname = f"{label}_fs{fs}_nperseg{nperseg}_t{acq_time:.2f}s_{window}_overlap{int(overlap_frac*100)}.png"
    filepath = os.path.join(save_dir, fname)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(f, Pxx)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title(f"{label} PSD\nfs={fs}Hz, nperseg={nperseg}, t={acq_time:.2f}s, {window}, {int(overlap_frac*100)}% overlap")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


# Define a reusable PSD plotting function with peak annotations
def plot_psd_with_peaks(f, Pxx, freq_of_interest, fs, nperseg, duration, n_cycles_used, bin_error):
    # Find peak nearest to driving frequency
    peaks, _ = find_peaks(Pxx)
    peak_freqs = f[peaks]
    peak_vals = Pxx[peaks]
    nearest_idx = np.argmin(np.abs(peak_freqs - freq_of_interest))
    nearest_freq = peak_freqs[nearest_idx]
    nearest_psd = peak_vals[nearest_idx]

    # Global max PSD peak
    global_peak_idx = np.argmax(Pxx)
    global_peak_freq = f[global_peak_idx]
    global_peak_val = Pxx[global_peak_idx]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(f, Pxx, label="PSD")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title(f"n_cycles={n_cycles_used}, fs={fs}Hz, nperseg={nperseg}, t={duration:.2f}s\nbin error={bin_error:.4f}Hz")

    # Mark driving frequency peak (if found)
    plt.axvline(x=nearest_freq, color='red', linestyle='--', label=f"Driving freq: {nearest_freq:.2f} Hz")
    plt.text(nearest_freq, nearest_psd, f"({nearest_freq:.2f}, {nearest_psd:.2e})",
             color='red', fontsize=9, ha='left', va='top', rotation=0)

    # Mark global max PSD
    plt.axvline(x=global_peak_freq, color='navy', linestyle='--', label=f"Max PSD: {global_peak_freq:.2f} Hz")
    plt.text(global_peak_freq, global_peak_val, f"({global_peak_freq:.2f}, {global_peak_val:.2e})",
             color='navy', fontsize=9, ha='left', va='top', rotation=0)

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"top_psd_plot_fs{fs}_nperseg{nperseg}_cycles{n_cycles_used}.png")
    plt.close()




if __name__ == "__main__":
    
    # Define the frequency of interest
    freq_of_interest = 35.0  # Hz
    
    # Top and bottom data directories
    top_dir = "top_unzip/202506050517_top_bestYet2Hz_flexureOnly_0.381kg_35to1000Hz_10sSleep_x1_tf_100points_1.5V"
    bot_dir = "bot_unzip/202506050541_bot_bestYet2Hz_flexureOnly_0.381kg_35to1000Hz_10sSleep_x1_tf_100points_1.5V"

    top_path, bot_path = find_matching_files(freq_of_interest, top_dir, bot_dir)
    top_data = load_acceleration_data(top_path)
    bot_data = load_acceleration_data(bot_path)

    # List of sampling frequencies to evaluate
    fs_list = [10000]
    freq_of_interest = 35.0

    for fs in fs_list:
        # Use fixed fs and iterate through n_cycles to find a valid config
        try:
            fs_used, nperseg, k_peak, bin_error, n_cycles_used, duration = compute_optimal_fs_nperseg(
                data_len=len(top_data),
                freq_of_interest=freq_of_interest,
                osc_sample_rate=10000,
                fs_raw=fs,
                min_cycles=2,
                max_cycles=10,
                nominal_fs=fs
            )
            
            # Compute the top PSD using Welch
            f, psd_top = welch(top_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window='hamming')
            plot_psd_with_peaks(f, psd_top, freq_of_interest=freq_of_interest, fs=fs_used, nperseg=nperseg,
                                duration=duration, n_cycles_used=n_cycles_used, bin_error=bin_error)
            print(f"Using fs={fs_used} Hz, nperseg={nperseg}, n_cycles={n_cycles_used}, bin error={bin_error:.4f} Hz")
            
            # Compute bottom PSD using Welch and the transmissibility
            f, psd_bot = welch(bot_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window='hamming')
            transmissibility =  psd_bot/psd_top
            
            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(f, psd_top, label="Top PSD")
            plt.plot(f, psd_bot, label="Bottom PSD")
            plt.plot(f, transmissibility, label="Transmissibility", color='green')
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (m²/s⁴/Hz) or Transmissibility")
            plt.title(f"driving freq={freq_of_interest:.3f}Hz, n_cycles={n_cycles_used}, fs={fs}Hz, nperseg={nperseg},\nbin error={bin_error:.4f}Hz")

            # Mark driving frequency peak (if found)
            peaks, _ = find_peaks(psd_top)
            peak_freqs = f[peaks]
            peak_vals = psd_top[peaks]
            nearest_idx = np.argmin(np.abs(peak_freqs - freq_of_interest))
            nearest_freq = peak_freqs[nearest_idx]
            nearest_psd = peak_vals[nearest_idx]
            
            # Mark the driving frequency peak
            plt.axvline(x=freq_of_interest, color='tab:blue', linestyle='--', label=f"Driving freq: {freq_of_interest:.2f} Hz")
            plt.text(freq_of_interest, nearest_psd, f"({freq_of_interest:.2f})",
                    color='tab:blue', fontsize=9, ha='left', va='top', rotation=0)

            # Mark the major peaks in the transmissibility
            peaks, _ = find_peaks(transmissibility)
            peak_freqs = f[peaks]
            peak_vals = transmissibility[peaks]
            if len(peak_freqs) > 0:
                global_peak_idx = np.argmax(transmissibility)
                global_peak_freq = f[global_peak_idx]
                global_peak_val = transmissibility[global_peak_idx]
                
                plt.axvline(x=global_peak_freq, color='tab:blue', linestyle='--', label=f"Major peak at {global_peak_freq:.2f} Hz")
                plt.text(global_peak_freq, global_peak_val, f"({global_peak_freq:.2f}, {global_peak_val:.2e})",
                        color='navy', fontsize=9, ha='left', va='top', rotation=0)
            
            plt.grid(True, which="both")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"AAAdrivFreq{freq_of_interest}_fs{fs}_nperseg{nperseg}_cycles{n_cycles_used}.png")

            #break  # Use the first valid n_cycles found
        except Exception as e:
            print(f"No valid Welch config found for fs={fs}Hz: {e}")
            continue
            
            
            


        

        

                
                
            