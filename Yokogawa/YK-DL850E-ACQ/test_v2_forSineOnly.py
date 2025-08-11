import os
import re
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from task_v2 import extract_metadata_from_path, extract_freq
import psd as psd


# If you have noise spectra (background), subtract them from the PSDs?

import matplotlib.pyplot as plt

def plot_psd_for_driving_frequency(it, 
    driving_frequency,
    fs_adj, nperseg, nfft, n_cycles,
    freqs_top, psd_top, peak_freq_top, peak_val_top,
    psd_bot, peak_freq_bot, peak_val_bot,
    trans, peak_freq_trans, peak_val_trans
):
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Top PSD
    axs[0].loglog(freqs_top, psd_top, color='cornflowerblue', linewidth=1.2)
    axs[0].axvline(peak_freq_top, color='black', linestyle='--', linewidth=1)
    axs[0].axhline(peak_val_top, color='black', linestyle='--', linewidth=1)
    axs[0].plot(peak_freq_top, peak_val_top, 'ko')
    axs[0].text(peak_freq_top, peak_val_top,
                f'({peak_freq_top:.2f}, {peak_val_top:.2e})',
                fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('PSD (Acceleration²/Hz)')
    axs[0].set_title('Top PSD')
    axs[0].grid(True, which='both', ls='--', lw=0.5)

    # Bottom PSD
    axs[1].loglog(freqs_top, psd_bot, color='salmon', linewidth=1.2)
    axs[1].axvline(peak_freq_bot, color='black', linestyle='--', linewidth=1)
    axs[1].axhline(peak_val_bot, color='black', linestyle='--', linewidth=1)
    axs[1].plot(peak_freq_bot, peak_val_bot, 'ko')
    axs[1].text(peak_freq_bot, peak_val_bot,
                f'({peak_freq_bot:.2f}, {peak_val_bot:.2e})',
                fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('PSD (Acceleration²/Hz)')
    axs[1].set_title('Bottom PSD')
    axs[1].grid(True, which='both', ls='--', lw=0.5)

    # Transmissibility
    axs[2].semilogx(freqs_top, trans, color='lightseagreen', linewidth=1.2)
    axs[2].axvline(peak_freq_trans, color='black', linestyle='--', linewidth=1)
    axs[2].axhline(peak_val_trans, color='black', linestyle='--', linewidth=1)
    axs[2].plot(peak_freq_trans, peak_val_trans, 'ko')
    axs[2].text(peak_freq_trans, peak_val_trans,
                f'({peak_freq_trans:.2f}, {peak_val_trans:.2f})',
                fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Transmissibility')
    axs[2].set_title('Transfer Function')
    axs[2].grid(True, which='both', ls='--', lw=0.5)
    axs[2].set_xlim([freqs_top[1], fs_adj / 2])

    # Main title
    plt.suptitle(
        f'Driving Frequency = {driving_frequency:.2f} Hz | fs_adj = {fs_adj:.2f} Hz | nperseg = {nperseg} | nfft = {nfft} | cycles = {n_cycles}',
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"raw_data/{it}psd_{driving_frequency:.2f}Hz.png", dpi=300)
    plt.close()

def plot_psd(df_results, base_name):
    """
    Plot the Power Spectral Density (PSD) with optional logarithmic scaling.
    """
    # Ensure numeric array for peak detection
    trans = df_results['transmissibility'].to_numpy()
    peak_indices, _ = find_peaks(trans)
    
    # The x-axis values for the transmissibility and PSDs
    driv_freq = df_results['driving_frequency']
    
    
    plt.figure(figsize=(10, 6))
    
    
    plt.figure(figsize=(10, 6))
    plt.loglog(driv_freq, trans,
                linestyle='-', marker='.', color='lightseagreen', label='Transmissibility')
    plt.loglog(driv_freq, df_results['peak_val_bot'],
                linestyle='-', marker='.', color='cornflowerblue', label='Bottom PSD')
    plt.loglog(driv_freq, df_results['peak_val_top'],
             linestyle='-', marker='.', color='salmon', label='Top PSD')
    
    # Mark transmissibility peaks
    
    for idx in peak_indices:
        x_val = driv_freq.iloc[idx]
        y_val = df_results['transmissibility'].iloc[idx]
        plt.plot(x_val, y_val, marker='*', color='black', markersize=10)
        plt.text(x_val, y_val, f'({x_val:.1f}, {y_val:.2f})',
                 fontsize=8, ha='left', va='bottom')
    
    
    plt.xlabel('Driving Frequency [Hz]')
    plt.ylabel('Transmissibility or PSD')
    plt.title('Transmissibility and PSD vs Driving Frequency')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(f"tf_psd_plots/tf_{base_name}.png", transparent=True, dpi=300)

""" 
def optimal_fs(data, freq_of_interest, waveform_length=10000, initial_fs=500, n_cycles=10, tol=1.0, max_iter=50):
    
    #Iteratively adjust fs so that the PSD peak frequency aligns with freq_of_interest.

    Parameters:
    - data: input signal (1D array)
    - freq_of_interest: target frequency
    - waveform_length: maximum usable fs
    - initial_fs: starting guess for sampling rate
    - n_cycles: cycles to fit in nperseg
    - tol: peak freq tolerance in Hz
    - max_iter: maximum steps to avoid infinite loops

    Returns:
    - fs: adjusted sampling frequency
    - peak_freq: final PSD peak frequency
    - nperseg: segment length used
    - error: abs(peak_freq - freq_of_interest)
    
    fs = initial_fs
    iterations = 0

    while 2 * freq_of_interest <= fs <= waveform_length and iterations < max_iter:
        nperseg = int(round(n_cycles * fs / freq_of_interest))
        if nperseg < 32 or nperseg > len(data):
            break

        freqs, psd = welch(data, fs=fs, window='hamming', nperseg=nperseg, noverlap=nperseg//2)
        peak_freq = freqs[np.argmax(psd)]
        error = abs(peak_freq - freq_of_interest)

        if error < tol:
            return fs, peak_freq, nperseg, error

        # Adjust fs based on error direction
        fs += 20 if peak_freq < freq_of_interest else -20
        iterations += 1

    raise RuntimeError(f"Failed to converge within {tol} Hz after {iterations} iterations.")

 """


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


def compute_psd_simple(output_dir, top_filepath, bot_filepath, fs, nperseg, noverlap=None, window='hamming', nfft=None):
    """
    Compute the Power Spectral Density (PSD) using Welch's method.
    
    Parameters:
    - data: 1D array of input signal
    - fs: sampling frequency
    - nperseg: length of each segment for FFT
    - noverlap: number of points to overlap between segments
    - window: type of window to apply (default is 'hamming')
    - nfft: length of the FFT (optional, defaults to nperseg if None)
    
    Returns:
    - freqs: frequencies at which the PSD is computed
    - psd: computed PSD values
    """
    # Load and flatten acceleration data
    df_top = pd.read_csv(top_filepath)
    df_bot = pd.read_csv(bot_filepath)
    data_top = df_top.iloc[:, 0].dropna().to_numpy().flatten()
    data_bot = df_bot.iloc[:, 0].dropna().to_numpy().flatten()

    if data_top.shape != data_bot.shape:
        raise ValueError(f"Shape mismatch: {data_top.shape} vs {data_bot.shape}")

    waveform_length = min(len(data_top), len(data_bot))

    # Extract driving frequency
    match = re.search(r"([\d\.]+)Hz", os.path.basename(top_filepath))
    if not match:
        raise ValueError(f"Filename {top_filepath} does not contain 'X.XXHz'.")
    freq_of_interest = float(match.group(1))
    
    
    # Final PSD computation
    freqs_top, psd_top = welch(
        data_top, fs=fs, window=window,
        nperseg=nperseg
    )
    freqs_bot, psd_bot = welch(
        data_bot, fs=fs, window=window,
        nperseg=nperseg
    )

    transmissibility = psd_bot / psd_top

    # Extract peaks
    peak_freq_top = freqs_top[np.argmax(psd_top)]
    peak_val_top = np.max(psd_top)

    peak_freq_bot = freqs_bot[np.argmax(psd_bot)]
    peak_val_bot = np.max(psd_bot)

    peak_freq_trans = freqs_top[np.argmax(transmissibility)]
    peak_val_trans = np.max(transmissibility)
    
    # Save a plot of the TOP PSDs
    plt.plot(freqs_top, psd_top, label='Top PSD', color='cornflowerblue')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.xlim([freqs_top[1], 100])
    plt.savefig(os.path.join(output_dir, f"{it}_toppsd_{freq_of_interest:.2f}Hz.png"), dpi=300)
    plt.close()
    
    # Save a plot of the TOP PSDs log
    plt.loglog(freqs_top, psd_top, label='Top PSD', color='cornflowerblue')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.xlim([freqs_top[1], 100])
    plt.savefig(os.path.join(output_dir, f"{it}log_toppsd_{freq_of_interest:.2f}Hz.png"), dpi=300)
    plt.close()
    
    
    # Repeat for the BOTTOM PSDs
    plt.plot(freqs_bot, psd_bot, label='Bottom PSD', color='salmon')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.xlim([freqs_bot[1], 100])
    plt.savefig(os.path.join(output_dir, f"{it}_botpsd_{freq_of_interest:.2f}Hz.png"), dpi=300)
    plt.close()

    plt.loglog(freqs_bot, psd_bot, label='Bottom PSD', color='salmon')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.xaxis([freqs_bot[1], 100])
    plt.savefig(os.path.join(output_dir, f"{it}log_botpsd_{freq_of_interest:.2f}Hz.png"), dpi=300)
    plt.close()

    return {
        "driving_frequency": freq_of_interest,
        "fs_adj": fs,
        "nperseg": nperseg,
        "freqs_top": freqs_top,
        "psd_top": psd_top,
        "peak_freq_top": peak_freq_top,
        "peak_val_top": peak_val_top,
        "psd_bot": psd_bot,
        "peak_freq_bot": peak_freq_bot,
        "peak_val_bot": peak_val_bot,
        "transmissibility": peak_val_bot / peak_val_top,
        "peak_freq_trans": peak_freq_trans,
        "peak_val_trans": peak_val_trans
    }
    
    

def adjust_fs_and_compute_psd(output_dir, top_filepath, bot_filepath, nominal_fs=None,
                              min_cycles=5, max_cycles=18,
                              window="hamming", zero_pad=False):
    """
    Compute PSDs and transmissibility, correcting for optimal FFT bin alignment.
    """
    # Load and flatten acceleration data
    df_top = pd.read_csv(top_filepath)
    df_bot = pd.read_csv(bot_filepath)
    data_top = df_top.iloc[:, 0].dropna().to_numpy().flatten()
    data_bot = df_bot.iloc[:, 0].dropna().to_numpy().flatten()

    if data_top.shape != data_bot.shape:
        raise ValueError(f"Shape mismatch: {data_top.shape} vs {data_bot.shape}")

    waveform_length = min(len(data_top), len(data_bot))

    # Extract driving frequency
    match = re.search(r"([\d\.]+)Hz", os.path.basename(top_filepath))
    if not match:
        raise ValueError(f"Filename {top_filepath} does not contain 'X.XXHz'.")
    freq_of_interest = float(match.group(1))

    # Use optimal nperseg finder
    best_nperseg, best_k, best_peak_freq, best_bin_error, best_psd_error = optimal_nperseg(
        data_top, nominal_fs, freq_of_interest, waveform_length,
        min_cycles, max_cycles, window=window, noverlap=None
    )

    nfft_final = None if not zero_pad else 2 * best_nperseg
    safe_noverlap = min(best_nperseg - 1, best_nperseg // 2)

    # Final PSD computation
    freqs_top, psd_top = welch(
        data_top, fs=nominal_fs, window=window,
        nperseg=best_nperseg, noverlap=safe_noverlap, nfft=nfft_final
    )
    freqs_bot, psd_bot = welch(
        data_bot, fs=nominal_fs, window=window,
        nperseg=best_nperseg, noverlap=safe_noverlap, nfft=nfft_final
    )

    transmissibility = psd_bot / psd_top

    # Extract peaks
    peak_freq_top = freqs_top[np.argmax(psd_top)]
    peak_val_top = np.max(psd_top)

    peak_freq_bot = freqs_bot[np.argmax(psd_bot)]
    peak_val_bot = np.max(psd_bot)

    peak_freq_trans = freqs_top[np.argmax(transmissibility)]
    peak_val_trans = np.max(transmissibility)
    
    # Save a plot of the PSDs
    plt.plot(freqs_top, psd_top, label='Top PSD', color='cornflowerblue')
    plt.plot(freqs_bot, psd_bot, label='Bottom PSD', color='salmon')
    plt.savefig(os.path.join(output_dir, f"{it}_oppsd_{freq_of_interest:.2f}Hz.png"), dpi=300)

    return {
        "driving_frequency": freq_of_interest,
        "fs_adj": nominal_fs,
        "nperseg": best_nperseg,
        "nfft": nfft_final,
        "n_cycles": min_cycles,  # Starting cycle count; may not be final
        "freqs_top": freqs_top,
        "psd_top": psd_top,
        "peak_freq_top": peak_freq_top,
        "peak_val_top": peak_val_top,
        "psd_bot": psd_bot,
        "peak_freq_bot": peak_freq_bot,
        "peak_val_bot": peak_val_bot,
        "transmissibility": peak_val_bot / peak_val_top,
        "peak_freq_trans": peak_freq_trans,
        "peak_val_trans": peak_val_trans
    }


# Build a map {frequency: filepath} for top and bottom
def build_freq_map(file_list):
    freq_map = {}
    for fp in file_list:
        m = re.search(r"([\d\.]+)Hz", os.path.basename(fp))
        if m:
            freq = float(m.group(1))
            freq_map[freq] = fp
    return freq_map



if __name__ == "__main__":

    basedir = r"Z:\Users\Soyeon\JulyQZS\202508041538_bestYet2Hz_1.19kg_ch1top_ch2bot_x10_tf_3.0VSINusoid"
    fs = 10000
    zoom_max_lim = 100
    
    try:

        metadata = extract_metadata_from_path(basedir)
        mass = metadata["mass"]
        dampers = metadata["dampers"]

        print(f"Processing data for {mass}-{dampers}")


        label_peaks_zoomed = True
        label_peaks_total = False

        plot_tf_flag = True
        plot_psd_flag = True

        peak_params = {
            'height': 1e-7,
            'distance': None,
            'prominence': None,
            'width': None
        }
    except Exception as e:
        print("First block:")
        print(e)
        input("Enter to exit")

    try:
        ch1_files = sorted(glob.glob(os.path.join(basedir, "ch1_acceleration_data_*Hz.csv")))
        ch2_files = sorted(glob.glob(os.path.join(basedir, "ch2_acceleration_data_*Hz.csv")))

        ch1_dict = {extract_freq(f): f for f in ch1_files}
        ch2_dict = {extract_freq(f): f for f in ch2_files}

        shared_freqs = sorted(set(ch1_dict.keys()) & set(ch2_dict.keys()))
        print(f"Found {len(shared_freqs)} matching frequency pairs.")

        for driving_freq in shared_freqs:
            print(f"\n--- Processing {driving_freq} Hz ---")

            ch1 = ch1_dict[driving_freq]
            ch2 = ch2_dict[driving_freq]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            ch1_data = pd.read_csv(ch1).values.flatten()
            ch2_data = pd.read_csv(ch2).values.flatten()
            waveform_length = min(len(ch1_data), len(ch2_data))

            nperseg, _, _, _, _ = optimal_nperseg(
                ch1_data, fs, driving_freq, waveform_length,
                min_cycles=5, max_cycles=18, window='hamming', noverlap=None
            )
            windowType = f"opt_n{nperseg}"

            freq_ch1, psd_ch1 = welch(ch1_data, fs=fs, nperseg=nperseg)
            freq_ch2, psd_ch2 = welch(ch2_data, fs=fs, nperseg=nperseg)
            transmissibility = psd_ch2/psd_ch1

            # Save PSD and Transmissibility CSV
            psd_data = {
                'frequency': freq_ch1,
                'psd_ch1': psd_ch1,
                'psd_ch2': psd_ch2,
                'transmissibility': transmissibility
            }
            psd_df = pd.DataFrame(psd_data)
            psd_csvname = f"{timestamp}_psd_data_{driving_freq}Hz_{int(fs/1e3)}kS_{windowType}_{dampers}_{mass}.csv"
            psd_df.to_csv(os.path.join(basedir, psd_csvname), index=False)


            # Plot PSD
            if plot_tf_flag:
                plt.figure(figsize=(10, 6))
                plt.semilogy(freq_ch1, transmissibility, label ="$|V_{out}/V_{in}|$", c="darkorange", marker='.', linestyle='solid')
                
                if plot_psd_flag:
                    plt.semilogy(freq_ch1, psd_ch1, label="$V_{in}$", c="cornflowerblue", marker='.', linestyle='dashed')
                    plt.semilogy(freq_ch2, psd_ch2,  label="$V_{out}$", c="mediumblue", marker='.', linestyle='dashed')
                
                
                plt.grid(True, which='both', ls='--')
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Power Spectral Density [m²/s⁴/Hz]")
                plt.legend()
                plt.title(f"Transfer Fucntion and PSD ({mass})")
                total_name = f"{timestamp}_{driving_freq}Hz_nperseg{nperseg}_{windowType}_{dampers}_{mass}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(basedir, total_name), dpi=300)
                plt.close()


                plt.figure(figsize=(10, 6))
                plt.xlim(0, zoom_max_lim)
                plt.semilogy(freq_ch1, transmissibility, label ="$|V_{out}/V_{in}|$", c="darkorange", marker='.', linestyle='solid')
                
                if plot_psd_flag:
                    plt.semilogy(freq_ch1, psd_ch1, label="$V_{in}$", c="cornflowerblue", marker='.', linestyle='dashed')
                    plt.semilogy(freq_ch2, psd_ch2,  label="$V_{out}$", c="mediumblue", marker='.', linestyle='dashed')
                
                plt.grid(True, which='both', ls='--')
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Power Spectral Density [m²/s⁴/Hz]")
                plt.legend()
                plt.title(f"Zoomed-in Transfer Function and PSD ({mass})")
                zoomed_name = f"{timestamp}_PSD_zoomed{zoom_max_lim}Hz_nperseg{nperseg}_{windowType}_{dampers}_{mass}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(basedir, zoomed_name), dpi=300)
                plt.close()
    except Exception as e:
        print(e)
        input("Press enter to exit")
    
                