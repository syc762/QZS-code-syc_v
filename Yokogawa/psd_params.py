import os
import re
import zipfile
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks


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
    plt.savefig(f"raw_data/tf_{base_name}.png", transparent=True, dpi=300)

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



def adjust_fs_and_compute_psd(top_filepath, bot_filepath, nominal_fs=None,
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
    plt.savefig(f"raw_data/{it}_psd_{freq_of_interest:.2f}Hz.png", dpi=300)

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
    
    it = '12it_0.381kg'

    # 1) Paths to the two uploaded zip files
    top_zip_path = '/home/vscode/QZS-code-syc_v/202505231501_top_bestYet2Hz_flexureOnly_0.8718kg_1to26Hz_vol67_6sSleep_tf_100points_2.0V_unamp_x1.zip'
    bot_zip_path = '/home/vscode/QZS-code-syc_v/202505231544_bot_bestYet2Hz_flexureOnly_0.8718kg_1to26Hz_vol67_6sSleep_tf_100points_2.0V_unamp_x1.zip'
    
    # Create directories for extraction
    os.makedirs('top_unzip', exist_ok=True)
    os.makedirs('bot_unzip', exist_ok=True)

    # 1) Unzip the archives
    with zipfile.ZipFile(top_zip_path, 'r') as z:
        z.extractall('top_unzip')
    with zipfile.ZipFile(bot_zip_path, 'r') as z:
        z.extractall('bot_unzip')

    # 2) Recursively find all CSVs in each unzip folder
    top_files = glob.glob(os.path.join('top_unzip', '**', 'acceleration_data_*.csv'), recursive=True)
    bot_files = glob.glob(os.path.join('bot_unzip', '**', 'acceleration_data_*.csv'), recursive=True)

    top_map = build_freq_map(top_files)
    bot_map = build_freq_map(bot_files)

    # Find all driving frequencies common to both
    common_freqs = sorted(set(top_map.keys()) & set(bot_map.keys()))
    osc_sampling_freq = 10000
    
    # 3) For each common frequency, compute PSD and transmissibility metrics
    results_list = []
    for freq in common_freqs:
        top_fp = top_map[freq]
        bot_fp = bot_map[freq]
        # initial fs_values to go off off
        
        try:
            metrics = adjust_fs_and_compute_psd(
                top_fp, bot_fp,
                nominal_fs=osc_sampling_freq,    # initial guess; will be adjusted internally
                min_cycles=5,
                max_cycles=18,
                window='hamming',
                zero_pad=False       # no zero padding
            )
            results_list.append(metrics)
            
            # Plot the PSDs and transmissibility
            """
            plot_psd_for_driving_frequency(it,
                metrics['driving_frequency'],
                metrics['fs_adj'],
                metrics['nperseg'],
                metrics['nfft'],
                metrics['n_cycles'],
                metrics['freqs_top'],
                metrics['psd_top'],
                metrics['peak_freq_top'],
                metrics['peak_val_top'],
                metrics['psd_bot'],
                metrics['peak_freq_bot'],
                metrics['peak_val_bot'],
                metrics['transmissibility'],
                metrics['peak_freq_trans'],
                metrics['peak_val_trans']
            )
            """
            

        except Exception as e:
            # Skip any frequency that fails
            print(f"Skipped {freq} Hz: {e}")

    # Convert the list of dicts into a DataFrame
    df_results = pd.DataFrame(results_list)
    if df_results.empty:
        raise RuntimeError("All frequencies failed. Check earlier error messages.")

    df_results = df_results.sort_values(by='driving_frequency').reset_index(drop=True)
   
    base_name = os.path.splitext(os.path.basename(top_zip_path))[0]
    csv_name = f"{it}_{base_name}.csv"
    df_results.to_csv(csv_name, index=False)

    # Plot and save the results
    plot_psd(df_results, f"{it}_tf")
    print(f"Results saved to {csv_name} and plots saved as {it}_{base_name}.png")