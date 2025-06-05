from scipy.signal import welch
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

def run_welch_loop(example_waveform, fs_default=10000):
    while True:
        try:
            fs = float(input(f"Enter sampling frequency [Hz] (default={fs_default}): ") or fs_default)
            nperseg = int(input("Enter nperseg value (e.g., 1024): "))

            freqs, psd = welch(example_waveform, fs=fs, nperseg=nperseg)
            
            plt.figure(figsize=(5, 3))
            plt.semilogy(freqs, psd, label=f"nperseg={nperseg}, fs={fs}")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD [V²/Hz]")
            plt.title("Welch PSD Estimate")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()

            confirm = input("Are you satisfied with these parameters? (y/n): ").lower()
            if confirm == 'y':
                return fs, nperseg
        except Exception as e:
            print(f"Error: {e}. Please try again.")


def plot_psd(frequencies, psd_values, title="Power Spectral Density"):
    plt.figure(figsize=(8, 5))
    plt.loglog(frequencies, psd_values, marker='o')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("psd_plot.png", dpi=500)





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