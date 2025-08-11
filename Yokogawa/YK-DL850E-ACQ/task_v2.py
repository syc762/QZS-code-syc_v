import os
import glob
import scipy.signal as ss
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np
import textwrap
import re
import psd as psd

# Function to extract frequency from filename
def extract_driving_frequency(filename):
    match = re.search(r'_(\d+(?:\.\d+)?)Hz', filename)
    if match:
        return float(match.group(1))
    return None

def extract_metadata_from_path(path):
    """
    Extract metadata from a directory path string, such as mass, gain, waveform type, and damper configuration.

    Parameters:
    -----------
    path : str
        The full directory path containing metadata in the folder name (e.g., '..._1.3511kg_x10_VSQUare_vhDamp...').

    Returns:
    --------
    dict
        A dictionary containing:
        - 'mass' (str): e.g., '1.3511kg' or 'unknown_mass'
        - 'gain' (str): e.g., 'x10' or 'unknown_gain'
        - 'waveform' (str): 'VSQUARE', 'VSINUSOID', or 'Unknown'
        - 'dampers' (str): e.g., 'vhDamp', 'hDamp', 'vDamp', 'Dampers', or 'noDampers'

    Notes:
    ------
    - Case-insensitive match is used for waveform detection.
    - If no matching keywords are found, default values are returned.
    """
    mass_match = re.search(r'(\d+\.\d+)kg', path)
    gain = re.search(r'x(\d+)', path)
    waveform = "VSQUARE" if "VSQU" in path.upper() else "VSINUSOID" if "VSIN" in path.upper() else "Unknown"

    damper_label = "noDampers"
    if "vhDamp" in path:
        damper_label = "vhDamp"
    elif "hDamp" in path:
        damper_label = "hDamp"
    elif "vDamp" in path:
        damper_label = "vDamp"
    elif re.search(r'damp', path, re.IGNORECASE):
        damper_label = "Dampers"

    return {
        "mass": f"{mass_match.group(1)}kg" if mass_match else "unknown_mass",
        "gain": f"x{gain.group(1)}" if gain else "unknown_gain",
        "waveform": waveform,
        "dampers": damper_label
    }



def get_filename(targetFile, extension='.csv'):
    # Get the base filename with extension
    base_filename = os.path.basename(targetFile)
    
    # Remove the extension
    filename_without_extension = os.path.splitext(base_filename)[0]
    
    return filename_without_extension

# Function that swaps input & outputs that are reversed
def switch_VinOut(f, col1=1, col2=2, col3=3):

    # Error check
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        try:
            df = pd.read_excel(f, sheet_name=None)
        except Exception as e:
            print(f"Error reading Excel: {e}")
            return 1

    # Get the filename to rename the new df
    directory = os.path.dirname(f)
    orig_filename = os.path.basename(f)
    new_filename = f"fixed_{orig_filename}"
    new_filepath = os.path.join(directory, new_filename)
    new_filelabel = get_filename(new_filename, extension='.csv')

    # Swap columns 1 and 2
    df.iloc[:,[col1, col2]] = df.iloc[:, [col2, col1]]
    
    # Handling division by 0
    df.iloc[:, col3] = np.where(df.iloc[:, col2] != 0,
        df.iloc[:, col1] / df.iloc[:, col2],
        np.nan  # or any other value you prefer for division by zero
    )

    df.to_csv(f, index=False)
    return df, new_filelabel


# Function to rename the csv files whose data columns are labeled channel 1 and channel 2
def rename_cols(f):
    # Error check
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        try:
            df = pd.read_excel(f, sheet_name=None)
        except Exception as e:
            print(f"Error reading Excel: {e}")
            return 1
    
    # Check whether the columns are labeled as channel1 and channel2
    if 'channel1' in df.columns:
        col1 = 'channel1'
        col2 = 'channel2'
        # The larger entries are from V_in, and the smaller entries are from V_out
        if df[col1].iloc[0] < df[col2].iloc[0]:
            new_names = {col1: 'V_out', col2: 'V_in'}
        else:
            new_names = {col1: 'V_in', col2: 'V_out'}

        df = df.rename(columns=new_names)

        # Get the filename to rename the new df
        #directory = os.path.dirname(f)
        #orig_filename = os.path.basename(f)
        #new_filename = f"fixed_{orig_filename}"
        #new_filepath = os.path.join(directory, new_filename)
        #new_filelabel = get_filename(new_filename, extension='.csv')
        df.to_csv(f,index=False)

    else:
        print("This file is already labeled by V_in and V_out.")

    



# Plots the transfer function and saves it
# Modifying it to return an Ax object for further manipulation
def plot_tf(f, save_dir, filename="someFile.png"):

    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(e)
        df = pd.read_excel(f, sheet_name=None)
        return 0

    x = df.iloc[:,0].to_numpy()
    normalized = df.iloc[:,3].to_numpy()

    if 'channel1' in df.columns:
        V_out = df.iloc[:,1].to_numpy() 
        V_in = df.iloc[:,2].to_numpy()
    elif 'V_in' in df.columns:
        V_out = df['V_out'].to_numpy()
        V_in = df['V_in'].to_numpy()
    else:
        print(f"Error reading in columns, check the column names")
        return 1

    
    plt.plot(x, normalized, label ="$|V_{out}/V_{in}|$", c="darkorange", marker='.', linestyle='solid')
    plt.plot(x, V_in, label="$V_{in}$", c="cornflowerblue", marker='.', linestyle='dashed')
    plt.plot(x, V_out, label="$V_{out}$", c="mediumblue", marker='.', linestyle='dashed')
    #plt.plot(df["frequency"], df["channel2"])
    #plt.plot(df["frequency"], df["channel1"])
    #plt.plot(df["frequency"], df["normalized"])
    plt.loglog()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Signal") # Attenuation
    
    title = get_filename(f)
    wrapped_title = textwrap.fill(title, width=52)
    plt.title(wrapped_title, fontsize=11, wrap=True)
    plt.legend()
    try:
        plt.savefig(os.path.join(save_dir, filename), transparent=True, bbox_inches='tight')
    except Exception as e:
        print(e)
    plt.close()
    

### In Progress ###
# Plots the transfer function and saves it
# Modifying it to return an Ax object for further manipulation
def plot_return_tf(f, save_dir, filename="someFile.png", save=False):

    try:
        df = pd.read_csv(f, encoding='unicode_escape')
    except Exception as e:
        print(e)
        return 0
    
    # freq = df['frequency'].to_numpy()
    # top = df['channel1'].to_numpy()
    # bot = df['channel2'].to_numpy()
    x = df.iloc[:,0].to_numpy()
    
    channel1 = df.iloc[:,1].to_numpy() 
    channel2 = df.iloc[:,2].to_numpy()
    normalized = df.iloc[:,3].to_numpy()
    # normalized = channel2/channel1
    _, ax = plt.subplots()
    ax.plot(x, channel1, label="$V_{in}$", c="mediumblue", marker='.', linestyle='dashed')
    ax.plot(x, channel2, label="$V_{out}$", c="orange", marker='.', linestyle='dashed')
    ax.plot(x, normalized, label ="$|V_{out}/V_{in}|$", c="cornflowerblue", marker='.', linestyle='solid')
    #plt.plot(df["frequency"], df["channel2"])
    #plt.plot(df["frequency"], df["channel1"])
    #plt.plot(df["frequency"], df["normalized"])
    ax.loglog()
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Signal") # Attenuation
    ax.set_title("Transfer Function")
    ax.legend()

    if save:
        plt.savefig(os.path.join(save_dir, filename))
    
    return ax

    

if __name__ == "__main__":
    
    # -------- CONFIG --------
    basedir = r"Z:\Users\Soyeon\JulyQZS\202508040929_bestYet2Hz_0.9383kg_ch1top_ch2bot_x10_tf_3.0VSQUare"
    fs = 10000
    zoom_max_lim = 100
    
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

    # -------- Find all matching pairs --------
    ch1_files = sorted(glob.glob(os.path.join(basedir, "ch1_acceleration_data_*Hz.csv")))
    ch2_files = sorted(glob.glob(os.path.join(basedir, "ch2_acceleration_data_*Hz.csv")))

    # Make dicts for fast lookup by frequency
    def extract_freq(file): return extract_driving_frequency(file)
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
        nperseg = len(ch1_data)
        windowType = "full length" if nperseg == len(ch1_data) else f"{nperseg:.2f}"

        freq_ch1, psd_ch1 = ss.welch(ch1_data, fs=fs, nperseg=nperseg)
        freq_ch2, psd_ch2 = ss.welch(ch2_data, fs=fs, nperseg=nperseg)

        # Save PSD
        psd_data = {
            'frequency': freq_ch1,
            'channel1': psd_ch1,
            'channel2': psd_ch2
        }
        psd_df = pd.DataFrame(psd_data)
        psd_csvname = f"{timestamp}_psd_data_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.csv"
        psd_df.to_csv(os.path.join(basedir, psd_csvname), index=False)

        # Peak summary
        if driving_freq != 0:
            summary_ch1 = psd.find_peaks_at_odd_harmonics_summary(freq_ch1, psd_ch1, driving_freq, fs, tol=0.2, peak_params=peak_params)
            summary_ch2 = psd.find_peaks_at_odd_harmonics_summary(freq_ch2, psd_ch2, driving_freq, fs, tol=0.2, peak_params=peak_params)

            summary_ch1 = summary_ch1.rename(columns={
                "Peak_Freq_Hz": "CH1_Peak_Freq_Hz",
                "Peak_Height": "CH1_PSD",
                "Found": "CH1_Found"
            })
            summary_ch2 = summary_ch2.rename(columns={
                "Peak_Freq_Hz": "CH2_Peak_Freq_Hz",
                "Peak_Height": "CH2_PSD",
                "Found": "CH2_Found"
            })

            combined = pd.merge(summary_ch1, summary_ch2, on=["Harmonic", "Target_Freq_Hz"], how="outer")

            for i in range(len(combined)):
                target_freq = combined.at[i, 'Target_Freq_Hz']

                if pd.isna(combined.at[i, 'CH1_PSD']):
                    psd_val, actual_freq = psd.get_psd_at_freq(freq_ch1, psd_ch1, target_freq)
                    combined.at[i, 'CH1_PSD'] = psd_val
                    combined.at[i, 'CH1_Peak_Freq_Hz'] = actual_freq

                if pd.isna(combined.at[i, 'CH2_PSD']):
                    psd_val, actual_freq = psd.get_psd_at_freq(freq_ch2, psd_ch2, target_freq)
                    combined.at[i, 'CH2_PSD'] = psd_val
                    combined.at[i, 'CH2_Peak_Freq_Hz'] = actual_freq

            combined['Transmissibility(Bot/Top)'] = combined['CH2_PSD'] / combined['CH1_PSD']

            filename = f"{timestamp}_oddHarmonicPeaks_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.csv"
            combined.to_csv(os.path.join(basedir, filename), index=False)
            print(f"Peak summary saved: {filename}")

        # Plot zoomed PSD
        if plot_psd_flag:
            plt.figure(figsize=(10, 6))
            plt.semilogy(freq_ch1, psd_ch1, label="CH1 PSD", color='orange')
            plt.semilogy(freq_ch2, psd_ch2, label="CH2 PSD", color='green')

            if label_peaks_zoomed:
                visible_peaks1 = combined["CH1_Peak_Freq_Hz"].dropna().values
                visible_peaks2 = combined["CH2_Peak_Freq_Hz"].dropna().values
                for f in visible_peaks1:
                    if f <= zoom_max_lim:
                        idx = np.argmin(np.abs(freq_ch1 - f))
                        plt.annotate(f"{f:.1f}", (freq_ch1[idx], psd_ch1[idx]),
                                     textcoords="offset points", xytext=(0, 5), fontsize=8, color='orange')
                for f in visible_peaks2:
                    if f <= zoom_max_lim:
                        idx = np.argmin(np.abs(freq_ch2 - f))
                        plt.annotate(f"{f:.1f}", (freq_ch2[idx], psd_ch2[idx]),
                                     textcoords="offset points", xytext=(0, -10), fontsize=8, color='green')

            plt.xlim(0, zoom_max_lim)
            plt.grid(True, which='both', ls='--')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power Spectral Density [m²/s⁴/Hz]")
            plt.legend()
            plt.title(f"Zoomed PSD ({driving_freq} Hz)")
            zoomed_name = f"{timestamp}_PSD_zoomed{zoom_max_lim}Hz_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(basedir, zoomed_name), dpi=300)
            plt.close()
            print(f"PSD zoomed saved: {zoomed_name}")

        # Plot TF
        if plot_tf_flag:
            tf_filename = f"{timestamp}_tf_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.png"
            plt.figure(figsize=(10, 6))
            plt.semilogy(combined['Target_Freq_Hz'], combined['Transmissibility(Bot/Top)'],
                         label="Transmissibility", color='salmon', marker='o', markersize=3, linestyle=':')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Transmissibility")
            plt.title(f"TF at {driving_freq} Hz")
            plt.grid(True, which='both', ls='--')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(basedir, tf_filename), dpi=300)
            plt.close()
            print(f"TF saved: {tf_filename}")

            # Plot Zoomed TF
            zoomed_tf_filename = f"{timestamp}_TF_zoomed{zoom_max_lim}Hz_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.png"
            plt.figure(figsize=(10, 6))
            plt.semilogy(combined['Target_Freq_Hz'], combined['Transmissibility(Bot/Top)'],
                        label="Transmissibility", color='salmon', marker='o', markersize=3, linestyle='-')

            plt.xlim(0, zoom_max_lim)
            plt.grid(True, which='both', ls='--')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Transmissibility")
            plt.title(f"Zoomed TF ({driving_freq} Hz)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(basedir, zoomed_tf_filename), dpi=300)
            plt.close()
            print(f"Zoomed TF saved: {zoomed_tf_filename}")

