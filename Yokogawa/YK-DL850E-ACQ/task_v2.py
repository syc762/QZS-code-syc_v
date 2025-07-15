import os
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



#### Main ####
if __name__ == "__main__":

    plot_psd = True
    plot_tf = True
    plot_tf_psd_combined = True                                                                                                                    

    basedir = r"Z:\Users\Soyeon\JulyQZS\250709_noDampers_0.381kg\done3Hz_202507101324_bestYet2Hz_ch1top_ch2bot_tf_3.0VSQUare"
    fs =10000
    mass = "0.381kg"
    dampers = "noDampers"
    label_peaks_zoomed = True
    zoom_max_lim = 500
    label_peaks_total = False

    # Customizable peak finding parameters
    peak_params = {
        'height':1e-8,       # Example: 1e-7
        'distance': None,     # Example: 5 (min # bins between peaks)
        'prominence': None,   # Example: 1e-8
        'width': None         # Example: (2, 20)
    }

    from datetime import datetime

    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    ch1 = os.path.join(basedir, "ch1_acceleration_data_3Hz.csv")
    ch2 = os.path.join(basedir, "ch2_acceleration_data_3Hz.csv")

    freq_ch1 = extract_driving_frequency(ch1)
    freq_ch2 = extract_driving_frequency(ch2)

    if freq_ch1 != freq_ch2:
        print(f"Warning: Frequencies do not match: CH1 = {freq_ch1} Hz, CH2 = {freq_ch2} Hz")
    else:
        print(f"Both channels have the same frequency: {freq_ch1} Hz")

    driving_freq = freq_ch1
    ch1_data = pd.read_csv(ch1).values.flatten()
    ch2_data = pd.read_csv(ch2).values.flatten()
    nperseg = len(ch1_data) # int(len(ch1_data)//10) # 
    windowType = "full length" if nperseg == len(ch1_data) else f"{nperseg:.2f}"

    freq_ch1, psd_ch1 = ss.welch(ch1_data, fs=fs, nperseg=nperseg)
    freq_ch2, psd_ch2 = ss.welch(ch2_data, fs=fs, nperseg=nperseg)

    # ---- Save PSD Data to CSV ----
    psd_data = {
        'frequency': freq_ch1,
        'channel1': psd_ch1,
        'channel2': psd_ch2
    }
    psd_df = pd.DataFrame(psd_data)
    psd_csvname = f"{timestamp}_psd_data_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.csv"
    psd_df.to_csv(os.path.join(basedir, psd_csvname), index=False)

    # # ---- Peak Detection ----
    summary_ch1 = psd.find_peaks_at_odd_harmonics_summary(freq_ch1, psd_ch1, driving_freq, fs, tol=0.2, peak_params=peak_params)
    summary_ch1 = summary_ch1.rename(columns={
        "Peak_Freq_Hz": "CH1_Peak_Freq_Hz",
        "Peak_Height": "CH1_PSD",
        "Found": "CH1_Found"
    })

    summary_ch2 = psd.find_peaks_at_odd_harmonics_summary(freq_ch2, psd_ch2, driving_freq, fs, tol=0.2, peak_params=peak_params)
    summary_ch2 = summary_ch2.rename(columns={
        "Peak_Freq_Hz": "CH2_Peak_Freq_Hz",
        "Peak_Height": "CH2_PSD",
        "Found": "CH2_Found"
    })
    
    combined = pd.merge(summary_ch1, summary_ch2, on=["Harmonic", "Target_Freq_Hz"], how="outer")

    # # Fill missing CH1 PSDs
    # missing_ch1 = combined['CH1_Found'] == False
    # combined.loc[missing_ch1, 'CH1_PSD'] = combined.loc[missing_ch1, 'Target_Freq_Hz'].apply(
    #     lambda f: psd.get_psd_at_freq(freq_ch1, psd_ch1, f))

    # # Fill missing CH2 PSDs
    # missing_ch2 = combined['CH2_Found'] == False
    # combined.loc[missing_ch2, 'CH2_PSD'] = combined.loc[missing_ch2, 'Target_Freq_Hz'].apply(
    #     lambda f: psd.get_psd_at_freq(freq_ch2, psd_ch2, f))

    # # --- Final safeguard: fill any remaining NaNs using PSD curves ---
    # combined['CH1_PSD'] = combined.apply(
    #     lambda row: psd.get_psd_at_freq(freq_ch1, psd_ch1, row['Target_Freq_Hz']) 
    #     if pd.isna(row['CH1_PSD']) else row['CH1_PSD'], axis=1)

    # combined['CH2_PSD'] = combined.apply(
    #     lambda row: psd.get_psd_at_freq(freq_ch2, psd_ch2, row['Target_Freq_Hz']) 
    #     if pd.isna(row['CH2_PSD']) else row['CH2_PSD'], axis=1)

    for i in range(len(combined)):
        target_freq = combined.at[i, 'Target_Freq_Hz']

        # --- CH1 ---
        ch1_missing = (
            ('CH1_Found' in combined.columns and combined.at[i, 'CH1_Found'] == False)
            or pd.isna(combined.at[i, 'CH1_PSD'])
        )
        if ch1_missing:
            psd_value_ch1, actual_freq_ch1 = psd.get_psd_at_freq(freq_ch1, psd_ch1, target_freq)
            combined.at[i, 'CH1_PSD'] = psd_value_ch1
            combined.at[i, 'CH1_Peak_Freq_Hz'] = actual_freq_ch1

        # --- CH2 ---
        ch2_missing = (
            ('CH2_Found' in combined.columns and combined.at[i, 'CH2_Found'] == False)
            or pd.isna(combined.at[i, 'CH2_PSD'])
        )
        if ch2_missing:
            psd_value_ch2, actual_freq_ch2 = psd.get_psd_at_freq(freq_ch2, psd_ch2, target_freq)
            combined.at[i, 'CH2_PSD'] = psd_value_ch2
            combined.at[i, 'CH2_Peak_Freq_Hz'] = actual_freq_ch2


    # Compute the transmissibilities for the overtones
    combined['Transmissibility(Bot/Top)'] =  combined['CH2_PSD'] / combined['CH1_PSD']

    # ---- Save to CSV ----
    filename = f"{timestamp}_oddHarmonicPeaks_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.csv"
    save_path = os.path.join(basedir, filename)
    combined.to_csv(save_path, index=False)

    print(f"Peak summary saved to:\n{save_path}")
    

    # ---- Plot Zoomed PSD (0-100 Hz) ----
if plot_psd:
    plt.figure(figsize=(10, 6))
    plt.semilogy(freq_ch1, psd_ch1, label="CH1 PSD", color='orange')
    plt.semilogy(freq_ch2, psd_ch2, label="CH2 PSD", color='green')

    if label_peaks_zoomed:
        peaks_ch1_idx = [np.argmin(np.abs(freq_ch1 - f)) for f in summary_ch1["CH1_Peak_Freq_Hz"].dropna()]
        peaks_ch2_idx = [np.argmin(np.abs(freq_ch2 - f)) for f in summary_ch2["CH2_Peak_Freq_Hz"].dropna()]

        visible_ch1 = [idx for idx in peaks_ch1_idx if freq_ch1[idx] <= zoom_max_lim]
        visible_ch2 = [idx for idx in peaks_ch2_idx if freq_ch2[idx] <= zoom_max_lim]

        for idx in visible_ch1:
            plt.annotate(f"{freq_ch1[idx]:.1f}", (freq_ch1[idx], psd_ch1[idx]),
                         textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='orange')
        for idx in visible_ch2:
            plt.annotate(f"{freq_ch2[idx]:.1f}", (freq_ch2[idx], psd_ch2[idx]),
                         textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='green')

    label_str = "labeled" if label_peaks_zoomed else "nolabel"
    zoomed_title = f"PSDs at {driving_freq}Hz, {fs/1e3:.0f}kS/s, {dampers}, {mass}, {windowType} window"
    zoomed_filename = f"{timestamp}_PSD_zoomed{zoom_max_lim}Hz_{label_str}_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.png"
    plt.title(zoomed_title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [m²/s⁴/Hz]")
    plt.xlim(0, zoom_max_lim)
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(basedir, zoomed_filename), dpi=300)
    plt.close()
    print(f"Zoomed plot saved as: {zoomed_filename}")

    # ---- Plot Full Spectrum PSD ----
    plt.figure(figsize=(10, 6))
    plt.semilogy(freq_ch1, psd_ch1, label="CH1 PSD", color='orange')
    plt.semilogy(freq_ch2, psd_ch2, label="CH2 PSD", color='green')

    full_title = f"PSDs at {driving_freq}Hz, {fs/1e3:.0f}kS/s, {dampers}, {mass}, {windowType} window"
    full_filename = f"{timestamp}_PSD_full_{label_str}_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.png"
    plt.title(full_title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [m²/s⁴/Hz]")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(basedir, full_filename), dpi=300)
    plt.close()
    print(f"Full spectrum plot saved as: {full_filename}")
    

    # ---- Plot Transfer Function ----
    if plot_tf:
        tf_filename = f"{timestamp}_tf_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.png"
        plt.figure(figsize=(10, 6))
        
        plt.semilogy(combined['Target_Freq_Hz'], combined['Transmissibility(Bot/Top)'],
                     label="Transmissibility", color='salmon',
                     marker='o', markersize='3', linestyle=':')  # Limit x-axis to 1000 Hz
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Transmissibility")
        plt.title(f"Transmissibility at {driving_freq}Hz, {fs/1e3:.0f}kS/s, {dampers}, {mass}")
        plt.grid(True, which='both', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(basedir, tf_filename), dpi=300)
        plt.close()
        print(f"Transfer function plot saved as: {tf_filename}")

        tf_filename_zoomed = f"{timestamp}_tf_zoomed{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}.png"
        plt.figure(figsize=(10, 6))
        plt.xlim(0, zoom_max_lim)
        plt.semilogy(combined['Target_Freq_Hz'], combined['Transmissibility(Bot/Top)'], label="Transmissibility",
                     color='salmon', marker='.', linestyle=':')  # Limit x-axis to 1000 Hz
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Transmissibility")
        plt.title(f"Transmissibility at {driving_freq}Hz, {fs/1e3:.0f}kS/s, {dampers}, {mass}")
        plt.grid(True, which='both', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(basedir, tf_filename_zoomed), dpi=300)
        plt.close()
        print(f"Zoomed in transfer function plot saved as: {tf_filename}")
        
    
    # if plot_tf_psd_combined:
    #     # ---- Plot PSDs + Transmissibility on shared y-axis ----
    #     combined_filename = f"combined_PSD_TF_{driving_freq}Hz_{int(fs/1e3)}kS_n{nperseg}_{dampers}_{mass}_{timestamp}.png"

    #     plt.figure(figsize=(10, 6))

    #     # Plot both PSDs
    #     plt.semilogy(freq_ch1, psd_ch1, label='Ch1-Top PSD', color='cornflowerblue', linestyle='solid')
    #     plt.semilogy(freq_ch1, psd_ch2, label='Ch2-Bottom PSD', color='orange', linestyle='solid')

    #     # Plot transfer function
    #     plt.semilogy(freq_ch1, tf_df['normalized'], label='Transmissibility', color='salmon')

    #     # Axis labels and title
    #     plt.xlabel("Frequency [Hz]")
    #     plt.ylabel("Transmissibility or Power Spectral Density [m²/s⁴/Hz]")
    #     plt.title(f"PSDs & TF: {driving_freq}Hz, {fs/1e3:.0f}kS/s, {dampers}, {mass}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()

    #     # Save
    #     plt.savefig(os.path.join(basedir, combined_filename), dpi=300)
    #     print(f"Combined PSD+TF plot saved as: {combined_filename}")

