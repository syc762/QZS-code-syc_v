import os
import re
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import numpy as np
from scipy.signal import find_peaks


matplotlib.rcParams.update({'font.size': 8})

# Extract numeric mass values for sorting
def extract_mass(col_name):
    match = re.search(r"([\d.]+)\s*kg", col_name)
    return float(match.group(1)) if match else float('inf')  # put non-matching labels last



def normalize(top, bottom, save=False, savedir="./", name="normalized"):
    top_df = pd.read_csv(os.path.join(savedir, top))
    xfreq = top_df.iloc[:,0]
    topfreq = top_df.iloc[:,2]*0.01

    bot_df = pd.read_csv(os.path.join(savedir, bottom))
    botfreq = bot_df.iloc[:,2]

    normfreq = [b/t for b,t in zip(botfreq, topfreq)]

    if save==True:
        norm_df = pd.DataFrame({'freq': list(xfreq),
                                'top': list(topfreq),
                                'bottom': list(botfreq),
                                'normalized': list(normfreq)

        })

        norm_df.to_csv(os.path.join(savedir, name))
    
    return xfreq, normfreq, topfreq, botfreq

def plotTf(x, normalized, V_in, V_out, title='Transfer function', labels=["Plate only"], savedir="~\\Users\students\Desktop\SoyeonChoi"):
 
    # Labels used in the past: label="$V_{in}$", "$V_{out}$"
    plt.plot(x, normalized, label ="$|V_{out}/V_{in}|$", c="darkorange", marker='.', linestyle='solid')
    plt.plot(x, V_in, label="$V_{in}$", c="cornflowerblue", marker='.', linestyle='dashed')
    plt.plot(x, V_out, label="$V_{out}$", c="mediumblue", marker='.', linestyle='dashed')
    #plt.plot(df["frequency"], df["channel2"])
    #plt.plot(df["frequency"], df["channel1"])
    #plt.plot(df["frequency"], df["normalized"])
    plt.loglog()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Vout/Vin")
    plt.legend()
    
    plt.savefig(os.path.join(savedir, title+".jpg"))
    print("Saved")

    
    
if __name__ == "__main__":
    baseDir=r"C:\Users\students\Desktop\SoyeonChoi\dampers\damper_transfer_function_all_masses_interpolated.csv"
    save_dir = r"C:\Users\students\Desktop\SoyeonChoi"
    
    
    df = pd.read_csv(baseDir, encoding='unicode_escape')
    # df_peaks = pd.read_csv(r"C:\Users\students\Desktop\SoyeonChoi\c:\Users\students\Desktop\SoyeonChoi", encoding='unicode_escape')
    # df_peaks.columns = df_peaks.columns.str.strip()  # <-- Strip leading/trailing whitespace

    # Assume first column is frequency and read in the data
    frequencies = df.iloc[:, 0]
    columns = df.columns[1:]  # All other columns are different mass conditions


    # Sort columns based on mass values
    sorted_columns = sorted(columns, key=extract_mass)

    # Obtain the peak positions

    # Define color palette
    # adjusted_colors_list = ['cornflowerblue', 'lightcoral','gold']
    # For the varying mass plot:
    adjusted_colors_list =['dodgerblue','limegreen', 'gold', 'darkorange', 'firebrick']
    dampers_colors_list = ['mediumturquoise', 'mediumseagreen','gold','firebrick']

    # Pull up the peak values
    #peak_dict = {
    #    row['Mass']: f"{row['Mass']} ({row['Peak']})"
    #    for _, row in df_peaks.iterrows()
    #}

    peak_dict = {
        '0.459kg': '0.459kg (11.29Hz)',
        '0.821kg': '0.821kg (10.35Hz)',
        '0.950kg': '0.950kg (8.86Hz)',
        '1.117kg': '1.117kg (8.27Hz)',
        '2.000kg': '2.000kg (35.18Hz)'
    }

    damp_peak_dict = {
        '0.658kg': '0.658kg',
        '0.790kg': '0.790kg',
        '0.950kg': '0.950kg',
        '2.000kg': '2.000kg',
    }

    label_dict = {
        'top': r'$V_{\mathrm{in}}$',
        'bottom': r'$V_{\mathrm{out}}$',
        'normalized': r'$V_{\mathrm{out}} / V_{\mathrm{in}}$'
    }


    plt.figure(figsize=(4, 3))
    # Plot each transfer function
    print(sorted_columns)
    for i, col in enumerate(sorted_columns):
        tf_values = df[col]
        mass_label = damp_peak_dict.get(col, col)

        color = dampers_colors_list[i % len(dampers_colors_list)]
        plt.loglog(frequencies, df[col], 'o-', label=mass_label, markersize=1, linewidth=0.5, color=color)

    plt.xlabel("Frequency [Hz]", fontsize=7)
    plt.ylabel("Transmissibility", fontsize=7) # If PSD [m²/s⁴/Hz]
    plt.title("Transfer Functions for Varying Mass", fontweight='bold', fontsize=8)
    plt.grid(True, which="both", linestyle="--", markersize=2, linewidth=0.5)
    plt.legend(title="Hanging Mass", loc='best', fontsize=5, title_fontproperties={'weight':'bold', 'size':5.5})
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dampers_flexureV21._transfer_functions_by_mass_final.png"), dpi=300, bbox_inches='tight')