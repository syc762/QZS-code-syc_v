import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -------- Parameters --------
data_dir = r"Z:\Users\Soyeon\JulyQZS\250709_noDampers_0.381kg\noDampers_0.381kg_combinedTF"
pattern = "*oddHarmonicPeaks_*.csv"
zoom_xlim = 500  # Hz
save_dir = data_dir  # Change if you want plots saved elsewhere

# Optional: assign specific colors to specific labels (or leave blank for auto)
custom_colors = {
    # Example: "0.381kg-3.0Hz": "tab:orange",
    # "0.4441kg-1.0Hz": "tab:green"
}

# -------- Extract Meta Info from Directory --------
dampers = "noDampers" if "noDampers" in data_dir else "Dampers"
mass_match = re.search(r'_(\d+\.\d+)kg', data_dir)
mass_str = mass_match.group(1) + "kg" if mass_match else "unknownMass"
# Create timestamp string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -------- Helper: Label & Color --------
def extract_label_from_filename(filename):
    basename = os.path.basename(filename)
    freq_match = re.search(r'_oddHarmonicPeaks_([\d\.]+)Hz', basename)
    mass_match = re.search(r'_(\d+\.\d+)kg', basename)
    
    freq = freq_match.group(1) if freq_match else "?"
    mass = mass_match.group(1) if mass_match else "?"
    return f"{mass}kg-{freq}Hz"

def get_color_for_label(label):
    return custom_colors.get(label, None)

# -------- Load CSV Files --------
csv_files = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
if not csv_files:
    print("No matching CSV files found.")
else:
    print(f"Found {len(csv_files)} CSV files.")

# -------- Plot 1: Full Range --------
plt.figure(figsize=(10, 6))

for f in csv_files:
    try:
        df = pd.read_csv(f)
        x = df['CH2_Peak_Freq_Hz'].fillna(df['Target_Freq_Hz'])
        y = df['Transmissibility(Bot/Top)']
        label = extract_label_from_filename(f)
        color = get_color_for_label(label)
        plt.yscale('log')  # explicitly set log scale
        plt.scatter(x, y, s=2, label=label, color=color)
    except Exception as e:
        print(f"[Error reading {f}]: {e}")

title_full = f"Transmissibility vs Frequency — {mass_str}, {dampers}"
plt.title(title_full)
plt.xlabel("Peak Frequency [Hz]")
plt.ylabel("Transmissibility (CH2_PSD / CH1_PSD)")
plt.grid(True, which='both', linestyle='--')
plt.legend(fontsize=8, loc='best')
plt.tight_layout()

filename_full = f"{timestamp}_transmissibility_full_{mass_str}_{dampers}.png"
plt.savefig(os.path.join(save_dir, filename_full), dpi=300)
print(f"Saved full-range plot to: {filename_full}")

# -------- Plot 2: Zoomed In --------
plt.figure(figsize=(10, 6))

for f in csv_files:
    try:
        df = pd.read_csv(f)
        x = df['CH2_Peak_Freq_Hz'].fillna(df['Target_Freq_Hz'])
        y = df['Transmissibility(Bot/Top)']
        label = extract_label_from_filename(f)
        color = get_color_for_label(label)
        plt.yscale('log')  # explicitly set log scale, plt.semilogy refuses to plot a scatterplot
        plt.scatter(x, y, s=2, label=label, color=color)
    except Exception as e:
        print(f"[Error reading {f}]: {e}")

title_zoomed = f"Zoomed Transmissibility (x ≤ {zoom_xlim} Hz) — {mass_str}, {dampers}"
plt.title(title_zoomed)
plt.xlabel("Peak Frequency [Hz]")
plt.ylabel("Transmissibility (CH2_PSD / CH1_PSD)")
plt.xlim(0, zoom_xlim)
plt.grid(True, which='both', linestyle='--')
plt.legend(fontsize=8, loc='best')
plt.tight_layout()

filename_zoom = f"{timestamp}_transmissibility_zoomed{zoom_xlim}Hz_{mass_str}_{dampers}.png"
plt.savefig(os.path.join(save_dir, filename_zoom), dpi=300)
print(f"Saved zoomed-in plot to: {filename_zoom}")
