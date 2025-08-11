import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks

# -------- Parameters --------
data_dir = r"Z:\Users\Soyeon\JulyQZS\202508041304_bestYet2Hz_1.19kg_ch1top_ch2bot_x10_tf_3.0VSQUare"
pattern = "*oddHarmonicPeaks_*.csv"
zoom_xlim = 100  # Hz
save_dir = data_dir  # Change if you want plots saved elsewhere

# Optional: assign specific colors to specific labels (or leave blank for auto)
custom_colors = {
    # Example: "0.381kg-3.0Hz": "tab:orange",
    # "0.4441kg-1.0Hz": "tab:green"
}

# -------- Extract Meta Info from Directory --------
dampers = "Dampers" if "Dampers" in data_dir.lower() else "noDampers"
mass_match = re.search(r'_(\d+\.\d+)kg', data_dir)
mass_str = mass_match.group(1) + "kg" if mass_match else "0.498kg"
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



###################### Beginning of Main ########################
label_peaks = True


# -------- Load CSV Files --------
csv_files = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
if not csv_files:
    print("No matching CSV files found.")
else:
    print(f"Found {len(csv_files)} CSV files.")




# Arrays to store the transmissibility (y), driving frequency (x)
all_x = []
all_y = []
all_labels = []
all_colors = []

# -------- Plot 1: Full Range --------
plt.figure(figsize=(10, 6))

for f in csv_files:
    try:
        df = pd.read_csv(f)
        x = df['CH2_Peak_Freq_Hz'].fillna(df['Target_Freq_Hz'])
        y = df['Transmissibility(Bot/Top)']
        label = extract_label_from_filename(f)
        color = get_color_for_label(label)
        
        all_x.extend(x)
        all_y.extend(y)
        all_labels.extend([label] * len(x))
        all_colors.extend([color] * len(x))
        
    except Exception as e:
        print(f"[Error reading {f}]: {e}")

# Convert to numpy arrays
all_x = np.array(all_x)
all_y = np.array(all_y)
all_labels = np.array(all_labels)
all_colors = np.array(all_colors)

# Sort by x
sorted_indices = np.argsort(all_x)
x_sorted = all_x[sorted_indices]
y_sorted = all_y[sorted_indices]
labels_sorted = all_labels[sorted_indices]
colors_sorted = all_colors[sorted_indices]

# Save the data for easy access
# -------- Save Sorted Data to CSV --------
sorted_df = pd.DataFrame({
    'Frequency_Hz': x_sorted,
    'Transmissibility': y_sorted,
    'Label': labels_sorted
})

sorted_csv_name = f"{timestamp}_sortedTransmissibility_{mass_str}_{dampers}.csv"
sorted_df.to_csv(os.path.join(save_dir, sorted_csv_name), index=False)
print(f"Saved sorted data to: {sorted_csv_name}")


# # Group data by different color schemes
# unique_labels = np.unique(labels_sorted)

# for label in unique_labels:
#     mask = labels_sorted == label
#     plt.plot(x_sorted[mask], y_sorted[mask], marker='o', markersize=4,
#              linestyle='-', label=label, color=get_color_for_label(label))


# -------- Find Peaks --------
peak_indices, _ = find_peaks(y_sorted,
                             prominence=0.1,
                             distance=2)  # adjust prominence as needed
peak_freqs = x_sorted[peak_indices]
peak_vals = y_sorted[peak_indices]

# Save to CSV
peak_df = pd.DataFrame({
    'Peak_Frequency_Hz': peak_freqs,
    'Transmissibility': peak_vals
})

peak_csv_name = f"{timestamp}_peaks_{mass_str}_{dampers}.csv"
peak_df.to_csv(os.path.join(save_dir, peak_csv_name), index=False)
print(f"Saved peak data to: {peak_csv_name}")

title_full = f"Transmissibility vs Frequency — {mass_str}, {dampers}"
plt.title(title_full)
 # explicitly set log scale
plt.semilogy(x_sorted, y_sorted, marker='o', markersize=2, linestyle='--', color='tab:red')
plt.xlabel("Peak Frequency [Hz]")
plt.ylabel("Transmissibility (CH2_PSD / CH1_PSD)")
plt.grid(True, which='both', linestyle='--')
plt.legend(fontsize=8, loc='best')
plt.tight_layout()
if label_peaks:
    plt.semilogy(peak_freqs, peak_vals, 'v', markersize=4, label="Peaks")

    for x, y in zip(peak_freqs, peak_vals):
        plt.text(x, y * 1.05, f"{x:.1f}Hz", ha='center', va='bottom', fontsize=7, rotation=45)


filename_full = f"{timestamp}_transmissibility_full_{mass_str}_{dampers}.png"
plt.savefig(os.path.join(save_dir, filename_full), dpi=300)
print(f"Saved full-range plot to: {filename_full}")

# -------- Plot 2: Zoomed In --------
plt.figure(figsize=(10, 6))

title_zoomed = f"Zoomed Transmissibility (x ≤ {zoom_xlim} Hz) — {mass_str}, {dampers}"
plt.title(title_zoomed)
plt.semilogy(x_sorted, y_sorted, marker='o', markersize=2, linestyle='--', color='tab:red')
plt.xlabel("Peak Frequency [Hz]")
plt.ylabel("Transmissibility (CH2_PSD / CH1_PSD)")
plt.xlim(0, zoom_xlim)
plt.grid(True, which='both', linestyle='--')
plt.legend(fontsize=8, loc='best')
plt.tight_layout()
if label_peaks:
    plt.semilogy(peak_freqs, peak_vals, 'v', markersize=4, label="Peaks")

    for x, y in zip(peak_freqs, peak_vals):
        plt.text(x, y * 1.05, f"{x:.1f}Hz", ha='center', va='bottom', fontsize=7, rotation=45)


filename_zoom = f"{timestamp}_transmissibility_zoomed{zoom_xlim}Hz_{mass_str}_{dampers}.png"
plt.savefig(os.path.join(save_dir, filename_zoom), dpi=300)
print(f"Saved zoomed-in plot to: {filename_zoom}")


