import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime
from collections import defaultdict
from matplotlib import cm
import matplotlib.colors as mcolors
import plotly.graph_objs as go
from plotly.subplots import make_subplots



# -------- CONFIG --------
root_dir = r"Z:\Users\Soyeon\JulyQZS\202507211238_bestYet2Hz_0.7076kg_ch1top_ch2bot_x10_tf_3.0VSQUare"
pattern = "*oddHarmonicPeaks_*Hz_*kS_n*_dampers_*.csv"
zoom_xlim = 100
label_peaks = False


custom_colors = {}


# To produce a single plot with all transfer functions
global_data = defaultdict(lambda: {'x': [], 'y': []})


# -------- Helpers --------
def extract_label_from_filename(filename):
    basename = os.path.basename(filename)
    freq_match = re.search(r'_oddHarmonicPeaks_([\d\.]+)Hz', basename)
    mass_match = re.search(r'_(\d+\.\d+)kg', basename)
    freq = freq_match.group(1) if freq_match else "?"
    mass = mass_match.group(1) if mass_match else "?"
    return f"{mass}kg-{freq}Hz"

def extract_mass_float(group_key):
    match = re.search(r"([\d\.]+)kg", group_key)
    return float(match.group(1)) if match else float('inf')


def get_color_for_label(label):
    return custom_colors.get(label, None)

def plot_all_from_dir(data_dir):
    print(f"\n[Processing] {data_dir}")
    csv_files = glob.glob(os.path.join(data_dir, pattern))

    if not csv_files:
        print("No matching CSV files found.")
        return

    dampers = "noDampers" if "noDampers" in data_dir else "Dampers"
    mass_match = re.search(r'_(\d+\.\d+)kg', data_dir)
    mass_str = mass_match.group(1) + "kg" if mass_match else "unknownMass"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_x, all_y = [], []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            x = df['CH2_Peak_Freq_Hz'].fillna(df['Target_Freq_Hz'])
            y = df['Transmissibility(Bot/Top)']
            label = extract_label_from_filename(f)

            all_x.extend(x)
            all_y.extend(y)

           
        except Exception as e:
            print(f"[Error reading {f}]: {e}")

    # Sort
    all_x, all_y = np.array(all_x), np.array(all_y)
    sorted_idx = np.argsort(all_x)
    x_sorted = all_x[sorted_idx]
    y_sorted = all_y[sorted_idx]


    # Save for global plot
    group_key = f"{mass_str} {dampers}"
    global_data[group_key]['x'].extend(all_x)
    global_data[group_key]['y'].extend(all_y)


    # Peaks
    peak_indices, _ = find_peaks(y_sorted, prominence=0.05)
    peak_freqs, peak_vals = x_sorted[peak_indices], y_sorted[peak_indices]

    # --- Plot Full ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(x_sorted, y_sorted, marker='o', markersize=2, linestyle='--', color='tab:pink')
    if label_peaks:
        plt.semilogy(peak_freqs, peak_vals, 'v', markersize=4, label="Peaks")
        plt.legend(fontsize=8)

    plt.title(f"Transmissibility vs Frequency — {mass_str}, {dampers}")
    plt.xlabel("Peak Frequency [Hz]")
    plt.ylabel("Transmissibility (CH2_PSD / CH1_PSD)")
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

    filename_full = f"{timestamp}_transmissibility_full_{mass_str}_{dampers}.png"
    plt.savefig(os.path.join(data_dir, filename_full), dpi=300)
    print(f"Saved full-range plot: {filename_full}")

    # Save Peaks CSV
    peak_df = pd.DataFrame({
        'Peak_Frequency_Hz': peak_freqs,
        'Transmissibility': peak_vals
    })
    peak_csv_name = f"{timestamp}_peaks_{mass_str}_{dampers}.csv"
    peak_df.to_csv(os.path.join(data_dir, peak_csv_name), index=False)
    print(f"Saved peak CSV: {peak_csv_name}")

    # --- Plot Zoomed ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(x_sorted, y_sorted, marker='o', markersize=2, linestyle='--', color='tab:pink')
    if label_peaks:
        plt.semilogy(peak_freqs, peak_vals, 'v', markersize=4, label="Peaks")
        plt.legend(fontsize=8)

    plt.title(f"Zoomed Transmissibility (x ≤ {zoom_xlim} Hz) — {mass_str}, {dampers}")
    plt.xlabel("Peak Frequency [Hz]")
    plt.ylabel("Transmissibility (CH2_PSD / CH1_PSD)")
    plt.grid(True, which='both', linestyle='--')
    plt.xlim(0, zoom_xlim)
    plt.tight_layout()

    filename_zoom = f"{timestamp}_transmissibility_zoomed{zoom_xlim}Hz_{mass_str}_{dampers}.png"
    plt.savefig(os.path.join(data_dir, filename_zoom), dpi=300)
    print(f"Saved zoomed plot: {filename_zoom}")


# -------- MAIN --------
if __name__ == "__main__":
    for root, dirs, files in os.walk(root_dir):
        for subdir in dirs:
            #if "combinedTF" in subdir:
            plot_all_from_dir(os.path.join(root, subdir))

    # -------- Global Plot --------
    print("\n[Creating Global Plot of All Data]")
    if not global_data:
        print("No data found to plot globally.")
    else:

        peak_summary_rows = []

        num_groups = len(global_data)
        colormap = cm.get_cmap("plasma", num_groups)
        
       
        sorted_group_keys = sorted(global_data.keys(), key=extract_mass_float)

        for i, group_key in enumerate(sorted_group_keys):
            rgba = colormap(i)
            custom_colors[group_key] = mcolors.to_hex(rgba)

        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax_full, ax_zoom = axes

        for group_key in sorted_group_keys:
            data = global_data[group_key]
            x_vals = np.array(data['x'])
            y_vals = np.array(data['y'])

            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            y_sorted = y_vals[sort_idx]

            color = get_color_for_label(group_key)

            # Plot on both axes
            ax_full.semilogy(x_sorted, y_sorted, marker='.', linestyle='-', linewidth=0.5, markersize=0.5,
                 label=group_key, color=color)
            ax_zoom.semilogy(x_sorted, y_sorted, marker='.', linestyle='-', linewidth=0.5, markersize=0.5,
                            label=group_key, color=color)


            # -------- Peak Detection --------
            peaks, _ = find_peaks(y_sorted, prominence=0.05)
            peak_freqs = x_sorted[peaks]
            peak_vals = y_sorted[peaks]

            # Filter peaks to < 200 Hz
            mask = peak_freqs < 200
            filtered_freqs = peak_freqs[mask]
            filtered_vals = peak_vals[mask]

            # Sort by peak height and take top 2
            if len(filtered_vals) > 0:
                top_indices = np.argsort(filtered_vals)[-2:]  # top 2 highest
                for pf, pv in zip(filtered_freqs[top_indices], filtered_vals[top_indices]):
                    ax_zoom.annotate(f"{pf:.1f} Hz", xy=(pf, pv),
                                    xytext=(pf + 5, pv * 1.5),
                                    textcoords='data',
                                    fontsize=7, color=color,
                                    arrowprops=dict(arrowstyle='-', lw=0.5, color=color))

                # Save to CSV structure
                for pf, pv in zip(filtered_freqs[top_indices], filtered_vals[top_indices]):
                    peak_summary_rows.append({
                        "Group": group_key,
                        "Peak_Freq_Hz": pf,
                        "Transmissibility": pv
                    })


        # Full plot
        ax_full.set_title("Transfer functions (full range)")
        ax_full.set_ylabel("Transmissibility")
        ax_full.grid(True, which='both', linestyle='--')

        # Zoomed plot
        ax_zoom.set_title(f"Transfer functions (zoomed to ≤ {zoom_xlim} Hz)")
        ax_zoom.set_xlabel("Driving Frequency [Hz]")
        ax_zoom.set_ylabel("Transmissibility")
        ax_zoom.set_xlim(0, zoom_xlim)
        ax_zoom.grid(True, which='both', linestyle='--')

        # Legend
        ax_full.legend(fontsize=8, loc='best')

        plt.tight_layout()
        global_filename = f"ALL_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}_tf_dualplot.png"
        fig.savefig(os.path.join(root_dir, global_filename), dpi=300)
        print(f"Saved dual-range global plot: {global_filename}")


        # -------- Export Global Peaks CSV --------
        peak_df = pd.DataFrame(peak_summary_rows)
        csv_filename = f"ALL_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}_top2peaks_below200Hz.csv"
        peak_df.to_csv(os.path.join(root_dir, csv_filename), index=False)
        print(f"Saved peak summary CSV: {csv_filename}")

print("\n[Creating Interactive Plotly Global Plot]")



# Create Plotly figure
fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                    subplot_titles=("Transfer functions (full range)",
                                    f"Transfer functions (zoomed to ≤ {zoom_xlim} Hz)"))

# Add a trace for each dataset
for group_key in sorted_group_keys:
    data = global_data[group_key]
    x_vals = np.array(data['x'])
    y_vals = np.array(data['y'])

    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]

    color = custom_colors.get(group_key, None)

    # Add to both subplots
    fig.add_trace(
        go.Scatter(x=x_sorted, y=y_sorted, mode='lines+markers',
                   name=group_key, line=dict(width=1, color=color),
                   marker=dict(size=3),
                   legendgroup=group_key, showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_sorted, y=y_sorted, mode='lines+markers',
                   name=group_key, line=dict(width=1, color=color),
                   marker=dict(size=3),
                   legendgroup=group_key, showlegend=False),  # hide 2nd legend entry
        row=2, col=1
    )

# Zoom second subplot
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(type="log", row=2, col=1)
fig.update_xaxes(title_text="Driving Frequency [Hz]", row=2, col=1)
fig.update_yaxes(title_text="Transmissibility", row=1, col=1)
fig.update_yaxes(title_text="Transmissibility", row=2, col=1)
fig.update_xaxes(range=[0, zoom_xlim], row=2, col=1)

fig.update_layout(
    height=800,
    title_text="Interactive Global Transfer Function Plot",
    legend=dict(title="Mass Groups", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    hovermode="x unified"
)

# Save to HTML file
html_filename = f"ALL_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}_interactive_tf.html"
fig.write_html(os.path.join(root_dir, html_filename))
print(f"Saved interactive plotly plot: {html_filename}")

