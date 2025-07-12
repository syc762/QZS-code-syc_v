import pandas as pd
import os
import matplotlib.pyplot as plt


basedir = r"Z:\Users\Soyeon\JulyQZS\250709_noDampers_0.381kg\nowpeakfinderop_202507101324_bestYet2Hz_ch1top_ch2bot_tf_3.0VSQUare"
filename = "tf_data_3.0Hz_10kS_n2002000_noDampers_0.381kg_20250712_133040.csv"


tf_data = pd.read_csv(os.path.join(basedir, filename), low_memory=False)

# Set your threshold value
threshold = 2e-9

# Apply filter: keep rows where both PSD_top and PSD_bot are above the threshold
filtered_df = tf_data[(tf_data['channel1'] > threshold) & (tf_data['channel2'] > threshold)]

# Optional: save the filtered results
filtered_df.to_csv("filtered_psd_data.csv", index=False)

# Optional: preview the result
print(filtered_df.head())