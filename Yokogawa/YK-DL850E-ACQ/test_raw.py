import pandas as pd

df = pd.read_csv(r'C:\Users\students\Documents\GitHub\QZS-code-syc_v\raw_data\20241113_151253_raw_data.csv')

processed_df = df * 50.00000E-03 * 10 / 24000

print(processed_df.head())