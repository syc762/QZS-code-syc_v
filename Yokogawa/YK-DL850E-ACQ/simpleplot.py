import os
import scipy.signal as ss
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np
import textwrap



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


### Directly use the data to obtain the transfer function
# fig, ax = plt.figure()



    
# plot_tf(files[0], os.get_cwd())


#### Main ####
if __name__ == "__main__":
    
    # filepath = os.path.expanduser("~\\Desktop\SoyeonChoi\20240716_afterGround_PE016_horizontalDamp_weight")
    # realFilepath = filepath
    realFilepath = r"Z:\Users\Soyeon\QZS\Newer QZS Data\unsure_OFFSET_20240722_afterGround_brokenFlexure_noSprings_trial"

    files = [os.path.join(realFilepath, f) for f in os.listdir(realFilepath) if f.endswith(".csv")]
    [print(f) for f in files]

    # targetFile = files[0]
    # https://stackoverflow.com/questions/46766530/python-split-a-string-by-the-position-of-a-character
    # Filename splicing

    
    for f in files:
        # filename = get_filename(f)
        # print(filename)
        try:
            
            # print(newDf)
            
            # newDf, new_filelabel = switch_VinOut(f)
            
            new_filelabel = "fixed_" + get_filename(f)
            rename_cols(f)
            
            print("Plotting " + new_filelabel)

            plot_tf(f, realFilepath, new_filelabel+".png")
        except Exception as e:
            print("error")
            print(e)
            pass
            
    
    # Try plotting using plot_return_tf




"""
# sampling frequency
fs = 1000
# number of data points used in each block for the FFT
n = 10
frequencies, psd = ss.welch(data, fs=fs, nperseg=n)

# For 60Hz signal, collect for at least 5s (~10s)
# File power spe
plt.figure()
plt.semilogy(frequencies, psd)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency')

springType = "Hello"
datestamp = datetime.now().strftime('%Y%m%d')
save_folder = f"{datestamp}_{springType}"

save_dir=os.path.join(os.path.expanduser("~\\Desktop\SoyeonChoi"), save_folder)
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir,'example2.png'))

df = pd.DataFrame()
""" 
