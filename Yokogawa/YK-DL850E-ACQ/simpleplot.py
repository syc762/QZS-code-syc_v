import os
import scipy.signal as ss
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np
import re


def get_filename(targetFile, sep='data_', extension='.csv'):
    pos = targetFile.find(sep) + len(sep)
    # print("Index to splice at:")
    # print(pos)
    filename = targetFile[pos:-len(extension)]
    return filename


# Plots the transfer function and saves it
# Modifying it to return an Ax object for further manipulation
def plot_tf(f, save_dir, filename="someFile.png"):

    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(e)
        df = pd.read_excel(f, sheet_name=None)
        return 0
    
    # freq = df['frequency'].to_numpy()
    # top = df['channel1'].to_numpy()
    # bot = df['channel2'].to_numpy()
    x = df.iloc[:,0].to_numpy()
    
    channel1 = df.iloc[:,1].to_numpy() 
    channel2 = df.iloc[:,2].to_numpy()
    normalized = df.iloc[:,3].to_numpy()
    plt.plot(x, channel1, label="$V_{in}$", c="mediumblue", marker='.', linestyle='dashed')
    plt.plot(x, channel2, label="$V_{out}$", c="grey", marker='.', linestyle='dashed')
    plt.plot(x, normalized, label ="$|V_{out}/V_{in}|$", c="cornflowerblue", marker='.', linestyle='solid')
    #plt.plot(df["frequency"], df["channel2"])
    #plt.plot(df["frequency"], df["channel1"])
    #plt.plot(df["frequency"], df["normalized"])
    plt.loglog()
    plt.xlabel("Frequency")
    plt.ylabel("Signal") # Attenuation
    plt.title("Transfer Function")
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    

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
    _, ax = plt.subplots()
    ax.plot(x, channel1, label="$V_{in}$", c="mediumblue", marker='.', linestyle='dashed')
    ax.plot(x, channel2, label="$V_{out}$", c="grey", marker='.', linestyle='dashed')
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
    
    filepath = os.path.expanduser("~\\Desktop\SoyeonChoi\QZS")
    realFilepath = os.path.join(filepath, "231120")
    print("done!")

    files = [os.path.join(realFilepath, f) for f in os.listdir(realFilepath) if f.endswith(".csv")]
    # [print(f) for f in files]

    targetFile = files[0]
    # https://stackoverflow.com/questions/46766530/python-split-a-string-by-the-position-of-a-character
    # Filename splicing

    for f in files:
        filename = get_filename(f)
        print(filename)
        try:
            plot_tf(f, realFilepath, filename+".png")
        except Exception as e:
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
