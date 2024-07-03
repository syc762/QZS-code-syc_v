import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import numpy as np
from scipy.signal import find_peaks


matplotlib.rcParams.update({'font.size': 8})

def normalize(top, bottom, save=False, savedir="./", name="normalized"):
    top_df = pd.read_csv(top)
    xfreq = top_df.iloc[:,0]
    topfreq = top_df.iloc[:,1]

    bot_df = pd.read_csv(bottom)
    botfreq = bot_df.iloc[:,1]

    normfreq = [b/t for b,t in zip(botfreq, topfreq)]

    if save==True:
        norm_df = pd.DataFrame({'freq': list(xfreq),
                                'top': list(topfreq),
                                'bottom': list(botfreq),
                                'normalized': list(normfreq)

        })

        norm_df.to_csv(os.path.join(savedir, name))
    
    return xfreq, normfreq

def plotTf(xfreq, normfreq, title='Transfer function', labels=["Plate only"], num=1, save=True, savedir="~\\Users\students\Desktop\SoyeonChoi"):
    if num==1:
        fig, ax0 = plt.subplots(1,1,figsize=(8,8))
        line0, = ax0.plot(xfreq, normfreq, label0=labels[0])
        ax0.set_tittle(title, size=12)
        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax0.set_xlabel('Frequency [Hz]')
        ax0.set_ylabel('Attenuation |Vout/Vin|')
        ax0.legend()
    else:
        pass

def plotRes(xfreq, normfreq, title='Resonance Frequency', labels=["Plate only", "Fundamental Freq", "First Order"], num=1, save=True, savedir="~\\Users\students\Desktop\SoyeonChoi"):
    fig, ax0 = plt.subplots(1,1,figsize=(8,8))
    line0, = ax0.plot(xfreq, normfreq, label=labels[0])
    ax0.set_title(title, size=12)
    ax0.set_xlabel("Frequency [Hz]")
    ax0.set_ylabel("Attenuation |Vout\Vin|")
    ax0.legend

baseDir="~\\Desktop\SoyeonChoi\230815"
Top="c:\Users\students\Desktop\SoyeonChoi\230815\202308141919_transfer_data_TOP.csv"
Bot="c:\Users\students\Desktop\SoyeonChoi\230815\202308141919_transfer_data_Bottom_cleaner.csv"

print("Normalizing...")
xf, nf = normalize(Top, Bot, True, baseDir, "20230815_resonance_normalized")
