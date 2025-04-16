import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import numpy as np
from scipy.signal import find_peaks


matplotlib.rcParams.update({'font.size': 8})

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
 
    """
    fig, ax0 = plt.subplots(1,1,figsize=(8,8))
    line0, = ax0.plot(xfreq, normfreq, label=labels[0])
    line1, = ax0.plot(xfreq, topfreq, label=labels[1])
    line2, = ax0.plot(xfreq, botfreq, label=labels[2])
    ax0.set_title(title, size=12)
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel("Frequency [Hz]")
    ax0.set_ylabel("Attenuation |Vout\Vin|")
    ax0.legend()
    """
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

    
    

baseDir=r"C:\Users\students\Desktop\SoyeonChoi\QZS\20250305_channelsCombined_flexureV2.1_setupA.2_0rot_DAMPERS_67.1gAdded_tf_400points_4.0V"
Top="202503052110_transfer_data_botChannel1_topChannel2_flexureV2.1_setupA.2_0rot_DAMPERS_67.1gAdded_tf_400points_4.0V.csv"
Bot="202503121826_transfer_data_topChannel1_botChannel2_flexureV2.1_setupA.2_0rot_DAMPERS_67.1gAdded_tf_400points_4.0V.csv"

print("Normalizing...")
xf, nf, tf, bf = normalize(Top, Bot, True, baseDir, "channelsCombined_flexureV2.1_setupA.2_0rot_DAMPERS_67.1gAdded_tf_400points_4.0V.csv")

plotTf(xf, nf, tf, bf, "4V_Driving_Amp_flexureV2.1_transfer_function", ["Normalized", "V_in", "V_out"], baseDir)
