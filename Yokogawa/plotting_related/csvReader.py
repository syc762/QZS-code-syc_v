import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def linear(x, a, b):
    return a*x + b

file = r"C:\Users\students\Desktop\SoyeonChoi\QZS\position_sensor_calibration.csv"
saveDir = r"C:\Users\students\Desktop\SoyeonChoi\QZS"

# Skip the first row that contains headers

f = np.genfromtxt(file, delimiter=',', dtype='str', skip_header=True)
f = f.astype(float)

#print(f.shape)
ch1 = f[:,0]
ch2 = f[:,-1]

# Fit the linear part of the Displace
range = [100, 350000]
fitx = ch1[range[0]:range[1]]
fity = ch2[range[0]:range[1]]

initial_guess=[-1.0, 0.75]
params, pcov = curve_fit(linear, fitx, fity, p0=initial_guess)

a, b = params
#print("Slope:" + str(a))
#print("Offset: " + str(b))


colors = plt.cm.tab10.colors

fig, ax = plt.subplots(2,2)
fig.suptitle('Linear Position Sensor Calibration', fontsize=10)
# 1s/division, total of 10 divisions
x = np.linspace(0, 10, len(ch1))

# First Figure
ax[0,0].scatter(x, ch1, color=colors[0])
ax[0,0].set_title('Force Sensor Output', fontsize=9)
ax[0,0].set_xlabel('Time Steps [a.u.]', fontsize=8)
ax[0,0].set_ylabel('Measured Voltage [V]', fontsize=8)

# Second Figure
ax[0,1].scatter(x, ch2, color=colors[-1])
ax[0,1].set_title('Displacement Sensor Output', fontsize=9)
ax[0,1].set_xlabel('Time Steps [a.u.]', fontsize=8)
ax[0,1].set_ylabel('Measured Voltage [V]', fontsize=8)

# Third Figure
ax[1,0].scatter(ch1, ch2, color=colors[4], s=1)
#ax[1,0].vlines(fitx[0], ymin=fity[0], ymax=fity[-1], linestyle='--', color='orange')
#ax[1,0].vlines(fitx[-1], ymin=fity[0], ymax=fity[-1], linestyle='--', color='orange')
#ax[1,0].hlines(fity[0], xmin=fitx[-1], xmax=fitx[0], linestyle='--', color='orange')
#ax[1,0].hlines(fity[-1], xmin=fitx[-1], xmax=fitx[0], linestyle='--', color='orange')

# For comparison
ax[1,0].vlines(ch1[range[0]], ymin=fity[0], ymax=fity[-1], linestyle='--', color='orange')
ax[1,0].vlines(ch1[range[-1]], ymin=fity[0], ymax=fity[-1], linestyle='--', color='orange')
ax[1,0].hlines(ch2[range[0]], xmin=ch1[range[-1]], xmax=ch1[range[0]], linestyle='--', color='orange')
ax[1,0].hlines(ch2[range[-1]], xmin=ch1[range[-1]], xmax=ch1[range[0]], linestyle='--', color='orange')

ax[1,0].plot(fitx, linear(fitx, a, b), linestyle='solid', color='orange')
ax[1,0].set_title('Displacement vs. Applied Force Curve', fontsize=9)
ax[1,0].set_xlabel('Force Sensor [V]', fontsize=8)
ax[1,0].set_ylabel('Displacement Sensor [V]', fontsize=8)


# Fourth Figure
ax[1,1].scatter(fitx, fity, label='Raw data', color=colors[4])
equation = f'Fitted Function: y = {a:.2f}*x + {b:.2f}'
ax[1,1].plot(fitx, linear(fitx, a, b), color='orange', label=f'Fitted Function: y = {a:.2f}*x + {b:.2f}')
ax[1,1].set_title('Fitted Spring Constant of Position Sensor', fontsize=9)
ax[1,1].set_xlabel('Applied Force [V]', fontsize=8)
ax[1,1].set_ylabel('Measured Displacement [V]', fontsize=8)
ax[1,1].legend(fontsize=6)

plt.tight_layout()
plt.savefig(os.path.join(saveDir, "position_sensor_calibration_fig.png"))








