# LIBRARIES
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "py_neuromodulation":
    # this check is necessary, so we can also run the script from the root directory
    SCRIPT_DIR = os.path.join(SCRIPT_DIR, "examples")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import py_neuromodulation as nm
import xgboost
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
)
from sklearn import metrics, model_selection
import json
import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy import signal
from scipy import stats

# Read SEEG file 
PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1703\sEEG-sz\xxxxxx~ xxxxxx_2f0f3fec-1d36-4e97-b7f9-169a66744a68.EDF"
#PATH_RNS = "/Volumes/Nexus2/RNS_DataBank/PITT/PIT-RNS1703/sEEG-Sz/xxxxxx~ xxxxxx_2f0f3fec-1d36-4e97-b7f9-169a66744a68.EDF"

line_noise_freqs = np.arange(60,1000,60)

#raw = mne.io.read_raw_edf(PATH_RNS)
raw = mne.io.read_raw_edf(PATH_RNS, preload=True)
raw.notch_filter(line_noise_freqs)


raw.resample(200)


# Plot Raw file
#raw.plot()

# Plot PSD
#raw.plot_psd(picks='RSF11')

for i in range(len(raw.annotations)):
    if(raw.annotations[i]['description'].lower() == 'szonset'):
        sz_index = i

sz_time = raw.annotations[sz_index]['onset']
intrvl_st = raw.time_as_index(sz_time-20)[0]
intrvl_end = raw.time_as_index(sz_time+20)[0]

#y = raw.get_data(return_times=True)

""" 
ch1_interval = raw.get_data(return_times=True)[0][0,intrvl_st:intrvl_end]
times_interval = raw.get_data(return_times=True)[1][intrvl_st:intrvl_end]
fs = raw.info['sfreq']
w = 6.
freq = np.linspace(1, fs/2, 250)
widths = w*fs / (2*freq*np.pi)

cwtm = signal.cwt(ch1_interval, signal.morlet2, widths, w=w)

plt.pcolormesh(times_interval, freq, stats.zscore(np.abs(cwtm)), cmap='viridis', shading='gouraud')
plt.plot([sz_time, sz_time], [0,250], color="red", linewidth=3)
plt.clim(-0.5, 0.5)
plt.show()

print("DONE1")

"""


rows = 4
cols = 4

fig, axs = plt.subplots(rows, cols)

fs = raw.info['sfreq']
w = 6.
freq = np.linspace(1, fs/2, 100)
widths = w*fs / (2*freq*np.pi)

for i in range(rows*cols):
    ch_interval = raw.get_data(return_times=True)[0][0+i,intrvl_st:intrvl_end]
    times_interval = raw.get_data(return_times=True)[1][intrvl_st:intrvl_end]

    cwtm = signal.cwt(ch_interval, signal.morlet2, widths, w=w)
    axs[i//rows, i%cols].pcolormesh(times_interval, freq, stats.zscore(np.abs(cwtm)), cmap='viridis', shading='gouraud',vmin=-0.5,vmax=0.5)
    #axs[i//8, i%8].set_clim(-0.5, 0.5)
    axs[i//rows, i%cols].plot([sz_time, sz_time], [1,100], color="red", linewidth=2)
    axs[i//rows, i%cols].set_title(raw.info['ch_names'][0+i])


plt.tight_layout()
plt.show()
#plt.savefig("cwtplot.png")


rows = 4
cols = 4

fig, axs = plt.subplots(rows, cols)

for i in range(rows*cols):
    ch_interval = raw.get_data(return_times=True)[0][16+i,intrvl_st:intrvl_end]
    times_interval = raw.get_data(return_times=True)[1][intrvl_st:intrvl_end]

    cwtm = signal.cwt(ch_interval, signal.morlet2, widths, w=w)
    axs[i//rows, i%cols].pcolormesh(times_interval, freq, stats.zscore(np.abs(cwtm)), cmap='viridis', shading='gouraud',vmin=-0.5,vmax=0.5)
    #axs[i//8, i%8].set_clim(-0.5, 0.5)
    axs[i//rows, i%cols].plot([sz_time, sz_time], [1,100], color="red", linewidth=2)
    axs[i//rows, i%cols].set_title(raw.info['ch_names'][16+i])


plt.tight_layout()
plt.show()


rows = 4
cols = 4

fig, axs = plt.subplots(rows, cols)

for i in range(rows*cols):
    ch_interval = raw.get_data(return_times=True)[0][32+i,intrvl_st:intrvl_end]
    times_interval = raw.get_data(return_times=True)[1][intrvl_st:intrvl_end]

    cwtm = signal.cwt(ch_interval, signal.morlet2, widths, w=w)
    axs[i//rows, i%cols].pcolormesh(times_interval, freq, stats.zscore(np.abs(cwtm)), cmap='viridis', shading='gouraud',vmin=-0.5,vmax=0.5)
    #axs[i//8, i%8].set_clim(-0.5, 0.5)
    axs[i//rows, i%cols].plot([sz_time, sz_time], [1,100], color="red", linewidth=2)
    axs[i//rows, i%cols].set_title(raw.info['ch_names'][32+i])


plt.tight_layout()
plt.show()



rows = 4
cols = 4

fig, axs = plt.subplots(rows, cols)

for i in range(rows*cols):
    ch_interval = raw.get_data(return_times=True)[0][48+i,intrvl_st:intrvl_end]
    times_interval = raw.get_data(return_times=True)[1][intrvl_st:intrvl_end]

    cwtm = signal.cwt(ch_interval, signal.morlet2, widths, w=w)
    axs[i//rows, i%cols].pcolormesh(times_interval, freq, stats.zscore(np.abs(cwtm)), cmap='viridis', shading='gouraud',vmin=-0.5,vmax=0.5)
    #axs[i//8, i%8].set_clim(-0.5, 0.5)
    axs[i//rows, i%cols].plot([sz_time, sz_time], [1,100], color="red", linewidth=2)
    axs[i//rows, i%cols].set_title(raw.info['ch_names'][48+i])


plt.tight_layout()
plt.show()


rows = 4
cols = 4

fig, axs = plt.subplots(rows, cols)

for i in range(rows*cols):
    ch_interval = raw.get_data(return_times=True)[0][64+i,intrvl_st:intrvl_end]
    times_interval = raw.get_data(return_times=True)[1][intrvl_st:intrvl_end]

    cwtm = signal.cwt(ch_interval, signal.morlet2, widths, w=w)
    axs[i//rows, i%cols].pcolormesh(times_interval, freq, stats.zscore(np.abs(cwtm)), cmap='viridis', shading='gouraud',vmin=-0.5,vmax=0.5)
    #axs[i//8, i%8].set_clim(-0.5, 0.5)
    axs[i//rows, i%cols].plot([sz_time, sz_time], [1,100], color="red", linewidth=2)
    axs[i//rows, i%cols].set_title(raw.info['ch_names'][64+i])


plt.tight_layout()
plt.show()



print("DONE")



























#t, dt = np.linspace(0, 1, 200, retstep=True)
#fs = 1/dt
#w = 6.
#sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
#freq = np.linspace(1, fs/2, 100)
#widths = w*fs / (2*freq*np.pi)
#cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
#plt.figure()
#plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
#plt.show()


print("DONE")