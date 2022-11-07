# Read and plot time-frequency plots for SEEG seizure events

# STEPS:
# Read SEEG file 
# Plot Raw file
# Plot PSD
# Bandpass filter 
# Downsample
# plot psd again
# Identify seizure events
# plot -20 to +20 seconds of seizure event for all channels
# compute time-frequency features of the intervals for all channels


# Starting steps

# Read SEEG file 
# Plot Raw file
# Plot PSD
# Identify seizure events
# plot -20 to +20 seconds of seizure event for one channel
# compute time-frequency features of the intervals for the channel channels



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

raw = mne.io.read_raw_edf(PATH_RNS, preload=True)
raw.notch_filter(line_noise_freqs)

raw.resample(500)


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

rows = 5
cols = 5

fig, axs = plt.subplots(rows, cols)

for i in range(rows*cols):
    ch_interval = raw.get_data(return_times=True)[0][i,intrvl_st:intrvl_end]
    times_interval = raw.get_data(return_times=True)[1][intrvl_st:intrvl_end]

    f, t, Sxx = signal.spectrogram(ch_interval, raw.info['sfreq'], nperseg= 500, noverlap=450)
    axs[i//rows, i%cols].pcolormesh(t, f, stats.zscore(Sxx), shading='gouraud',vmin=-0.5,vmax=0.5)
    #axs[i//8, i%8].set_clim(-0.5, 0.5)
    axs[i//rows, i%cols].plot([20,20], [6,250], color="red", linewidth=2)
    axs[i//rows, i%cols].set_title(raw.info['ch_names'][i])


plt.show()



plt.pcolormesh(t, f, stats.zscore(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.clim(-0.5, 0.5)
plt.plot([20,20], [6,250], color="red", linewidth=3)
plt.show()



ch1_interval = raw.get_data(return_times=True)[0][0,intrvl_st:intrvl_end]
times_interval = raw.get_data(return_times=True)[1][intrvl_st:intrvl_end]

f, t, Sxx = signal.spectrogram(ch1_interval, raw.info['sfreq'], nperseg= 500, noverlap=450)
plt.pcolormesh(t, f[5:], stats.zscore(Sxx[5:]), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.clim(-0.5, 0.5)
plt.plot([20,20], [6,250], color="red", linewidth=3)
plt.show()





print("DONE")