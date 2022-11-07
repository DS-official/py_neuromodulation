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
import mne_connectivity
import scipy
from scipy.fft import fft, fftfreq

from intervals import get_intervals
from remove_artefact import *

###################

# Load data
#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS0427\iEEG\PIT-RNS0427_PE20181218-1_EOF_SZ-NZ.EDF"  # First programming epoch

#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1090\iEEG\PIT-RNS1090_PE20170607-1_EOF_SZ-NZ.EDF"  # 
#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1090\iEEG\PIT-RNS1090_PE20180627-1_EOF_SZ-VK.EDF"  # 
PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1090\iEEG\PIT-RNS1090_PE20190806-1_EOF_SZ-NZ.EDF"

#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1438\iEEG\PIT-RNS1438_PE20190723-1_EOF_SZ-NZ.EDF"
#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1438\iEEG\PIT-RNS1438_PE20190409-1_EOF_SZ-VK.EDF"

raw = mne.io.read_raw_edf(PATH_RNS)
#all_intervals, seizure_interval_start, non_sz_intervals, sz_intervals = get_intervals(raw)

#raw.plot_psd(picks='channel3')

# Artefact mask
clean_y, masks, clean_times = get_stim_clip_cleaned(raw.get_data(), raw.get_data(return_times=True)[1], raw.info['sfreq'])

ch_names = ['channel1']
ch_types = ['eeg']

info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=250.0)
simulated_raw = mne.io.RawArray(clean_y[0].reshape(1, len(clean_y[0])), info)


raw.plot()
raw.plot_psd(picks='channel1')

simulated_raw.plot()
simulated_raw.plot_psd()



# Compute and plot fft for a channel using scipy
# from scipy.fft import fft, fftfreq
# y = raw.get_data()[2,:]
# Number of sample points
# N = y.shape[0]
# sample spacing
# T = 1.0 / raw.info['sfreq']
#x = raw.get_data(return_times=True)[0]
# yf = fft(y)
# xf = fftfreq(N, T)[1:N//2]
# import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(xf, 2.0/N * np.abs(yf[1:N//2]))
#plt.grid()
#plt.show()






"""
interval = 50000
# plot using matplotlib
y1 = raw.get_data()[0,0:interval]
time = raw.get_data(return_times=True)[1][0:interval]
plt.figure()
plt.plot(time,y1)
plt.title('Raw data for Channel 1')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# plot fft 
N = len(y1)
T = 1.0 / raw.info['sfreq']
yf = fft(y1)
xf = fftfreq(N, T)[1:N//2]
plt.figure()
plt.plot(xf, 2.0/N * np.abs(yf[1:N//2]))
plt.grid()
plt.title("FFT for Channel 1")
plt.xlabel("Frequencies (Hz)")
plt.ylabel("power?")
plt.show()

# Raw plot
plt.figure()
plt.plot(clean_times[0][0:interval],clean_y[0][0:interval])
plt.title('Cleaned Raw data for Channel 1')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# plot fft 
N = len(clean_y[0][0:interval])
T = 1.0 / raw.info['sfreq']
yf = fft(clean_y[0][0:interval])
xf = fftfreq(N, T)[1:N//2]
plt.figure()
plt.plot(xf, 2.0/N * np.abs(yf[1:N//2]))
plt.grid()
plt.title("FFT for Cleaned Channel 1")
plt.xlabel("Frequencies (Hz)")
plt.ylabel("power?")
plt.show()



# plot using matplotlib
y3 = raw.get_data()[2,0:interval]
time = raw.get_data(return_times=True)[1][0:interval]
plt.figure()
plt.plot(time,y1)
plt.title('Raw data for Channel 3')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# plot fft 
N = len(y3)
T = 1.0 / raw.info['sfreq']
yf = fft(y3)
xf = fftfreq(N, T)[1:N//2]
plt.figure()
plt.plot(xf, 2.0/N * np.abs(yf[1:N//2]))
plt.grid()
plt.title("FFT for Channel 3")
plt.xlabel("Frequencies (Hz)")
plt.ylabel("power?")
plt.show()

# Raw plot
plt.figure()
plt.plot(clean_times[2][0:interval],clean_y[2][0:interval])
plt.title('Cleaned Raw data for Channel 3')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# plot fft 
N = len(clean_y[2][0:interval])
T = 1.0 / raw.info['sfreq']
yf = fft(clean_y[2][0:interval])
xf = fftfreq(N, T)[1:N//2]
plt.figure()
plt.plot(xf, 2.0/N * np.abs(yf[1:N//2]))
plt.grid()
plt.title("FFT for Cleaned Channel 3")
plt.xlabel("Frequencies (Hz)")
plt.ylabel("power?")
plt.show()

"""
print("DONE")

