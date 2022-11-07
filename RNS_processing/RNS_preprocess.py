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
#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS0427\iEEG\PIT-RNS0427_PE20181218-1_EOF_SZ-VK.EDF"


#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1090\iEEG\PIT-RNS1090_PE20170607-1_EOF_SZ-NZ.EDF"  # 
PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1090\iEEG\PIT-RNS1090_PE20180627-1_EOF_SZ-VK.EDF"  # 
#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1090\iEEG\PIT-RNS1090_PE20190806-1_EOF_SZ-NZ.EDF"

#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1438\iEEG\PIT-RNS1438_PE20190723-1_EOF_SZ-NZ.EDF"
#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1438\iEEG\PIT-RNS1438_PE20190409-1_EOF_SZ-VK.EDF"

raw = mne.io.read_raw_edf(PATH_RNS)
#all_intervals, seizure_interval_start, non_sz_intervals, sz_intervals = get_intervals(raw)


# Artefact mask
clean_y, masks, clean_times = get_stim_clip_cleaned(raw.get_data(), raw.get_data(return_times=True)[1], raw.info['sfreq'])


# Create bandpass filter
l_freq = 0.5 #Hz
h_freq = 60  #Hz
#filt_params = mne.filter.create_filter(clean_y[0].reshape(1, len(clean_y[0])), raw.info['sfreq'], l_freq, h_freq, method='iir')


# Filter Channel 1
filtered_y1 = mne.filter.filter_data(clean_y[0].reshape(1, len(clean_y[0])), raw.info['sfreq'], l_freq, h_freq, method='iir', iir_params=None, copy=True)











ch_names = ['channel1']
ch_types = ['eeg']

info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=250.0)
cleaned_raw = mne.io.RawArray(clean_y[0].reshape(1, len(clean_y[0])), info)

raw.plot()
raw.plot_psd(picks='channel1')

cleaned_raw.plot()
cleaned_raw.plot_psd()




print("DONE")

