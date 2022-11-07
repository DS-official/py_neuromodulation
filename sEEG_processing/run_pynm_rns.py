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

###################
def get_intervals(raw):
    all_intervals = []
    seizure_interval_start = []
    start_inter = 0
    end_inter = 0
    for i in range(len(raw.annotations)):
        if(raw.annotations[i]['description'] == 'eof'):
            end_inter = raw.annotations[i]['onset']
            all_intervals.append([start_inter, end_inter])
            start_inter = end_inter
        elif(raw.annotations[i]['description'] == 'sz_on'):
            seizure_interval_start.append(start_inter)

    if(all_intervals[-1][1] != raw.times[-1]):
        all_intervals.append([start_inter, raw.times[-1]])


    # From all_intervals, get intervals corresponding to seizures, and intervals with no seizure activity
    non_sz_intervals = []
    sz_intervals = []
    for interval in all_intervals:
        if(interval[0] not in seizure_interval_start):
            non_sz_intervals.append(interval)
        else:
            sz_intervals.append(interval)
    # sz_intervals and non_sz_intervals hold this

    return all_intervals, seizure_interval_start, non_sz_intervals, sz_intervals

####

def line_length(data_interval):
    """ Returns line-length metric for given data_interval
    
    Line-length metric is computed by LL_x(t) = (1/N-1)\sum_i[X(i+1) - X(i)]"""

    LL_sum = 0
    for i in range(1,len(data_interval)):
        LL_sum += abs(data_interval[i] - data_interval[i-1]) 
    
    LL = LL_sum/(len(data_interval)-1)
    return LL


def get_LL(data_chunk, time_chunk, sfreq, time_len):
    """ Returns an array of line-length calculations for given data chunk and time_len"""

    LL_arr = []
    time_arr = []
    # if len(time_chunk) != len(data_chunk), throw exception
    idx = 0
    interval_len = round(sfreq*time_len)
    while(idx+interval_len < len(data_chunk)):
        LL_arr.append(line_length(data_chunk[idx:idx+interval_len]))
        time_arr.append(time_chunk[round(idx + (interval_len/2) )])
        idx = idx + interval_len

    # if (idx < len(data_chunk)-1):
    #    LL_arr.append(line_length(data_chunk[idx:]))

    return LL_arr,time_arr

###################


PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1703\sEEG\EP1150_2af97812-453f-4210-92a2-c14afda0fcf0_SZ1.edf"

raw = mne.io.read_raw_edf(PATH_RNS)

#raw.plot_psd(fmax = 500)

sfreq = raw.info['sfreq']
#seeg_data_and_times = raw.get_data(return_times=True)

# number of time samples is raw.n_times
tot_secs = raw.n_times/sfreq


times = raw.get_data(return_times=True)[1]
seeg_data_all_ch = raw.get_data(return_times=True)[0]

#plt.plot(times,seeg_data_all_ch[1])
for i in range(len(seeg_data_all_ch)):
    seeg_data_all_ch[i] = seeg_data_all_ch[i] - seeg_data_all_ch[i].mean()


# Notch filter the data at 60 Hz


# Filter EEG

print("END OF FILE")
