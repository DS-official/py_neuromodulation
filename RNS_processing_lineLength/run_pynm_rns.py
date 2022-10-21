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


# PATH_RNS = r"C:\Users\DY548\Documents\Python Scripts\trial\test data\PIT-RNS0427_PE20181120-1_EOF_SZ-NZ.EDF"

PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS0427\iEEG\PIT-RNS0427_PE20181218-1_EOF_SZ-NZ.EDF"
#PATH_RNS = r"Z:\RNS_DataBank\PITT\PIT-RNS1090\iEEG\PIT-RNS1090_PE20170530-1_EOF_SZ-NZ.EDF"

raw = mne.io.read_raw_edf(PATH_RNS)

nm_channels = nm_define_nmchannels.set_channels(
    ch_names=raw.ch_names,
    ch_types=["ecog" for i in range(4)],
    reference=None,
    bads=None,
    new_names="default",
    used_types=("ecog",),
    target_keywords=None,

)

# Get intervals from raw file, and seizure onset time points. 
all_intervals, seizure_interval_start, non_sz_intervals,sz_intervals = get_intervals(raw)

# intervals now holds time points corresponding to each recording.
# all_interval[i] holds the [start,end] of ith interval




# Extract and store data corresponding to non_sz_intervals
sfreq = raw.info['sfreq']
all_data = raw.get_data(return_times=True)[0]
all_times = raw.get_data(return_times=True)[1]
all_non_sz_data = all_data[:,round(sfreq*non_sz_intervals[0][0]):round(sfreq*non_sz_intervals[0][1])]
all_non_sz_times = all_times[round(sfreq*non_sz_intervals[0][0]):round(sfreq*non_sz_intervals[0][1])]
for i in range(1,len(non_sz_intervals)):
    all_non_sz_data = np.concatenate((all_non_sz_data,all_data[:,round(sfreq*non_sz_intervals[i][0]):round(sfreq*non_sz_intervals[i][1])]), axis=1)
    all_non_sz_times = np.concatenate((all_non_sz_times,all_times[round(sfreq*non_sz_intervals[i][0]):round(sfreq*non_sz_intervals[i][1])]))   
# all_non_sz_data is of shape 4 * len, holding 4 channels of data corresponding to non-sz_intervals
# all_non_sz_times is an array for the corresponding time points

splice = range(round(sfreq*non_sz_intervals[0][0]),round(sfreq*non_sz_intervals[0][1]))
#splice = range(len(all_non_sz_times))
non_sz_LL,non_sz_LL_time = get_LL(all_non_sz_data[0][splice], all_non_sz_times[splice], sfreq, 0.04)


""""
Plot Line length and raw data on top of each other for smaller windows. (subplot etc.)
Do this multiple for multiple recordings and channels 
"""
plt.figure()
plt.title("PIT-RNS0427_PE20181218-1")
plt.subplot(211)
plt.title("Non-seizure interval data, channel 0")
plt.plot(all_non_sz_times[splice],all_non_sz_data[0][splice], linewidth=0.5)
plt.ylabel("RNS voltage units(muV?)")
plt.xlabel("time(s)")
plt.subplot(212)
plt.title("Line-Length for non-seizure interval data, channel 0")
plt.plot(non_sz_LL_time, non_sz_LL, linewidth=0.5)
plt.ylabel("Line-length values")
plt.xlabel("time(s)")
plt.show()
"""
Separate seizure and non seizure recordings.
Add that as a feature

Calculate Line length, add as feature

Phase coherence stuff for connectivity?

"""

#count = 0
#for i in range(len(raw.annotations)):
#    if(raw.annotations[i]['description'] == 'eof'):
#        count += 1

stream = nm.Stream(
    settings=None,
    nm_channels=nm_channels,
    path_grids=None,
    verbose=True,
)


stream.settings["preprocessing"]["notch_filter"] = False
stream.settings["preprocessing"]["preprocessing_order"] = []
stream.settings["preprocessing"]["raw_normalization"] = False
stream.settings["preprocessing"]["raw_resampling"] = False
stream.settings["preprocessing"]["re_referencing"] = False
stream.settings["features"]["fft"] = True
stream.settings["features"]["raw_hjorth"] = False
stream.settings["features"]["return_raw"] = False
stream.settings["features"]["bandpass_filter"] = False
stream.settings["features"]["stft"] = False
stream.settings["features"]["sharpwave_analysis"] = False
stream.settings["features"]["coherence"] = False
stream.settings["features"]["fooof"] = False
stream.settings["features"]["nolds"] = False
stream.settings["features"]["bursts"] = False
stream.settings["postprocessing"]["feature_normalization"] = False
stream.settings["postprocessing"]["project_cortex"] = False
stream.settings["postprocessing"]["project_subcortex"] = False

stream.settings["frequency_ranges_hz"] = \
{
"theta": [
            4,
            8
        ],
"alpha": [
            8,
            12
        ],
"low beta": [
            13,
            20
        ],
"high beta": [
            20,
            35
        ]
}

stream.settings["fft_settings"]["log_transform"] = False
stream.settings["fft_settings"]["kalman_filter"] = False

stream.init_stream(
    sfreq=250,
    line_noise=60,
    coord_list=[],
    coord_names=[],
)


stream.run(
    data=raw.get_data(),
    out_path_root=r"Z:\Users\DY548\RNS_processing\RNS_features\PIT-RNS0427",
    folder_name="TestRNS",
)
