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
import os


PATH_RNS = r"C:\Users\DY548\Documents\Python Scripts\trial\test data\PIT-RNS0427_PE20181120-1_EOF_SZ-NZ.EDF"

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
    out_path_root=os.getcwd(),
    folder_name="TestRNS",
)
