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
from scipy import stats 


# init analyzer
feature_reader = nm_analysis.Feature_Reader(
    feature_dir=r"Z:\Users\DY548\RNS_processing\RNS_features\PIT-RNS0427", feature_file="TestRNS"
)

print(feature_reader.feature_arr)

df = feature_reader.feature_arr

df_zscore = (df - df.mean())/df.std()

plt.imshow(df_zscore.T, aspect="auto")
plt.clim(-1, 1)
plt.yticks(np.arange(df_zscore.columns.shape[0]), df_zscore.columns)
plt.colorbar()
plt.title("PIT-RNS0427_PE20181218-1 Features All channels")
plt.show()

nm_plots.plot_corr_matrix(
        feature = df,
        ch_name= None,
        OUT_PATH= None,
        feature_names=df.columns,
        feature_file=feature_reader.feature_file,
        show_plot=True,
        save_plot=False,
)
plt.show()

#plt.figure(figsize=(7,4), dpi=300)
#plt.plot(df_zscore["channel1_fft_high beta"], linewidth=0.3)
