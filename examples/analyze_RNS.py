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
from scipy stats 


# init analyzer
feature_reader = nm_analysis.Feature_Reader(
    feature_dir=os.getcwd(), feature_file="TestRNS"
)

print(feature_reader.feature_arr)

df = feature_reader.feature_arr

df_zscore = (df - df.mean())/df.std()

plt.imshow(df_zscore.T, aspect="auto")
plt.clim(-1, 1)
plt.yticks(np.arange(df_zscore.columns.shape[0]), df_zscore.columns)
plt.colorbar()
plt.show()

nm_plots.plot_corr_matrix(
        feature = df,
        ch_name= None,
        OUT_PATH= None,
        feature_names=df.columns,
        feature_file=feature_reader.feature_file,
        show_plot=True,
)


plt.figure(figsize=(7,4), dpi=300)
plt.plot(df_zscore["channel1_fft_high beta"], linewidth=0.3)