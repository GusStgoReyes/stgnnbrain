from config import Config
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import os
import numpy as np
import pandas as pd

config = Config()
user_ID = config.current_user_ID

# This script is used to create the parcelled rs-fMRI data for the ST-GNN model.
# The AA116 atlas can be found in this link: https://github.com/brainspaces/aal116
atlas_pth = config.atlas_pth[user_ID]
fmri_pth = config.fmri_pth[user_ID]
confounds_pth = config.confounds_pth[user_ID]
save_timeseries = config.save_timeseries[user_ID]
save_connectivity = config.save_connectivity[user_ID]

if __name__ == "__main__":
    if not os.path.exists(save_timeseries):
        os.makedirs(save_timeseries)
    if not os.path.exists(save_connectivity):
        os.makedirs(save_connectivity)

    # Create the masker for the AAL116 atlas
    masker = NiftiLabelsMasker(
        labels_img=atlas_pth,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
        verbose=5,
    )

    fmri_filenames = os.listdir(fmri_pth)
    for fmri_filename in fmri_filenames:
        print(f"Currently on {fmri_filename[:14]}")
        if os.path.exists(os.path.join(save_timeseries, fmri_filename[:14] + ".txt")) or os.path.exists(os.path.join(save_connectivity, fmri_filename[:14] + ".txt")):
            print("Already processed")
        
        # Load the confounds
        confounds_df = pd.read_csv(os.path.join(confounds_pth, fmri_filename[:31] + "desc-confounds_timeseries.tsv"), sep="\t")[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]]

        # Create the time series data, regressing confounds out
        time_series = masker.fit_transform(os.path.join(fmri_pth, fmri_filename), confounds = confounds_df)

        # Create the connectivity matrix
        correlation_measure = ConnectivityMeasure(
            kind="correlation",
            standardize="zscore_sample",
        )
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        pd.DataFrame(correlation_matrix).to_csv(os.path.join(save_connectivity, fmri_filename[:14] + ".txt"))
        pd.DataFrame(time_series).to_csv(os.path.join(save_timeseries, fmri_filename[:14] + ".txt"))