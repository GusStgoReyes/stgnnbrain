from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import os
import numpy as np
import pandas as pd

# Change paths!
# The AA116 atlas can be found in this link: https://github.com/brainspaces/aal116
atlas_filename = "/scratch/99999/gustxsr/aa116_atlas/aal116MNI.nii.gz"
fmri_filenames = os.listdir("/scratch/99999/gustxsr/myconnectome_rest")
fmri_file_pth = "/scratch/99999/gustxsr/myconnectome_rest"
save_timeseries = "/scratch/99999/gustxsr/gnn_cs224w/timeseries"
save_connectivity = "/scratch/99999/gustxsr/gnn_cs224w/connectivity"
confounds_pth = "/scratch/99999/gustxsr/myconnectome_subsetted_confounds"

masker = NiftiLabelsMasker(
    labels_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)

for fmri_filename in fmri_filenames:
    print(f"Currently on {fmri_filename[:14]}")
    if os.path.exists(os.path.join(save_timeseries, fmri_filename[:14] + ".txt")) or os.path.exists(os.path.join(save_connectivity, fmri_filename[:14] + ".txt")):
        print("doing again")
    confounds_df = pd.read_csv(os.path.join(confounds_pth, fmri_filename[:31] + "desc-confounds_timeseries.tsv"), sep="\t")[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]]
    time_series = masker.fit_transform(os.path.join(fmri_file_pth, fmri_filename), confounds = confounds_df)
    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    pd.DataFrame(correlation_matrix).to_csv(os.path.join(save_connectivity, fmri_filename[:14] + ".txt"))
    pd.DataFrame(time_series).to_csv(os.path.join(save_timeseries, fmri_filename[:14] + ".txt"))