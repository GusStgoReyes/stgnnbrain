import os
import numpy as np
import pandas as pd
from STGNNBrain.config import Config

def functional_connectivity(data, save_pth = None, method='pearson'):
    if method == "pearson":
        connectivity = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            for j in range(i, data.shape[1]):
                connectivity[i, j] = np.corrcoef(data[:, i], data[:, j])[0, 1]
                connectivity[j, i] = connectivity[i, j]
    if save_pth:
        pd.DataFrame(connectivity).to_csv(save_pth)

    return connectivity

if __name__ == "__main__":
    config = Config()
    user_ID = config.current_user_ID
    data_pth = config.data_pth[user_ID]

    # loop through all rsdata and save in connectivity pth
    for scan_ID in os.listdir(os.path.join(data_pth, 'timeseries')):
        print(scan_ID[:-4])
        data = pd.read_csv(os.path.join(data_pth, "timeseries", scan_ID), sep=" ", header=None).to_numpy()
        save_pth = os.path.join(data_pth, 'connectivity', scan_ID)
        functional_connectivity(data, save_pth)