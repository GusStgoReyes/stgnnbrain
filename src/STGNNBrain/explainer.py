from models import TimeStaticGCN
from config import Config
from data import (load_connectivity_data, 
                  load_timeseries_data, 
                  load_labels,
                  ConnectomeDataset_TimeNodes)
import os
import numpy as np
import pandas as pd
import nibabel as nib
from torch_geometric.explain import Explainer, GNNExplainer
import torch

if __name__ == "__main__":
    # Paths
    config = Config()
    user_ID = config.current_user_ID
    data_pth = config.data_pth[user_ID]

    split_pth = './output/splits'
    model_pth = './output/model'
    model_name = 'TimeStaticGCN'
    
    # Hyperparameters
    n_splits = 5
    num_nodes = 116
    hidden_channels = 64
    out_channels = 2

    # Load the splits used per fold of training/eval
    split_df = pd.read_csv(os.path.join(split_pth, f'{model_name}_splits.csv'))

    # Load dataset
    labels = load_labels(os.path.join(data_pth, 'labels.csv'))
    subject_ids = list(labels.keys())
    print(f'Class distribution: {np.unique(list(labels.values()), return_counts=True)}')
    print('-' * 50)

    # Load connectivity matrices and subject IDs
    connectivity_matrices = load_connectivity_data(subject_ids, os.path.join(data_pth, 'connectivity_aa116'))
    print(f'Number of samples: {len(connectivity_matrices)}')
    print(f'Number of ROIs: {connectivity_matrices[list(connectivity_matrices)[0]].shape[0]}')
    print(connectivity_matrices.keys())

    # Load timeseries matrices
    timeseries_matrices = load_timeseries_data(subject_ids, os.path.join(data_pth, 'timeseries_aa116'))
    print(f'Number of timeseries matrices: {len(timeseries_matrices)}')

    # Initialize datasets
    dataset = ConnectomeDataset_TimeNodes(timeseries_matrices, labels, timepoints = 100)

    node_importances = []
    for fold in range(1, n_splits + 1):
        pth = os.path.join(model_pth, f'{model_name}_fold{fold}.pth')

        model = TimeStaticGCN(num_nodes, hidden_channels, out_channels)
        model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
        model.eval()
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='common_attributes',
            edge_mask_type=None,
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        print(f"Model '{model_name}' (fold {fold}) loaded successfully from {pth}")

        test_user_indxs = split_df.query("split == 'test' & fold == @fold")['user_indx'].to_list()
        
        # For each user on testing set, get the explanation of the node features of the model (brain ROIs)
        for user_indx in test_user_indxs:
            print(f'User {user_indx}')
            explanation = explainer(dataset[user_indx].x, dataset[user_indx].edge_index, edge_attr = dataset[user_indx].edge_attr)
            node_importances.append(explanation.node_mask.numpy()[0, :])
        
        # Average the importance of the Brain ROIs across all users
        node_importances = np.array(node_importances)
        node_importances /= node_importances.sum(axis=1, keepdims=True)
        mean_node_importances = node_importances.mean(axis=0)

        # Create brain image with the average importance of the Brain ROIs
        brain = nib.load(os.path.join(data_pth, "aal116MNI.nii.gz"))
        img_data = brain.get_fdata()

        # open /content/aal116NodeIndex.1D
        with open(os.path.join(data_pth, "aal116NodeIndex.1D"), 'r') as f:
            node_index = f.read().splitlines()
        
        new_img_data = np.zeros(img_data.shape)
        for n in range(116):
            new_img_data[img_data == int(node_index[n])] = mean_node_importances[n]

        # save into nii.gz
        new_brain = nib.Nifti1Image(new_img_data, brain.affine, brain.header)
        nib.save(new_brain, os.path.join(data_pth, "brain_importances.nii.gz"))



