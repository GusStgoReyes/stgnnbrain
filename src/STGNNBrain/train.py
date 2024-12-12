from models import (MLP,
                    SimpleLSTM,
                    StaticGCN,
                    CNNGCN,
                    TemporalGCN,
                    TemporalGAT,
                    TimeStaticGCN,
                    TimeStaticGCNGAT)

from data import (load_connectivity_data, 
                  load_timeseries_data, 
                  load_labels,
                  ConnectomeDataset, 
                  ConnectomeDataset_TimeNodes)

from fitting import train, evaluate

from config import Config
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch_geometric
from torch_geometric.data import Data, Dataset

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############ SET UP PARAMETERS ############
num_nodes = 116
num_timepoints = 518
num_timepoints_graph2 = 100
hidden_channels = 64
hidden_channels_time = 32
out_channels = 2  # binary classification
window_size = 16
stride = 2
dilation = 2
heads = 3

# Set up parameters for training
patience = 15
min_delta = 0.001
n_splits = 5
num_epochs = 100

def main():
    # Create folders
    os.makedirs('./output/model', exist_ok=True)
    os.makedirs('./output/splits', exist_ok=True)
    os.makedirs('./output/results', exist_ok=True)

    ########## LOAD DATA AND GRAPHS ##########
    config = Config()
    user_ID = config.current_user_ID
    data_pth = config.data_pth[user_ID]

    # Load labels
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
    dataset = ConnectomeDataset(connectivity_matrices, timeseries_matrices, labels)
    dataset_time = ConnectomeDataset_TimeNodes(timeseries_matrices, labels, timepoints = 100)
    
   ########## DEFINE MODELS TO TEST ##########
    models_roi_graph = {
        'MLP': MLP(num_nodes, [hidden_channels], out_channels),
        'LSTM': SimpleLSTM(num_nodes, hidden_channels, out_channels),
        'StaticGCN' : StaticGCN(hidden_channels, hidden_channels, out_channels, num_nodes = num_nodes),
        'CNNGCN': CNNGCN(hidden_channels, out_channels, num_timepoints = num_timepoints, window_size=window_size, stride = stride, dilation = dilation, less_layers = False),
        'CNNGCN_lessLayers': CNNGCN(hidden_channels, out_channels, num_timepoints = num_timepoints, window_size=window_size, stride = stride, dilation = dilation, less_layers = True),
        'LSTMGCN': TemporalGCN(hidden_channels_time, out_channels, less_layers = False, temporal_layer = 'LSTM'),
        'LSTMGCN_lessLayers': TemporalGCN(hidden_channels_time, out_channels, less_layers = True, temporal_layer = 'LSTM'),
        'RNNGCN' : TemporalGCN(hidden_channels_time, out_channels, less_layers = False, temporal_layer = 'RNN'),
        'RNNGCN_lessLayers' : TemporalGCN(hidden_channels_time, out_channels, less_layers = True, temporal_layer = 'RNN'),
        'LSTMGAT' : TemporalGAT(hidden_channels_time, out_channels, heads = heads, less_layers = False, temporal_layer = 'LSTM'),
        'LSTMGAT_lessLayers' : TemporalGAT(hidden_channels_time, out_channels, heads = heads, less_layers = True, temporal_layer = 'LSTM'),
        'RNNGAT' : TemporalGAT(hidden_channels_time, out_channels, heads = heads, less_layers = False, temporal_layer = 'RNN'),
        'RNNGAT_lessLayers' : TemporalGAT(hidden_channels_time, out_channels, heads = heads, less_layers = True, temporal_layer = 'RNN')
    }

    models_time_graph = {
        'MLPTime': MLP(num_timepoints_graph2, [hidden_channels], out_channels),
        'TimeStaticGCNGAT' : TimeStaticGCNGAT(num_nodes, hidden_channels, out_channels, heads = heads),
        'TimeStaticGCN' : TimeStaticGCN(num_nodes, hidden_channels, out_channels),
    }

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)

    ############## TRAINING AND SAVING RESULTS ##############
    for model_group, graph_type in zip([models_roi_graph, models_time_graph], [dataset, dataset_time]):
        for model_name, model in model_group.items():
            print(f'Training {model_name}...')

            # Create a DataFrame to store train/test split information
            split_data = []
            results = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(graph_type, [data.y for data in graph_type])):
                print(f'Fold {fold + 1}/{n_splits}')

                # Add fold information to the split_data list
                for idx in train_idx:
                    split_data.append({'user_indx': idx, 'fold': fold + 1, 'split': 'train'})
                for idx in val_idx:
                    split_data.append({'user_indx': idx, 'fold': fold + 1, 'split': 'test'})

                # Split data
                train_loader = torch_geometric.loader.DataLoader(graph_type[train_idx], batch_size=1, shuffle=True)
                val_loader = torch_geometric.loader.DataLoader(graph_type[val_idx], batch_size=1, shuffle=False)

                less_layers = len(model_name.split("_")) == 2
                if less_layers:
                    model_name_ = model_name.split("_")[0]
                else:
                    model_name_ = model_name
                print(model_name, f"Less Layers = {less_layers}")
                # Reset model for each fold
                if model_name_ == 'MLP':
                    model = models_roi_graph[model_name].__class__(num_nodes, [hidden_channels], out_channels)
                elif model_name_ == 'LSTM':
                    model = models_roi_graph[model_name].__class__(num_nodes, hidden_channels, out_channels)
                elif model_name_ == 'StaticGCN':
                    model = models_roi_graph[model_name].__class__(hidden_channels, hidden_channels, out_channels, num_nodes = num_nodes)
                elif model_name_ == 'CNNGCN':
                    model = models_roi_graph[model_name].__class__(hidden_channels, out_channels, num_timepoints = num_timepoints, window_size=window_size, stride = stride, dilation = dilation, less_layers = less_layers)
                elif model_name_ == 'RNNGCN' or model_name == 'LSTMGCN':
                    temporal_layer = 'RNN' if model_name_ == 'RNNGCN' else 'LSTM'
                    model = models_roi_graph[model_name].__class__(hidden_channels_time, out_channels, less_layers = less_layers, temporal_layer = temporal_layer)
                elif model_name_ == 'RNNGAT' or model_name == 'LSTMGAT':
                    temporal_layer = 'RNN' if model_name_ == 'RNNGAT' else 'LSTM'
                    model = models_roi_graph[model_name].__class__(hidden_channels_time, out_channels, heads = heads, less_layers = less_layers, temporal_layer = temporal_layer)
                elif model_name_ == 'TimeStaticGCNGAT':
                    model = models_time_graph[model_name].__class__(num_nodes, hidden_channels, out_channels, heads = heads)
                elif model_name_ == 'TimeStaticGCN':
                    model = models_time_graph[model_name].__class__(num_nodes, hidden_channels, out_channels)
                elif model_name_ == 'MLPTime':
                    model = models_time_graph[model_name].__class__(num_timepoints_graph2, [hidden_channels], out_channels)
                else:
                    print("MODEL NOT INCLUDED IN THE TRAINING")

                if torch.cuda.is_available():
                    model.cuda()

                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()

                # Train model
                train_loss = train(model, train_loader, criterion, optimizer, device,
                                num_epochs=num_epochs, min_delta = min_delta, patience = patience)

                # Evaluate model
                val_loss, metrics = evaluate(model_name, model, val_loader, criterion, device)
                for metric_name, values in metrics.items():
                    results.append({'metric_name' : metric_name, 'fold' : fold + 1, 'value' : np.mean(values)})

                print(f'Training Loss: {round(train_loss, 3)} | Validation Loss: {round(val_loss, 3)}')
                for metric_name, values in metrics.items():
                    print(f'{metric_name}: {np.mean(values)}')

                # Save the model
                torch.save(model.state_dict(), f'./output/model/{model_name}_fold{fold + 1}.pth')

            # Save train/test split data to CSV
            split_df = pd.DataFrame(split_data)
            split_df.to_csv(f'./output/splits/{model_name}_splits.csv', index=False)
            print(f'Saved train/test split for {model_name} to CSV.')

            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'./output/results/{model_name}_results.csv', index=False)
            print(f'Saved results for {model_name} to CSV.')

if __name__ == '__main__':
    main()