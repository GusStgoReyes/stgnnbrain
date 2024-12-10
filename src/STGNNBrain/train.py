import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import STGCN, STChebNet, STGAT, STSGConv, MLP
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
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def load_connectivity_data(subjects, path):
    conn_matrices = {}
    for subj_id in subjects:
      file_name = f'{subj_id}.txt'
      if os.path.exists(os.path.join(path, file_name)):
        matrix = np.loadtxt(os.path.join(path, file_name), dtype='str', delimiter=',')[1:, 1:]
        matrix = matrix.astype(np.float32)
        conn_matrices[subj_id] = torch.FloatTensor(matrix)
      else:
        raise ValueError(f"Missing connectivity data for subject {subj_id}")

    return conn_matrices

# Load corresponding timeseries (node features)
def load_timeseries_data(subjects, path):
    timeseries_matrices = {}

    for subj_id in subjects:
        file_name = f'{subj_id}.txt'
        if os.path.exists(os.path.join(path, file_name)):
            matrix = np.loadtxt(os.path.join(path, file_name), dtype='str', delimiter=',')[1:, 1:]
            matrix = matrix.astype(np.float32)
            timeseries_matrices[subj_id] = torch.FloatTensor(matrix)
        else:
            raise ValueError(f"Missing timeseries data for subject {subj_id}")
    return timeseries_matrices

def load_labels(path):
    labels = {}

    labels_df = pd.read_csv(path)
    for i, row in labels_df.iterrows():
        subject_id = row['subcode']
        label = row['caffeinated']
        labels[subject_id] = label

    return labels

# Define data loader as ConnectomeDataset
class ConnectomeDataset(Dataset):
  def __init__(self, connectivity_matrices, timeseries_matrices, labels, corr_threshold = 0.1):
      super().__init__()
      self.connectivity_matrices = connectivity_matrices
      self.timeseries_matrices = timeseries_matrices
      self.subjects = list(labels.keys())
      self.labels = labels
      self.corr_threshold = corr_threshold 

      self.subjects.sort()

  def len(self):
      return len(self.subjects)

  def get(self, idx):
      subject_id = self.subjects[idx]
      s_id = int(subject_id.split('sub')[1].split('.')[0])
      # Convert adjacency matrix to edge_index and edge_attr
      adj_matrix = torch.abs(self.connectivity_matrices[subject_id])
      edge_index = (adj_matrix > self.corr_threshold).nonzero().t()
      edge_attr = adj_matrix[edge_index[0], edge_index[1]].unsqueeze(1)

      # Node features from timeseries
      x = self.timeseries_matrices[subject_id]
      y = self.labels[subject_id]
      
      # Create PyG Data object
      data = Data(x=x,
                  edge_index=edge_index,
                  edge_attr=edge_attr,
                  y=y,
                  s_id=torch.LongTensor([s_id])
                  )

      return data
  
def train(model_name, model, train_loader, criterion, optimizer, num_epochs=100, min_delta = 0.001, patience = 10):
    best_train_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            # Move batch to the device
            batch = batch.to(device)
            
            # Forward pass
            if model_name == 'Baseline LSTM':
              output = model(batch.x)
            else:
              output = model(batch.x, batch.edge_index, batch.edge_attr)

            # Compute loss
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Early stopping check
        if avg_train_loss < best_train_loss - min_delta:
            best_train_loss = avg_train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping...')
                break

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {round(avg_train_loss, 3)}')

    return avg_train_loss

def evaluate(model_name, model, val_loader, criterion):
    # Switch model to evaluation mode
    model.eval()

    # Placeholders for metrics
    all_true_labels = []
    all_pred_probs = []
    all_pred_labels = []
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            if model_name == 'Baseline LSTM':
              output = model(batch.x)
            else:
              output = model(batch.x, batch.edge_index, batch.edge_attr)

            val_loss = criterion(output, batch.y)
            val_losses.append(val_loss.item())

            # Predictions
            val_pred_probs = F.softmax(output, dim=1)
            val_pred_labels = torch.argmax(val_pred_probs, dim=1)

            # Collect outputs
            all_true_labels.append(batch.y.cpu().numpy())
            all_pred_probs.append(val_pred_probs.cpu().numpy())
            all_pred_labels.append(val_pred_labels.cpu().numpy())

    # Aggregate results
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)
    all_pred_labels = np.concatenate(all_pred_labels, axis=0)

    # Compute metrics
    metrics = calculate_metrics(all_true_labels, all_pred_labels, all_pred_probs)
    avg_val_loss = np.mean(val_losses)

    return avg_val_loss, metrics

def calculate_metrics(y_true, y_pred, y_prob):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob[:, 1])
        }

def main():
    # Create folders
    os.makedirs('./output/model', exist_ok=True)
    os.makedirs('./output/splits', exist_ok=True)
    os.makedirs('./output/results', exist_ok=True)

    # Data paths
    data_pth = '/content/drive/MyDrive/CS224W Project/data'

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

    # Initialize dataset
    dataset = ConnectomeDataset(connectivity_matrices, timeseries_matrices, labels)
    
    # Initialize dataset
    dataset = ConnectomeDataset(connectivity_matrices, timeseries_matrices, labels)

    # Initialize models
    num_nodes = connectivity_matrices[subject_ids[0]].shape[1]
    num_timepoints = timeseries_matrices[subject_ids[0]].shape[0]
    hidden_channels = 64
    hidden_channels_time = 8
    out_channels = 2  # binary classification
    window_size = 16
    stride = 3
    
    models = {
        # 'GCNCNN': GCNCNN(num_nodes, hidden_channels, out_channels, num_timepoints = num_timepoints, window_size=window_size, stride = stride),
        # 'StaticGCN' : StaticGCN(hidden_channels, hidden_channels, out_channels, num_nodes = num_nodes)
        # 'LSTM': SimpleTimeSeriesLSTM(num_nodes, hidden_channels, out_channels),
        # 'MLP': MLP(num_nodes, [hidden_channels], out_channels),
        # 'STGCN': STGCN(hidden_channels_time, out_channels),
    }

    # Create dictionary to store results
    results = {model_name: {'accuracy' : [], 'precision' : [], 'recall' : [], 'f1' : [], 'auc' : []} for model_name in models.keys()}

    # Define early stopping parameters
    patience = 10
    min_delta = 0.001
    n_splits = 5
    num_epochs = 100

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)

    # K-fold cross validation loop
    for model_name, model in models.items():
        print(f'Training {model_name}...')

        # Create a DataFrame to store train/test split information
        split_data = []
        results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, [data.y for data in dataset])):
            print(f'Fold {fold + 1}/{n_splits}')

            # Add fold information to the split_data list
            for idx in train_idx:
                split_data.append({'user_indx': idx, 'fold': fold + 1, 'split': 'train'})
            for idx in val_idx:
                split_data.append({'user_indx': idx, 'fold': fold + 1, 'split': 'test'})

            # Split data
            train_loader = torch_geometric.loader.DataLoader(dataset[train_idx], batch_size=1, shuffle=True)
            val_loader = torch_geometric.loader.DataLoader(dataset[val_idx], batch_size=1, shuffle=False)

            # Reset model for each fold
            if model_name == 'MLP':
                model = models[model_name].__class__(num_nodes, [hidden_channels], out_channels)
            elif model_name == 'LSTM': 
                model = models[model_name].__class__(num_nodes, hidden_channels, out_channels)
            elif model_name == 'STGCN':
                model = models[model_name].__class__(hidden_channels_time, out_channels)
            elif model_name == 'GCNCNN':
                model = models[model_name].__class__(num_nodes, hidden_channels, out_channels, num_timepoints = num_timepoints, window_size=window_size, stride = stride)
            elif model_name == 'StaticGCN':
                model = models[model_name].__class__(hidden_channels, hidden_channels, out_channels, num_nodes = num_nodes)
            else:
                model = models[model_name].__class__(num_nodes, hidden_channels, out_channels)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Train model
            train_loss = train(model_name, model, train_loader, criterion, optimizer, 
                            num_epochs=num_epochs, min_delta = min_delta, patience = patience)

            # Evaluate model
            val_loss, metrics = evaluate(model_name, model, val_loader, criterion)
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