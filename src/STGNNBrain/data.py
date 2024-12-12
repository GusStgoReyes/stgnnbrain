import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset

def load_connectivity_data(subject_ids, data_path):
    """
    Load connectivity matrices for given subjects from the specified path.

    Inputs:
      subject_ids (list): List of subject IDs.
      data_path (str): Path to the directory containing connectivity matrices.

    Outputs:
      conn_matrices (dict): Dictionary of connectivity matrices with subject IDs as keys.
    """
    conn_matrices = {}
    for subject_id in subject_ids:
        file_name = f'{subject_id}.txt'
        file_path = os.path.join(data_path, file_name)
        
        if os.path.exists(file_path):
            # Load the connectivity matrix, skipping the first row and column
            matrix = np.loadtxt(file_path, dtype='str', delimiter=',')[1:, 1:]
            # Convert the matrix to float32
            matrix = matrix.astype(np.float32)
            # Convert the matrix to a PyTorch tensor and store it in the dictionary
            conn_matrices[subject_id] = torch.FloatTensor(matrix)
        else:
            raise ValueError(f"Missing connectivity data for subject {subject_id}")

    return conn_matrices

def load_timeseries_data(subject_ids, data_path):
    """
    Load time series data for given subjects from the specified path.

    Inputs:
      subject_ids (list): List of subject IDs.
      data_path (str): Path to the directory containing time series data.

    Outputs:
      timeseries_data (dict): Dictionary of time series data with subject IDs as keys.
    """
    timeseries_data = {}
    for subject_id in subject_ids:
        file_name = f'{subject_id}.txt'
        file_path = os.path.join(data_path, file_name)
        
        if os.path.exists(file_path):
            # Load the time series data
            timeseries = np.loadtxt(file_path, dtype='str', delimiter=',')[1:, 1:]
            # Convert the time series data to float32
            timeseries = timeseries.astype(np.float32)
            # Convert the time series data to a PyTorch tensor and store it in the dictionary
            timeseries_data[subject_id] = torch.FloatTensor(timeseries)
        else:
            raise ValueError(f"Missing time series data for subject {subject_id}")

    return timeseries_data

def load_labels(labels_path):
    """
    Load labels from a CSV file.

    Inputs:
      labels_path (str): Path to the labels.csv file.

    Outputs:
      labels (dict): Dictionary of labels with subject IDs as keys.
    """
    labels = {}

    # Read the CSV file into a DataFrame
    labels_df = pd.read_csv(labels_path)
    # Iterate over each row in the DataFrame
    for _, row in labels_df.iterrows():
        subject_id = row['subcode']
        label = row['caffeinated']
        # Store the label in the dictionary with the subject ID as the key
        labels[subject_id] = label

    return labels

class ConnectomeDataset(Dataset):
    """
    For each subject, creates a graph where the nodes are the Brain ROIs and the features are the time points.
    The connectivity matrix is the correlation coefficient between the time
    series data of two Brain ROIs. We only include the edges with absolute correlation
    greater than 0.1. Furthermore, we keep the edge attributes as the absolute value of the
    correlation coefficient.
    """
    def __init__(self, connectivity_matrices, timeseries_matrices, labels, corr_threshold=0.1):
        super().__init__()
        self.connectivity_matrices = connectivity_matrices
        self.timeseries_matrices = timeseries_matrices
        self.subject_ids = list(labels.keys())
        self.labels = labels
        self.corr_threshold = corr_threshold

        # Sort the subject IDs for consistent ordering
        self.subject_ids.sort()

    def len(self):
        """
        Return the number of subjects in the dataset.
        """
        return len(self.subject_ids)

    def get(self, idx):
        """
        Get the data for a specific subject by index.

        Inputs:
          idx (int): Index of the subject.

        Outputs:
          data (Data): PyTorch Geometric Data object containing the graph and label.
        """
        subject_id = self.subject_ids[idx]
        # Convert adjacency matrix to edge_index and edge_attr
        adj_matrix = torch.abs(self.connectivity_matrices[subject_id])
        edge_index = (adj_matrix > self.corr_threshold).nonzero().t()
        edge_attr = adj_matrix[edge_index[0], edge_index[1]].unsqueeze(1)

        # Create a PyTorch Geometric Data object
        data = Data(x=self.timeseries_matrices[subject_id].T, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr)
        data.y = torch.tensor([self.labels[subject_id]], dtype=torch.long)

        return data
  
class ConnectomeDataset_TimeNodes(Dataset):
    """
        For each subject, creates a graph where the nodes are the time points and the features are the value at the ROIs.
        The connectivity matrix is the correlation coefficient between the ROI values between two different time points.
        We only include the edges with absolute correlation greater than 0.1. Furthermore, we keep the edge attributes as the
        absolute value of the correlation coefficient.
    """
    def __init__(self, timeseries_matrices, labels, timepoints=100, corr_threshold=0.1):
        super().__init__()
        self.timepoints = timepoints
        self.timeseries_matrices = timeseries_matrices
        self.subject_ids = list(labels.keys())
        self.labels = labels
        self.corr_threshold = corr_threshold

        # Sort the subject IDs for consistent ordering
        self.subject_ids.sort()

    def get_connectivity_matrix(self, timeseries):
        """
        Compute the connectivity matrix for the given time series data.

        Inputs:
          timeseries (Tensor): Time series data for a subject.

        Outputs:
          corr_matrix (Tensor): Correlation matrix of the time series data.
        """
        # timeseries will be a num_nodes x timepoints array
        # create a timepoints x timepoints array of correlation coefficient
        corr_matrix = torch.corrcoef(timeseries)
        return corr_matrix

    def len(self):
        """
        Return the number of subjects in the dataset.
        """
        return len(self.subject_ids)

    def get(self, idx):
        """
        Get the data for a specific subject by index.

        Inputs:
          idx (int): Index of the subject.

        Outputs:
          data (Data): PyTorch Geometric Data object containing the graph and label.
        """
        subject_id = self.subject_ids[idx]
        timeseries = self.timeseries_matrices[subject_id][:self.timepoints, :]
        # Compute the connectivity matrix
        adj_matrix = torch.abs(self.get_connectivity_matrix(timeseries))
        edge_index = (adj_matrix > self.corr_threshold).nonzero().t()
        edge_attr = adj_matrix[edge_index[0], edge_index[1]].unsqueeze(1)

        # Create a PyTorch Geometric Data object
        data = Data(x=timeseries, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr)
        data.y = torch.tensor([self.labels[subject_id]], dtype=torch.long)

        return data