import torch
from torch_geometric.nn import GCNConv, ChebConv, GATConv, SGConv
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class SimpleTimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleTimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        lstm_out, _ = self.lstm(x.unsqueeze(0))  # Output for all time steps
        final_out = lstm_out[:, -1, :]  # Use the last time step's output
        return self.linear(final_out)  # Map to output size
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        # Define hidden layers
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList()
        in_features = input_size*input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        
        # Define the output layer
        self.output_layer = nn.Linear(in_features, output_size)
        
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the MLP model.
        
        Args:
            x (tensor): Input tensor.
            
        Returns:
            tensor: Output predictions.
        """
        matrix = torch.zeros(self.input_size, self.input_size)
        matrix[edge_index[0], edge_index[1]] = edge_attr[:, 0]

        x = matrix.flatten()
        for layer in self.hidden_layers:
            x = F.relu(layer(x))  # Apply ReLU activation after each hidden layer
        return self.output_layer(x).unsqueeze(0)  # No activation on the final layer (for raw output)

class StaticGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes = 116):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = torch.ones((self.num_nodes, self.in_channels))
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = global_mean_pool(x, None)
        return x
    
class STGCN(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(STGCN, self).__init__()
        self.hidden_channels = hidden_channels

        # Define an LSTM for each feature
        self.lstm_layer = nn.LSTM(1, hidden_channels, batch_first=True)

        # Graph Convolution layer
        self.conv1 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        x: Input feature matrix of shape [timepoints, num_features]
        edge_index: Edge indices for graph structure
        edge_attr: Edge weights
        """
        # Reshape input for batched LSTM processing
        x = x.transpose(0, 1).unsqueeze(-1)

        # Pass all features through the LSTM in a single batch
        lstm_out, _ = self.lstm_layer(x) 

        # Extract the last hidden state for each feature
        lstm_output_matrix = lstm_out[:, -1, :]

        # Apply GCN on the output of LSTMs
        gcn_output = self.conv1(lstm_output_matrix, edge_index, edge_weight=edge_attr)

        # Global mean
        gcn_output = global_mean_pool(gcn_output, None)

        return gcn_output

class STChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super(STChebNet, self).__init__()
        self.in_channels = in_channels
        self.conv1 = ChebConv(self.in_channels, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply ChebConv layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # GRU for temporal modeling
        x, _ = self.gru(x.unsqueeze(0))
        x = x.squeeze(0)
        # Handle the case where no batch is provided (e.g., single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Graph pooling
        x = global_mean_pool(x, batch)

        # Final linear layer for classification
        return self.linear(x)

    def process_edge_features(self, x, edge_index, edge_attr):
        # Example: Aggregate edge attributes to nodes
        aggregated_edge_attr = torch.zeros_like(x, device=x.device)
        for i in range(edge_index.size(1)):
            source, target = edge_index[:, i]
            aggregated_edge_attr[target] += edge_attr[i]

        # Concatenate aggregated edge_attr to node features
        return torch.cat([x, aggregated_edge_attr], dim=1)

class STGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(STGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads)
        self.temporal_conv = nn.Conv1d(hidden_channels * heads, hidden_channels, kernel_size=3, padding=1)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.temporal_conv(x.unsqueeze(0).transpose(1, 2)))
        return self.linear(x.squeeze(0).transpose(1, 2))

class STSGConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super(STSGConv, self).__init__()
        self.conv1 = SGConv(in_channels, hidden_channels, K)
        self.conv2 = SGConv(hidden_channels, hidden_channels, K)
        self.temporal_attn = nn.MultiheadAttention(hidden_channels, num_heads=4)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.unsqueeze(0)
        x, _ = self.temporal_attn(x, x, x)
        return self.linear(x.squeeze(0))
    
class GCNCNN(nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels, num_timepoints = 518, window_size=32, stride = 3):
        super(GCNCNN, self).__init__()
        
        # 1D Convolution to process time series data for each node
        self.conv1d_1 = nn.Conv1d(
                                in_channels=num_nodes,  # Number of time series (one per node)
                                out_channels=num_nodes,  # Maintain one series per node
                                kernel_size=window_size,  # Window size for the moving average
                                stride = stride,
                                groups=num_nodes  # Separate filters for each node
                            )
        
        # Hard coded, change, compute out dimension after first conv1d on time
        l_out = np.floor((num_timepoints - window_size)/stride + 1)
        in_channels = int(l_out)

        # One GCNConv layers
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Input tensor of shape (time_points, num_nodes).
            edge_index: Edge list of shape (2, num_edges).
            edge_attr: Edge weights of shape (num_edges,).
        """
        x = x.T.unsqueeze(0)  # Shape: (1, num_nodes, time_points)
        
        # Temporal summary
        x = F.relu(self.conv1d_1(x))
        x = x.squeeze(0)  # Shape: (num_nodes, time_points)

        # Apply spatial information
        x = self.gcn1(x, edge_index, edge_weight=edge_attr)

        x = global_mean_pool(x, None)
        
        return x