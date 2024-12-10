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
    
class STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch = None):

        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Apply LSTM for temporal modeling
        x, _ = self.lstm(x.unsqueeze(0))  # Add batch dimension for LSTM
        x = x.squeeze(0)  # Remove batch dimension from LSTM output
        final_out = x[-1, :] # grab last hidden state of lstm

        # # Pool node-level features into graph-level features
        # x = global_mean_pool(x, batch)

        # Final classification layer
        return self.linear(final_out).unsqueeze(0)

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