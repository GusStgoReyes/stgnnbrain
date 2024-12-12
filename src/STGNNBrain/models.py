import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

#############BASELINE MODELS################
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        # Define the input size
        self.input_size = input_size

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        in_features = input_size * input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size

        # Define the output layer
        self.output_layer = nn.Linear(in_features, output_size)

    def forward(self, node_features, edge_index, edge_attributes):
        """
        Forward pass for the MLP model.

        Inputs:
          node_features (Tensor): Node features.
          edge_index (Tensor): Edge indices.
          edge_attributes (Tensor): Edge attributes.

        Outputs:
          output (Tensor): Output of the MLP model.
        """
        device = node_features.device

        # Initialize an adjacency matrix with zeros
        adjacency_matrix = torch.zeros(self.input_size, self.input_size).to(device)
        # Fill the adjacency matrix with edge attributes
        adjacency_matrix[edge_index[0], edge_index[1]] = edge_attributes[:, 0]

        # Flatten the adjacency matrix
        x = adjacency_matrix.flatten()

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))  # Apply ReLU activation after each hidden layer

        # Pass through the output layer and unsqueeze the output
        output = self.output_layer(x).unsqueeze(0)  # No activation on the final layer (for raw output)

        return output

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, node_features, edge_index, edge_attributes):
        """
        Forward pass for the SimpleLSTM model.

        Inputs:
          node_features (Tensor): Time series data of shape (input_size, seq_len).
          edge_index (Tensor): Edge indices.
          edge_attributes (Tensor): Edge attributes.

        Outputs:
          output (Tensor): Output of the SimpleLSTM model.
        """
        x = node_features.T
        # Pass the time series data through the LSTM layer
        x, _ = self.lstm(x.unsqueeze(0))  # Output for all time steps

        # Use the last time step's output
        x = x[:, -1, :]
        x = F.relu(x)  # Apply ReLU activation

        # Pass through the output layer
        x = self.output_layer(x)

        return x
    
class StaticGCN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_nodes=116):
        super(StaticGCN, self).__init__()

        # Define the input channels and number of nodes
        self.input_channels = input_channels
        self.num_nodes = num_nodes

        # Define the first GCN layer
        self.conv1 = GCNConv(input_channels, hidden_channels)
        # Define the second GCN layer
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, node_features, edge_index, edge_attributes):
        """
        Forward pass for the StaticGCN model.

        Inputs:
          node_features (Tensor): Node features.
          edge_index (Tensor): Edge indices.
          edge_attributes (Tensor): Edge attributes.

        Outputs:
          output (Tensor): Output of the StaticGCN model.
        """
        device = node_features.device

        # Initialize node features with ones
        x = torch.ones((self.num_nodes, self.input_channels)).to(device)

        # Apply the first GCN layer with edge attributes
        x = self.conv1(x, edge_index, edge_weight=edge_attributes)
        x = F.relu(x) 

        # Apply the second GCN layer with edge attributes
        x = self.conv2(x, edge_index, edge_weight=edge_attributes)

        # Apply global mean pooling
        output = global_mean_pool(x, None)

        return output

#############GRAPH 1 MODELS################
class CNNGCN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_timepoints = 518, window_size=32, stride = 2, dilation = 2, less_layers = False):
        super(CNNGCN, self).__init__()
        self.less_layers = less_layers

        # 1D Convolution to process time series data for each node
        self.conv1d_1 = nn.Conv1d(
                                in_channels=1,  # Input features per time point
                                out_channels=1,  # Output features per time point
                                kernel_size=window_size,
                                stride = stride,
                                dilation = dilation,
                            )
        self.conv1d_2 = nn.Conv1d(
                                in_channels=1, # Input features per time point
                                out_channels=1, # Output features per time point
                                kernel_size=window_size,
                                stride = stride,
                                dilation = dilation,
                            )

        # Compute the output dimension of the Conv1d layers
        conv_output_dim = (num_timepoints - (window_size - 1) * dilation - 1) // stride + 1
        conv_output_dim = (conv_output_dim - (window_size - 1) * dilation - 1) // stride + 1
        in_channels = int(conv_output_dim)

        # Graph Convolution layer
        if self.less_layers:
            self.conv1 = GCNConv(in_channels, out_channels)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, node_features, edge_index, edge_attributes):
        """
        Forward pass for the CNNGCN model.

        Inputs:
          node_features (Tensor): Node features.
          edge_index (Tensor): Edge indices.
          edge_attributes (Tensor): Edge attributes.

        Outputs:
          output (Tensor): Output of the CNNGCN model.
        """
        x = node_features.unsqueeze(1)  # Shape: (num_nodes, 1, time_points)

        # Apply the first 1D convolution (temporal information)
        x = self.conv1d_1(x)
        x = F.relu(x)
        # Apply the second 1D convolution
        x = self.conv1d_2(x)
        x = F.relu(x)

        # Reshape the output of the 1D convolution
        x = x.squeeze(1)  # Shape: (num_nodes, time_points_new)
        
        # Apply GCNConv layers (spatial information)
        if self.less_layers:
           x = self.conv1(x, edge_index, edge_weight=edge_attributes)
        else:
            x = self.conv1(x, edge_index, edge_weight=edge_attributes)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_attributes)
        
        # Apply global mean pooling
        x = global_mean_pool(x, None)

        return x
    
class TemporalGCN(nn.Module):
    def __init__(self, hidden_channels, out_channels, less_layers = False, temporal_layer = 'LSTM'):
        super(TemporalGCN, self).__init__()
        self.less_layers = less_layers
        self.hidden_channels = hidden_channels
        
        if temporal_layer == 'LSTM':
            # LSTM layer to process time series data
            self.temporal_layer = nn.LSTM(input_size = 1, # 1 feature per time point
                                            hidden_size = hidden_channels, # Number of hidden units
                                            batch_first=True)
        elif temporal_layer == 'RNN':
            # RNN layer to process time series data
            self.temporal_layer = nn.RNN(input_size = 1, # 1 feature per time point
                                            hidden_size = hidden_channels, # Number of hidden units
                                            batch_first=True)

        # Graph Convolution layer
        if self.less_layers:
           self.conv1 = GCNConv(hidden_channels, out_channels)
        else:
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, node_features, edge_index, edge_attributes):
        """
        Forward pass for the TemporalGCN model.

        Inputs:
          node_features (Tensor): Node features.
          edge_index (Tensor): Edge indices.
          edge_attributes (Tensor): Edge attributes.

        Outputs:
          output (Tensor): Output of the TemporalGCN model.
        """
        # Reshape input for batched temporal processing (num_nodes x timeseries x 1)
        x = node_features.unsqueeze(-1)

        # Pass all features through the temporal layer in a single batch
        x, _ = self.temporal_layer(x)

        # Extract the last hidden state for each node
        x = x[:, -1, :]
        x = F.relu(x)

        # Apply GCNConv layers (spatial information)
        if self.less_layers:
          x = self.conv1(x, edge_index, edge_weight=edge_attributes)
        else:
          x = self.conv1(x, edge_index, edge_weight=edge_attributes)
          x = F.relu(x)
          x = self.conv2(x, edge_index, edge_weight=edge_attributes)

        # Global mean pooling
        x = global_mean_pool(x, None)

        return x
 
class TemporalGAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=3, less_layers = False, temporal_layer = 'LSTM'):
        super(TemporalGAT, self).__init__()
        self.less_layers = less_layers
        self.hidden_channels = hidden_channels
        
        if temporal_layer == 'LSTM':
            # LSTM layer to process time series data
            self.temporal_layer = nn.LSTM(input_size = 1, # 1 feature per time point
                                            hidden_size = hidden_channels, # Number of hidden units
                                            batch_first=True)
        elif temporal_layer == 'RNN':
            # RNN layer to process time series data
            self.temporal_layer = nn.RNN(input_size = 1, # 1 feature per time point
                                            hidden_size = hidden_channels, # Number of hidden units
                                            batch_first=True)

        # Graph Convolution layer
        if self.less_layers:
           self.conv1 = GATConv(hidden_channels, out_channels, heads = heads)
        else:
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels, heads = heads)

    def forward(self, node_features, edge_index, edge_attributes, return_attention_weights = False):
        """
        Forward pass for the TemporalGCN model.

        Inputs:
          node_features (Tensor): Node features.
          edge_index (Tensor): Edge indices.
          edge_attributes (Tensor): Edge attributes.

        Outputs:
          output (Tensor): Output of the TemporalGCN model.
        """
        # Reshape input for batched temporal processing (num_nodes x timeseries x 1)
        x = node_features.unsqueeze(-1)

        # Pass all features through the temporal layer in a single batch
        x, _ = self.temporal_layer(x)

        # Extract the last hidden state for each node
        x = x[:, -1, :]
        x = F.relu(x)

        # Apply spatial layers (spatial information)
        if self.less_layers:
          x, (edges, attnt_coeff) = self.conv1(x, edge_index, return_attention_weights = True)
        else:
          x = self.conv1(x, edge_index, edge_weight=edge_attributes)
          x = F.relu(x)
          x, (edges, attnt_coeff) = self.conv2(x, edge_index, return_attention_weights = True)

        # Global mean pooling
        x = global_mean_pool(x, None)

        if return_attention_weights:
          return x, edges, attnt_coeff
        else:
          return x

#############GRAPH 2 MODELS################
class TimeStaticGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(TimeStaticGCN, self).__init__()

        # Define the input channels
        self.input_channels = input_channels

        # Define the first GCN layer
        self.conv1 = GCNConv(input_channels, hidden_channels)
        # Define the second GCN layer
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, node_features, edge_index, edge_attributes):
        """
        Forward pass for the TimeStaticGCN model.

        Inputs:
          node_features (Tensor): Node features.
          edge_index (Tensor): Edge indices.
          edge_attributes (Tensor): Edge attributes.

        Outputs:
          output (Tensor): Output of the TimeStaticGCN model.
        """
        x = node_features  # (num_timepoints, num_nodes)
        # Apply the first GCN layer with edge attributes
        x = self.conv1(x, edge_index, edge_weight=edge_attributes)
        x = F.relu(x)  # Apply ReLU activation

        # Apply the second GCN layer with edge attributes
        x = self.conv2(x, edge_index, edge_weight=edge_attributes)

        # Apply global mean pooling
        output = global_mean_pool(x, None)

        return output

class TimeStaticGCNGAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, heads = 3):
        super(TimeStaticGCNGAT, self).__init__()

        # Define the input channels
        self.input_channels = input_channels

        # Define the first GCN layer
        self.conv1 = GCNConv(input_channels, hidden_channels)
        # Define the second GCN layer
        self.conv2 = GATConv(hidden_channels, output_channels, heads = heads)

    def forward(self, node_features, edge_index, edge_attributes, return_attention_weights = False):
        """
        Forward pass for the TimeStaticGCN model.

        Inputs:
          node_features (Tensor): Node features.
          edge_index (Tensor): Edge indices.
          edge_attributes (Tensor): Edge attributes.

        Outputs:
          output (Tensor): Output of the TimeStaticGCN model.
        """
        x = node_features  # (num_timepoints, num_nodes)
        # Apply the first GCN layer with edge attributes
        x = self.conv1(x, edge_index, edge_weight=edge_attributes)
        x = F.relu(x)  # Apply ReLU activation

        # Apply the second GCN layer with edge attributes
        x, (edges, attnt_coeff) = self.conv2(x, edge_index, return_attention_weights = True)

        # Apply global mean pooling
        x = global_mean_pool(x, None)

        if return_attention_weights:
          return x, edges, attnt_coeff
        else:
          return x