import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

def load_weights(file_path):
    weights_df = pd.read_excel(file_path)  
    weights_dict = dict(zip(weights_df['encoded_value'], weights_df['sigmoid']))
    return weights_dict

def apply_weights(graph, weights_dict):
    for i in range(graph.x.size(0)):
        encoded_value = int(graph.x[i, 0].item())
        graph.x[i, 0] = weights_dict.get(encoded_value, 0) 

def predict(model, graph_data):
    with torch.no_grad():
        out = model(graph_data)
        probabilities = torch.sigmoid(out).view(-1)
        predictions = (probabilities > 0.5).float()
        return predictions, probabilities

def load_data_from_directory(directory):
    files = os.listdir(directory)
    data_dict = {}
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(directory, file)
            data = pd.read_excel(file_path, index_col=0)
            data_dict[file] = data
    return data_dict

def create_graph_data(adj_matrix, features=None, max_dim=100):
    adj_matrix_np = adj_matrix.values
    edge_index = torch.tensor(list(zip(*np.nonzero(adj_matrix_np))), dtype=torch.long).t().contiguous()
    if features is None:
        features = torch.eye(adj_matrix.shape[0])
    else:
        features = torch.tensor(features, dtype=torch.float)
    if features.size(1) < max_dim:
        padding = torch.zeros(features.size(0), max_dim - features.size(1))
        features = torch.cat([features, padding], dim=1)
    x = features
    data = Data(x=x, edge_index=edge_index)
    return data


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


model = GNN(input_dim=100, hidden_dim=64, output_dim=1)
model_path = #your pre-training model PATH
model.load_state_dict(torch.load(model_path))
model.eval()


weights_file = #1target disease weight data PATH
weights_dict = load_weights(weights_file)
directory = #Herbal prescription data PATH
data_dict = load_data_from_directory(directory)


for file_name, data in data_dict.items():
    graph_data = create_graph_data(data)
    apply_weights(graph_data, weights_dict)
    predictions, probabilities = predict(model, graph_data)
    print(f"File: {file_name}, Predicted Class: {predictions.item()}, Probability: {probabilities.item()}")
