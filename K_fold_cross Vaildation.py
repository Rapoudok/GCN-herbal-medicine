import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def evaluate(model, val_graphs, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in val_graphs:
            out = model(data)
            loss = criterion(out, data.y.unsqueeze(1))
            total_loss += loss.item()
            preds = (out > 0).float().view(-1)
            y_true.extend(data.y.tolist())
            y_pred.extend(preds.tolist())
    val_loss = total_loss / len(val_graphs)
    val_acc = sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]) / len(y_true)
    return val_loss, val_acc, y_true, y_pred



def load_data_from_directory(directory):
    files = os.listdir(directory)
    data_list = []
    for file in files:
        if file.endswith('.xlsx'):  
            file_path = os.path.join(directory, file)
            data = pd.read_excel(file_path, index_col=0)  
            data_list.append(data)
    return data_list


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


def load_weights(file_path):
    weights_df = pd.read_excel(file_path)
    weights_dict = dict(zip(weights_df['encoded_value'], weights_df['sigmoid']))
    return weights_dict


def apply_weights(graph, weights_dict):
    for i in range(graph.x.size(0)):
        encoded_value = int(graph.x[i, 0].item())  
        graph.x[i, 0] = weights_dict.get(encoded_value, 0)


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_performance_metrics(precision, recall, f1):
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Performance Metrics')
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
    plt.show()


weights_file = #disease weight data PATH
weights_dict = load_weights(weights_file)



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
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


disease_directory = #'disease drug data PATH'
etc_directory = #'other drug data PATH'
disease_directory_list = load_data_from_directory(disease_directory)
etc_directory_list = load_data_from_directory(etc_directory)

disease_graphs = [create_graph_data(pd.DataFrame(d)) for d in disease_directory_list]
etc_graphs = [create_graph_data(pd.DataFrame(d)) for d in etc_directory_list]


for graph in disease_graphs + etc_graphs:
    apply_weights(graph, weights_dict) 

for graph in disease_graphs:
    graph.y = torch.tensor([0], dtype=torch.float) 

for graph in etc_graphs:
    graph.y = torch.tensor([1], dtype=torch.float) 

all_graphs = disease_graphs + etc_graphs
train_graphs, val_graphs = train_test_split(all_graphs, test_size=0.25, shuffle=True)

input_dim = 100 
hidden_dim = 64
output_dim = 1

model = GNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

model_path = #'your model PATH ex).pth data' 
model.load_state_dict(torch.load(model_path))
model.eval()

k = 5  
kf = KFold(n_splits=k, shuffle=True, random_state=42)

all_graphs = disease_graphs + etc_graphs
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_graphs)):
    train_graphs = [all_graphs[i] for i in train_idx]
    val_graphs = [all_graphs[i] for i in val_idx]
    model.load_state_dict(torch.load(model_path))
    model.eval()
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_graphs, criterion)
    fold_results.append((val_loss, val_acc, y_true, y_pred))
    print(f"Fold {fold + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


# 전체 결과 분석
avg_loss = np.mean([result[0] for result in fold_results])
avg_acc = np.mean([result[1] for result in fold_results])
total_y_true = [y for _, _, y, _ in fold_results]
total_y_pred = [y for _, _, _, y in fold_results]
# 전체 fold의 예측 결과를 단일 리스트로 변환
total_y_true_flat = [item for sublist in total_y_true for item in sublist]
total_y_pred_flat = [item for sublist in total_y_pred for item in sublist]

print(f"Average Validation Loss: {avg_loss:.4f}, Average Validation Accuracy: {avg_acc:.4f}")

