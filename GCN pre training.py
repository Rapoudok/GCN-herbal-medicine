import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns


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


def save_random_states(filename, torch_state):
    states = {
        'np_state': np.random.get_state(),
        'random_state': random.getstate(),
        'torch_state': torch_state
    }
    with open(filename, 'wb') as f:
        pickle.dump(states, f)


def save_model(epoch, model, directory=#"save model PATH"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"model_epoch_{epoch}.pth"
    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)


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


def plot_loss(epoch_train_losses, epoch_val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_train_losses, label='Train Loss')
    plt.plot(epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()


weights_file = #disease weight data PATH
weights_dict = load_weights(weights_file)


# Defining a graph neural network model
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


# Dataset load
disease_directory = #'disease drug data PATH'
etc_directory = #'other drug data PATH'


disease_directory_list = load_data_from_directory(disease_directory)
etc_directory_list = load_data_from_directory(etc_directory)


# Creating graph data objects
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


def train(model, train_loader, optimizer, criterion, l1_lambda=0.00005):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.view(-1), data.y)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


model = GNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
train_loader = DataLoader(all_graphs, batch_size=48, shuffle=True)


epochs = 500
best_val_loss = float('inf')
patience = 3
trigger_times = 0


epoch_train_losses = []
epoch_val_losses = []
validation_accuracies = []


# training loops
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_graphs, criterion)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    # save model
    save_model(epoch + 1, model)
    epoch_train_losses.append(train_loss)
    epoch_val_losses.append(val_loss)
    validation_accuracies.append(val_acc)
    # Early stopping 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


train_loss, train_acc, train_y_true, train_y_pred = evaluate(model, train_graphs, criterion)
test_loss, test_acc, test_y_true, test_y_pred = evaluate(model, val_graphs, criterion)


total_y_true = train_y_true + test_y_true
total_y_pred = train_y_pred + test_y_pred


total_conf_matrix = confusion_matrix(total_y_true, total_y_pred)
total_f1 = f1_score(total_y_true, total_y_pred)
total_recall = recall_score(total_y_true, total_y_pred)
total_precision = precision_score(total_y_true, total_y_pred)


precision = precision_score(total_y_true, total_y_pred)
recall = recall_score(total_y_true, total_y_pred)
f1 = f1_score(total_y_true, total_y_pred)


print(f"Total Confusion Matrix:\n{total_conf_matrix}")
print(f"Total F1 Score: {total_f1:.4f}")
print(f"Total Recall: {total_recall:.4f}")
print(f"Total Precision: {total_precision:.4f}")


plot_confusion_matrix(total_conf_matrix)
plot_loss(epoch_train_losses, epoch_val_losses)


def plot_validation_accuracy(validation_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.show()
plot_validation_accuracy(validation_accuracies)


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


plot_performance_metrics(precision, recall, f1)