import os
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 그래프 신경망 모델 정의
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

# 데이터셋 로드 및 전처리 함수 (가상의 함수, 실제 데이터셋에 맞게 수정 필요)
def load_and_preprocess_data():
    # 가중치 파일 경로
    weights_file = 'C:/Users/tae97/Desktop/GCN ver.2/최신버전/Sig_HLP score gene list ver.4.xlsx'
    weights_dict = load_weights(weights_file)

    # 데이터셋 디렉토리 경로
    hyperlipidemia_directory = 'C:/Users/tae97/Desktop/공용 data 폴더/hlp drug matirx'
    etc_directory = 'C:/Users/tae97/Desktop/공용 data 폴더/other drug matrix'

    # 데이터셋 로드
    hyperlipidemia_data_list = load_data_from_directory(hyperlipidemia_directory)
    etc_data_list = load_data_from_directory(etc_directory)

    # 그래프 데이터 객체 생성 및 가중치 적용
    all_graphs = []
    for data_list, label in [(hyperlipidemia_data_list, 0), (etc_data_list, 1)]:
        for data in data_list:
            graph = create_graph_data(pd.DataFrame(data))
            apply_weights(graph, weights_dict)
            graph.y = torch.tensor([label], dtype=torch.float)
            all_graphs.append(graph)

    return all_graphs

# 그래프 데이터 객체 생성 함수
def create_graph_data(adj_matrix, features=None, max_dim=100):
    adj_matrix_np = adj_matrix.values
    edge_index = torch.tensor(list(zip(*np.nonzero(adj_matrix_np))), dtype=torch.long).t().contiguous()

    if features is None:
        features = torch.eye(adj_matrix.shape[0])
    else:
        features = torch.tensor(features, dtype=torch.float)

    # 모든 특성을 max_dim 크기로 패딩
    if features.size(1) < max_dim:
        padding = torch.zeros(features.size(0), max_dim - features.size(1))
        features = torch.cat([features, padding], dim=1)

    x = features
    data = Data(x=x, edge_index=edge_index)
    return data

# 데이터 로드 함수
def load_data_from_directory(directory):
    files = os.listdir(directory)
    data_list = []
    for file in files:
        if file.endswith('.xlsx'):  # .xlsx 파일을 찾도록 변경
            file_path = os.path.join(directory, file)
            data = pd.read_excel(file_path, index_col=0)  # read_excel 함수 사용
            data_list.append(data)
    return data_list

# 가중치 데이터 로드 및 딕셔너리 변환 함수
def load_weights(file_path):
    weights_df = pd.read_excel(file_path)  # Excel 파일 로드
    weights_dict = dict(zip(weights_df['encoded_value'], weights_df['sigmoid']))
    return weights_dict

# 가중치 데이터 적용 함수
def apply_weights(graph, weights_dict):
    # 가중치를 적용할 노드 특성을 선택합니다.
    # 여기서는 'encoded_value'를 노드 특성으로 가정합니다.
    for i in range(graph.x.size(0)):
        encoded_value = int(graph.x[i, 0].item())  # 첫 번째 특성을 encoded_value로 가정
        graph.x[i, 0] = weights_dict.get(encoded_value, 0)  # 가중치 적용, 없으면 0



# 부트스트래핑을 위한 함수
def bootstrap_evaluation(model, datasets, n_iterations=100, test_size=0.25):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # 손실 함수와 최적화 함수 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(n_iterations):
        # 데이터셋에서 무작위로 샘플링 (복원 추출)
        sampled_datasets = [random.choice(datasets) for _ in range(len(datasets))]
        train_datasets, val_datasets = train_test_split(sampled_datasets, test_size=test_size)

        # 모델 훈련 및 평가
        model = train_model(model, train_datasets, val_datasets, criterion, optimizer, epochs=100, patience=3)
        precision, recall, f1 = evaluate_model(model, val_datasets, criterion)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # 부트스트랩 결과 요약
    print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# 모델 훈련 함수에 Early Stopping 적용
def train_model(model, train_graphs, val_graphs, criterion, optimizer, epochs=100, patience=3):
    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_graphs:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.view(-1), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 검증 데이터셋에 대한 성능 평가
        val_loss, _, _ = evaluate_model(model, val_graphs, criterion)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_graphs)}, Validation Loss: {val_loss}")

        # Early Stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return model

# 모델 평가 함수
def evaluate_model(model, test_graphs, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_graphs:
            out = model(data)
            loss = criterion(out, data.y.unsqueeze(1))
            total_loss += loss.item()

            preds = (out > 0).float().view(-1)
            y_true.extend(data.y.tolist())
            y_pred.extend(preds.tolist())

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


# 메인 로직
datasets = load_and_preprocess_data()  # 데이터셋 로드 및 전처리
model = GNN(input_dim=100, hidden_dim=64, output_dim=1)  # 모델 초기화
bootstrap_evaluation(model, datasets, n_iterations=100)  # 부트스트래핑을 통한 모델 평가
