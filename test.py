import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import config

model_name = input("모델 이름을 입력하세요")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(len(selected_feat), 200)
        self.linear2 = nn.Linear(200, 50)
        self.linear3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.mish = nn.Mish()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


def read_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)  # 파일 읽기
    df = df[:1000]
    return df


def my_feature_selection(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Index]:
    sel = SelectFromModel(RandomForestClassifier(n_estimators=1230))
    sel.fit(x_data, y_data)
    selected_feat = x_data.columns[(sel.get_support())]
    print(len(selected_feat))
    print(selected_feat)
    # 원본데이터에서 추출된 Feature 만 슬라이싱
    return x_data.loc[:, selected_feat], selected_feat


def my_over_sampling(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=42)
    x_data_smote, y_data_smote = smote.fit_resample(x_data, y_data)
    return [x_data_smote, y_data_smote]


def my_scaler(x_train: pd.DataFrame, x_test: pd.DataFrame) -> [np.array, np.array]:
    sc = StandardScaler()
    fit_x_train = sc.fit_transform(x_train)
    fit_x_test = sc.transform(x_test)
    return fit_x_train, fit_x_test


def model_train():
    pass


if __name__ == "__main__":
    # 1. 파일 읽기, x_data y_data 로 나눔
    df = read_file(config.dataset_file_path)
    x_data = df.iloc[:, :-2]  # y 값 제외하고 슬라이싱 , 고장단계도 제외(1,2,3 은 고장 유무가 0이고 4,5는 고장 유무가 1이기때문에 제거)
    y_data = df.iloc[:, -1]  # y 값 만 슬라이싱
    # 2. feature selection
    x_data, selected_feat = my_feature_selection(x_data, y_data)

    # 3. train test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.2, stratify=y_data,
                                                        random_state=32)
    print(Counter(y_train))
    print(Counter(y_test))
    # 4. sampling
    x_train, y_train = my_over_sampling(x_train, y_train)
    # 5. scaler
    x_train, x_test = my_scaler(x_train, x_test)
    # ============================== make model =================================
    # 6. covert to Tensor
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test.to_numpy())
    # 7. make dataset
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # hyper parameter
    epoch = 300
    lr = 0.1

    #
    loss_list = []  # loss 값을 저장할 리스트
    acc_list = []  # acc 값을 저장할 리스트
    f1_list = []  # f1 score 값을 저장할 리스트
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.to(device)  #
    cost_func = nn.BCELoss()  # 이진분류기 때문에 binary cross entropy 사용
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    for ep in range(1, epoch + 1):
        acc_data = 0  # 1 epoch accuracy
        loss_data = 0  # 1 epoch loss
        f1_data = 0  # 1 epoch f1 score
        recall_data = 0  # 1 epoch recall
        precision_data = 0  # 1 epoch precision
        total = 0
        f1_total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = cost_func(pred, y.reshape(-1, 1))
            optimizer.zero_grad()
            loss.backward()  # backpropagation
            optimizer.step()  # weight bias update
            # 예측값이 0.5 이상은 1로 처리 0.5 이하는 0 로 처리
            # 실제 정답값과 비교하여 accuracy 구함
            y_pred = pred.cpu().detach().numpy()
            y_pred = np.where(y_pred >= 0.5, 1, 0)
            y_true = y.cpu().detach().numpy().reshape(-1, 1)
            acc_data += np.sum(y_pred == y_true)
            # accuracy 를 구하기 위해 전체 데이터 사이즈 더함
            total += y.size(0)
            # loss 값 더함
            loss_data += loss
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            precision_data += precision
            recall_data += recall
            f1_data += f1_score(y_true, y_pred, average="weighted", zero_division=0)
            f1_total += 1
        acc = (100 * acc_data / total)
        f1_data = (100 * f1_data / f1_total)
        precision_data = (100 * precision_data / f1_total)
        recall_data = (100 * recall_data / f1_total)
        acc_list.append(acc)
        loss_list.append(loss_data)
        f1_list.append(f1_data)
        if ep % 100 == 0:
            print(
                f"epoch : {ep}/{epoch}\t\tloss:{loss_data}\t\t acc:{acc:.3f} \t\t f1 score :{f1_data:.3f} \t\t recall data:{recall_data:.3f} \t\t precision data :{precision_data:.3f} ")

    total = 0
    acc = 0
    f1_dict = {
        "weighted": 0,
        "macro": 0,
        "micro": 0
    }
    test_loss = 0
    f1_tot = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            y_pred = pred.cpu().detach().numpy()
            y_pred = np.where(y_pred >= 0.5, 1, 0)
            test_loss += cost_func(pred, y.reshape(-1, 1))
            y_true = y.cpu().detach().numpy().reshape(-1, 1)
            acc += np.sum(y_pred == y_true)
            for key, value in f1_dict.items():
                recall = recall_score(y_true, y_pred, average=key)
                precision = precision_score(y_true, y_pred, average=key)
                f1_dict[key] += 2 * ((recall * precision) / (recall + precision))
            f1_tot += 1
            total += y.size(0)
    acc = (100 * acc / total)
    print(f"accuracy: {acc:.2f}%")
    for val, key in f1_dict.items():
        print(f"f1 score {val} : {100 * key / f1_tot:.2f}%")

    torch.save(model.state_dict(),f"{model_name}.pth")
# rus = RandomUnderSampler(random_state=42)
# print(x_data, y_data)
# x_data_rus, y_data_rus = rus.fit_resample(x_data, y_data)
# print(x_data_rus, y_data_rus)
