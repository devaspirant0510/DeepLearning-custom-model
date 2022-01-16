from file_io.read_dataframe_file import ReadDataframeFile, Constant
from preprocessing.feature_selection.select_from_model import BaseSelectFromModel
from preprocessing.sampling.base_sampling import BaseSampling
import preprocessing.sampling.base_sampling as bs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from model.BaseDNN import BaseDNN
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def read_file():
    rdf = ReadDataframeFile("D:\\seungho\\moneyProgram\\scientificProject\\models\\dataset.xlsx", Constant.EXCEL_FILE)
    df = rdf.get_dataframe()
    df = df[:1000]
    return df.iloc[:, :-2], df.iloc[:, -1]


if __name__ == "__main__":
    # 파일 읽기
    x_data, y_data = read_file()
    # 피쳐 추출
    sfm = BaseSelectFromModel(x_data, y_data, 1300)
    feature_size = sfm.get_feature_size()
    x_data = sfm.get_data()
    # 샘플링
    baseSampling = BaseSampling(x_data, y_data, bs.Constant.SMOOTE, 34)
    x_data, y_data = baseSampling.get_data()
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)

    print(x_train)
    # 표준화
    sc = MinMaxScaler()
    sc.fit(x_train, y_train)

    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

    dropout = nn.Dropout(0.5)
    activation = nn.LeakyReLU()
    # 모델정의
    layer = nn.Sequential(
        nn.Linear(feature_size, 150),
        activation,
        dropout,
        nn.Linear(150, 100),
        activation,
        dropout,
        nn.Linear(100, 1),
        nn.Sigmoid(),
    )
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = BaseDNN(layer)
    optimizer = optim.NAdam(model.parameters(), lr=0.1)
    cost_func = nn.BCELoss()
    epoch = 2000
    loss_list = []  # loss 값을 저장할 리스트
    acc_list = []  # acc 값을 저장할 리스트
    f1_list = []  # f1 score 값을 저장할 리스트
    for ep in range(1, epoch + 1):
        acc_data = 0  # 1 epoch accuracy
        loss_data = 0  # 1 epoch loss
        f1_data = 0  # 1 epoch f1 score
        total = 0
        f1_total = 0
        for x, y in train_loader:
            pred = model(x)
            loss = cost_func(pred, y.reshape(-1, 1))
            optimizer.zero_grad()
            loss.backward()  # backpropagation
            optimizer.step()  # weight bias update
            # 예측값이 0.5 이상은 1로 처리 0.5 이하는 0 로 처리
            # 실제 정답값과 비교하여 accuracy 구함
            y_pred = pred.detach().numpy()
            y_pred = np.where(y_pred >= 0.5, 1, 0)
            y_true = y.detach().numpy().reshape(-1, 1)
            acc_data += accuracy_score(y_true,y_pred)
            # accuracy 를 구하기 위해 전체 데이터 사이즈 더함
            total += y.size(0)
            # loss 값 더함
            loss_data += loss
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            f1_data += f1_score(y_true, y_pred, average="weighted")
            f1_total += 1
        acc = (100 * acc_data / f1_total)
        f1_data = (100 * f1_data / f1_total)
        acc_list.append(acc)
        loss_list.append(loss_data)
        f1_list.append(f1_data)
        if ep % 100 == 0:
            print(f"epoch : {ep}/{epoch}\t\tloss:{loss_data}\t\t acc:{acc} \t\t f1 score :{f1_data} ")


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
            pred = model(x)

            y_pred = pred.detach().numpy()
            y_pred = np.where(y_pred >= 0.5, 1, 0)
            test_loss += cost_func(pred, y.reshape(-1, 1))
            y_true = y.detach().numpy().reshape(-1, 1)
            acc += accuracy_score(y_true,y_pred)
            for key, value in f1_dict.items():
                recall = recall_score(y_true, y_pred, average=key)
                precision = precision_score(y_true, y_pred, average=key)
                f1_dict[key] += f1_score(y_true,y_pred,average=key)
            f1_tot += 1
            total += 1
    acc = (100 * acc / total)
    print(f"accuracy: {acc:.2f}%")
    for val, key in f1_dict.items():
        print(f"f1 score {val} : {100 * key / f1_tot:.2f}%")