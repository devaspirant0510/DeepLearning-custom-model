import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler,Normalizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

file_path = "파일경로"
df = pd.read_excel(file_path)   # 파일 읽기
df = df[:1000]

x_train = df.iloc[:, :-2]  # y 값 제외하고 슬라이싱 , 고장단계도 제외(1,2,3 은 고장 유무가 0이고 4,5는 고장 유무가 1이기때문에 제거)
y_train = df.iloc[:, -1]  # y 값 만 슬라이싱


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)



ros = SMOTE(random_state=42)
x_train = pd.DataFrame(x_train, columns=df.columns[:-2])
X_res, y_res = ros.fit_resample(x_train, y_train)
print('Resampled dataset shape %s' % Counter(y_res))

train_x, test_x, train_y, test_y = train_test_split(X_res, y_res, random_state=42, stratify=y_res, test_size=0.3)
#sel = SelectFromModel(RandomForestClassifier(n_estimators=1230))
#sel.fit(x_train, y_train)
selected_feat = x_train.columns[()]
print(len(selected_feat)) #18
print(selected_feat)

# 원본데이터에서 추출된 Feature 만 슬라이싱
train_x = train_x.loc[:, selected_feat]
test_x = test_x.loc[:, selected_feat]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(len(selected_feat), 200)  # len(selected_feat 바꿔줌
        self.linear2 = nn.Linear(200, 50)
        self.linear3 = nn.Linear(50, 1)
        self.relu = nn.LeakyReLU()
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


train_x = torch.FloatTensor(train_x.to_numpy())
train_y = torch.FloatTensor(train_y.to_numpy())
test_x = torch.FloatTensor(test_x.to_numpy())
test_y = torch.FloatTensor(test_y.to_numpy())

train_dataset = TensorDataset(train_x, train_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

epoch = 2000
lr = 0.01
loss_list = []  # loss 값을 저장할 리스트
acc_list = []  # acc 값을 저장할 리스트
f1_list = []  # f1 score 값을 저장할 리스트

model = Model()
cost_func = nn.BCELoss()  # 이진분류기 때문에 binary cross entropy 사용
optimizer = optim.SGD(model.parameters(), lr=lr)  # 경사하강법은 SGD 알고리즘 사용

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
        acc_data += accuracy_score(y_true, y_pred)
        # accuracy 를 구하기 위해 전체 데이터 사이즈 더함
        total += y.size(0)
        # loss 값 더함
        loss_data += loss
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        f1_data += f1_score(y_true, y_pred, average="macro")
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
        acc += accuracy_score(y_true, y_pred)
        for key, value in f1_dict.items():
            recall = recall_score(y_true, y_pred, average=key)
            precision = precision_score(y_true, y_pred, average=key)
            f1_dict[key] += f1_score(y_true, y_pred, average=key)
        f1_tot += 1
        total += 1
acc = (100 * acc / total)
print(f"accuracy: {acc:.2f}%")
for val, key in f1_dict.items():
    print(f"f1 score {val} : {100 * key / f1_tot:.2f}%")

'''''
fig, ax1 = plt.subplots()
ax1.plot(acc_list, color='blue', label="accuracy")
ax1.plot(f1_list, color="green", label="f1 score")
ax1.set_ylabel("accuracy")
ax2 = ax1.twinx()
ax2.plot(loss_list, color='red', label="loss")
ax2.set_ylabel("loss")
ax1.set_xlabel("epoch")
plt.title("oversampling SMOTE")
fig.legend()
plt.savefig("../img/oversampling_smote.png")
'''''