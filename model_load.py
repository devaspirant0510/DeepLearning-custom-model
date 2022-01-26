import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from test import read_file, my_feature_selection, my_scaler_standard,Model
import config

name = input('불러올 모델이름을 입력하세요')
model = Model()
model.load_state_dict(torch.load(f"{name}.pt"))

df = read_file(config.dataset_file_path)
x_data = df.iloc[:, :-2]  # y 값 제외하고 슬라이싱 , 고장단계도 제외(1,2,3 은 고장 유무가 0이고 4,5는 고장 유무가 1이기때문에 제거)
y_data = df.iloc[:, -1]  # y 값 만 슬라이싱
# 2. feature selection
#x_data, selected_feat = my_feature_selection(x_data, y_data)
print(x_data.shape)

# 3. train test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.3, stratify=y_data,
                                                    random_state=32)
# 5. scaler
x_train, x_test = my_scaler_standard(x_train, x_test)
total = 0
acc = 0
rec = 0

cost_func = nn.BCELoss()
f1_dict = {
    "weighted": 0,
    "macro": 0,
    "micro": 0
}

x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test.to_numpy())
# 7. make dataset
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
test_loss = 0
f1_tot = 0
with torch.no_grad():  # gradient를 업데이트 하지 않는것 orch.no_grad()는 오차 역전파에 사용하는 계산량을 줄여서 처리 속도를 높인다.https://green-late7.tistory.com/48  https://go-hard.tistory.com/64
    for x, y in test_loader:
        pred = model(x)

        y_pred = pred.detach().numpy()
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        test_loss += cost_func(pred, y.reshape(-1, 1))
        y_true = y.detach().numpy().reshape(-1, 1)
        print(confusion_matrix(y_true, y_pred))
        # print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))
        acc += accuracy_score(y_true, y_pred)
        rec += recall_score(y_true, y_pred)
        for key, value in f1_dict.items():
            recall = recall_score(y_true, y_pred, average=key,
                                  zero_division=0)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html 0 나눗셈이 있을 때 반환할 값을 설정합니다. "warn"으로 설정하면 0으로 작동하지만 경고도 발생합니다.
            precision = precision_score(y_true, y_pred, average=key, zero_division=0)
            f1_dict[key] += f1_score(y_true, y_pred, average=key)
        f1_tot += 1
        total += 1

rec = (100 * rec / total)
acc = (100 * acc / total)

print(f"accuracy: {acc:.2f}%")
print(f"recall: {rec:.2f}%")
for val, key in f1_dict.items():
    print(f"f1 score {val} : {100 * key / f1_tot:.2f}%")
