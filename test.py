import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, KMeansSMOTE, BorderlineSMOTE, SMOTEN, SMOTENC, \
    SVMSMOTE
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import config
import pprint
from deprecated import deprecated
from sklearn.ensemble import ExtraTreesClassifier  # 이 방법도 있음


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
    print("\n선택된 feature의 개수 : %s \n" % len(selected_feat))
    # # 원본데이터에서 추출된 Feature 만 슬라이싱
    # estimator = SVR(kernel="linear")
    # selector = RFE(estimator, n_features_to_select=5, step=1)
    # selector = selector.fit(x_data, y_data)
    # print(selector.coef_)
    # print(selector.feature_importance_)
    return x_data.loc[:, selected_feat], selected_feat


def my_over_sampling_smote(x_data: pd.DataFrame, y_data: pd.DataFrame, random_state=42) -> [pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=random_state)
    x_data_smote, y_data_smote = smote.fit_resample(x_data, y_data)
    return [x_data_smote, y_data_smote]


def my_over_sampling_smoten(x_data: pd.DataFrame, y_data: pd.DataFrame, random_state=42) -> [pd.DataFrame, pd.Series]:
    smoten = SMOTEN(random_state=random_state)
    x_data_under, y_data_under = smoten.fit_resample(x_data, y_data)
    print('\nResampled dataset shape %s \n' % Counter(y_data_under))  # RandomUnderSampler or Tomelink
    return [x_data_under, y_data_under]


@deprecated("확장가능한 함수를 만들었습니다.")
def my_under_sampling(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    under_sampling = RandomUnderSampler(random_state=42)
    x_data_under, y_data_under = under_sampling.fit_resample(x_data, y_data)
    print('\nResampled dataset shape %s \n' % Counter(y_data_under))  # RandomUnderSampler or Tomelink
    return [x_data_under, y_data_under]


@deprecated("확장가능한 함수를 만들었습니다.")
def my_over_sampling(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=42)
    x_data_smote, y_data_smote = smote.fit_resample(x_data, y_data)
    print('\nResampled dataset shape %s \n' % Counter(y_data_smote))  # SMOTE or ADASYN
    return [x_data_smote, y_data_smote]


def my_over_sampling_adasyn(x_data: pd.DataFrame, y_data: pd.DataFrame, random_state=42) -> [pd.DataFrame, pd.Series]:
    adasyn = ADASYN(random_state=random_state)
    x_data_adasyn, y_data_adasyn = adasyn.fit_resample(x_data, y_data)
    return [x_data_adasyn, y_data_adasyn]


def my_under_sampling_tomelink(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    tomelink = TomekLinks()
    x_data_adasyn, y_data_adasyn = tomelink.fit_resample(x_data, y_data)
    return [x_data_adasyn, y_data_adasyn]


def my_scaler_normalizer(x_train: pd.DataFrame, x_test: pd.DataFrame) -> [np.array, np.array]:
    sc = Normalizer()
    fit_x_train = sc.fit_transform(x_train)
    fit_x_test = sc.transform(x_test)
    return fit_x_train, fit_x_test


@deprecated("확장가능한 함수를 만들었습니다.")
def my_scaler(x_train: pd.DataFrame, x_test: pd.DataFrame) -> [np.array, np.array]:
    sc = StandardScaler()
    fit_x_train = sc.fit_transform(x_train)
    fit_x_test = sc.transform(x_test)
    return fit_x_train, fit_x_test


def my_scaler_min_max(x_train: pd.DataFrame, x_test: pd.DataFrame) -> [np.array, np.array]:
    sc = MinMaxScaler()
    fit_x_train = sc.fit_transform(x_train)
    fit_x_test = sc.transform(x_test)
    return fit_x_train, fit_x_test


def my_scaler_standard(x_train: pd.DataFrame, x_test: pd.DataFrame) -> [np.array, np.array]:
    sc = StandardScaler()
    fit_x_train = sc.fit_transform(x_train)
    fit_x_test = sc.transform(x_test)
    return fit_x_train, fit_x_test


def model_train():
    pass


def my_random_forest(x_train, y_train, x_test, n_es, max_depth, random_state=43):
    clf = RandomForestClassifier(random_state=random_state, max_depth=max_depth, n_estimators=n_es)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return pred, clf


def my_knn_model(x_train, y_train, x_test, k=5, weights="uniform", algorithm="auto"):
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return pred, clf


def show_rating(y_pred, y_true):
    pass


if __name__ == "__main__":
    if config.is_model_save:
        model_name = input("모델 이름을 입력하세요")

    # 1. 파일 읽기, x_data y_data 로 나눔
    df = read_file(config.dataset_file_path)
    x_data = df.iloc[:, :-2]  # y 값 제외하고 슬라이싱 , 고장단계도 제외(1,2,3 은 고장 유무가 0이고 4,5는 고장 유무가 1이기때문에 제거)
    y_data = df.iloc[:, -1]  # y 값 만 슬라이싱

    df.info()  # type 및 자료 확인
    # print(df.describe())
    # print(df.corr()[["break_down"]])
    # df_corr = df.corr()
    # print("\n")
    # print(df_corr.loc[np.abs(df_corr["break_down"] > 0.1), "break_down"])

    # 2. feature selection
    x_data, selected_feat = my_feature_selection(x_data, y_data)
    print(x_data.shape)

    # 3. train test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, stratify=y_data, random_state=32)
    print(Counter(y_train))
    print(Counter(y_test))

    # 4. sampling
    x_train, y_train = my_over_sampling_smote(x_train, y_train)
    # # 5. scaler
    x_train, x_test = my_scaler_standard(x_train, x_test)
    # # ============================== make model =================================
    # # pred, clf = my_random_forest(x_train, y_train, x_test, 1300, 120, 34)
    # pred, clf = my_knn_model(x_train, y_train, x_test, k=4, weights="distance")
    # print(accuracy_score(y_test, pred))
    # clf = RandomForestClassifier()
    # pars = {"max_depth": list(range(90, 100)),
    #         "n_estimators": list(range(70, 75)),
    #         "random_state": [42]}
    # gcv = GridSearchCV(clf, pars)
    # gcv.fit(x_train, y_train)
    # pred = clf.predict(x_test)
    # print(accuracy_score(y_test, pred))
    # print(f1_score(y_test, pred))
    # print(recall_score(y_test, pred))
    # print(precision_score(y_test, pred))
    # print(confusion_matrix(y_test, pred))
    # print(f1_score(y_test, pred))

    # 5. scaler
    x_train, x_test = my_scaler(x_train, x_test)

    # ============================== make model =================================

    # clf = my_random_forest(x_train,y_train,1300,120,34)
    # clf = RandomForestClassifier()
    # pars = {"max_depth": list(range(80, 100)),
    #         "n_estimators": list(range(70, 80)),
    #         "random_state": [42]}
    # gcv = GridSearchCV(clf, pars)
    # gcv.fit(x_train, y_train)
    # pred = gcv.best_estimator_.predict(x_test)
    # print(accuracy_score(y_test, pred))
    # print(f1_score(y_test, pred))
    # print(recall_score(y_test, pred))
    # print(precision_score(y_test, pred))
    # print(confusion_matrix(y_test, pred))
    # print(f1_score(y_test, pred))

    # 6. covert to Tensor (인공신경망)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test.to_numpy())
    # 7. make dataset
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)  # batch size test data와 딱 떨어지게 맞추어줌
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True)

    # hyper parameter
    epoch = 2000
    lr = 0.1

    loss_list = []  # loss 값을 저장할 리스트
    acc_list = []  # acc 값을 저장할 리스트
    f1_list = []  # f1 score 값을 저장할 리스트
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.to(device)  #
    cost_func = nn.BCELoss()  # 이진분류기 때문에 binary cross entropy 사용
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for ep in range(1, epoch + 1):
        acc_data = 0  # 1 epoch accuracy
        loss_data = 0  # 1 epoch loss
        f1_data = 0  # 1 epoch f1 score
        recall_data = 0  # 1 epoch recall
        precision_data = 0  # 1 epoch precision
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        total = 0
        auc_data = 0
        f1_total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = cost_func(pred, y.reshape(-1, 1))
            optimizer.zero_grad()
            loss.backward()  # backpropagation
            optimizer.step()  # weight bias update
            # ===================== 성능 측정 (accuracy,f1 score,recall,precision) ========================
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
            # print(conf_matrix)
            conf_matrix = confusion_matrix(y_true, y_pred)
            TN = conf_matrix[0, 0]  # 정상인 기계를 정상이라 예측
            FP = conf_matrix[0, 1]
            FN = conf_matrix[1, 0]
            TP = conf_matrix[1, 1]  # 고장난 것을 고장이라 예측
            auc_score = roc_auc_score(y_true, y_pred) # roc 커브
            auc_data+=auc_score

        acc = (100 * acc_data / total)
        f1_data = (100 * f1_data / f1_total)
        precision_data = (100 * precision_data / f1_total)
        recall_data = (100 * recall_data / f1_total)
        auc_data = (100*auc_data/f1_total)
        acc_list.append(acc)
        loss_list.append(loss_data)
        f1_list.append(f1_data)
        if ep % 100 == 0:
            print(
                f"epoch : {ep}/{epoch}\t\tloss:{loss_data}\t\t accuracy:{acc:.3f} \t\t recall:{recall_data:.3f} \t\t precision:{precision_data:.3f} \t\t f1 score:{f1_data:.3f} \t\t auc score :{auc_data:.3f} \t\t TP:{TP} FN:{FN} FP:{FP} TN:{TN}")

    total = 0
    acc = 0
    rec = 0
    pre = 0
    f1s = 0
    f1_dict = {
        "weighted": 0,
        "macro": 0,
        "micro": 0
    }
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
            for key, value in f1_dict.items():
                recall = recall_score(y_true, y_pred, average=key,
                                      zero_division=0)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html 0 나눗셈이 있을 때 반환할 값을 설정합니다. "warn"으로 설정하면 0으로 작동하지만 경고도 발생합니다.
            print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))
            acc += accuracy_score(y_true, y_pred)
            rec += recall_score(y_true, y_pred)
            pre += precision_score(y_true, y_pred)
            f1s += f1_score(y_true, y_pred)
            for key, value in f1_dict.items():
                recall = recall_score(y_true, y_pred, average=key, zero_division=0)
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

    if config.is_model_save and model is not None:
        torch.save(model.state_dict(), f"{model_name}.pt")
    # print(y_true.shape)           왜 34 가 나오는지 확인한것
    pre = (100 * pre / total)
    f1s = (100 * f1s / total)

    print("<Class 1에 대한 지표>")
    print(f"\naccuracy: {acc:.2f}%")
    print(f"recall: {rec:.2f}%")
    print(f"precision: {pre:.2f}%")
    print(f"f1 score: {f1s:.2f}%\n")

    for val, key in f1_dict.items():
        print(f"f1 score {val} : {100 * key / f1_tot:.2f}%")
        # micro는 accuray 값과 동일하고
        # macro는 각 클래스의 불균형을 반영하지 않은 지표 (산술평균)
        # weighted는 가중치를 통해 각 클래스의 불균형을 반영해 준 지표이다. 분류 자체를 잘했는지를 보려면 이 지표가 괜찮은 것 같다.
    # fig, ax1 = plt.subplots()
    # ax1.plot(acc_list, color='blue', label="accuarcy")
    # ax1.plot(f1_list, color="green", label="f1 score")
    # ax1.set_ylabel("accuarcy")
    #
    # ax2 = ax1.twinx()
    # ax2.plot(loss_list, color='red', label="loss")
    # ax2.set_ylabel("loss")
    # ax1.set_xlabel("epoch")
    # fig.legend()
    # https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/ ...feature selection
    # https://machinelearningmastery.com/feature-selection-machine-learning-python/
