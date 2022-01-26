import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler
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
from deprecated import deprecated
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier  # 이 방법도 있음

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACC_DATA = "acc_data"
LOSS_DATA = "loss_data"
F1_DATA = "f1_data"
RECALL_DATA = "recall_data"
PRECISION_DATA = "precision_data"
AUC_DATA = "auc_data"


# ========================  신경망 모델 정의 ================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(len(selected_feat), 150)
        self.linear2 = nn.Linear(150, 70)
        self.linear3 = nn.Linear(70, 30)
        self.linear4 = nn.Linear(30, 1)
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()
        self.mish = nn.Mish()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        self.layer = nn.Sequential(
            self.linear1,
            self.leakyRelu,
            self.dropout,
            self.linear2,
            self.leakyRelu,
            self.dropout,
            self.linear3,
            self.leakyRelu,
            self.dropout,
            self.linear4,
            self.sigmoid
        )

    def forward(self, x):
        return self.layer(x)


# ===================================== 파일 읽기 ====================================
def read_file(path: str) -> pd.DataFrame:
    data = pd.read_excel(path)  # 파일 읽기
    data = data[:1000]
    return data


# ==================================== Feature Selection ===============================
def my_feature_selection(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Index]:
    sel = SelectFromModel(RandomForestClassifier(n_estimators=1230))
    sel.fit(x_data, y_data)
    selected_feat = x_data.columns[(sel.get_support())]
    print(len(selected_feat))
    print(selected_feat)
    print("\n선택된 feature의 개수 : %s \n" % len(selected_feat))
    return x_data.loc[:, selected_feat], selected_feat


# ==================================== over sampling =============================================
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
def my_over_sampling(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=42)
    x_data_smote, y_data_smote = smote.fit_resample(x_data, y_data)
    print('\nResampled dataset shape %s \n' % Counter(y_data_smote))  # SMOTE or ADASYN
    return [x_data_smote, y_data_smote]


# ==================================== under sampling =============================================
def my_over_sampling_adasyn(x_data: pd.DataFrame, y_data: pd.DataFrame, random_state=42) -> [pd.DataFrame, pd.Series]:
    adasyn = ADASYN(random_state=random_state)
    x_data_adasyn, y_data_adasyn = adasyn.fit_resample(x_data, y_data)
    return [x_data_adasyn, y_data_adasyn]


def my_under_sampling_tomelink(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    tomelink = TomekLinks()
    x_data_adasyn, y_data_adasyn = tomelink.fit_resample(x_data, y_data)
    return [x_data_adasyn, y_data_adasyn]


@deprecated("확장가능한 함수를 만들었습니다.")
def my_under_sampling(x_data: pd.DataFrame, y_data: pd.DataFrame) -> [pd.DataFrame, pd.Series]:
    under_sampling = RandomUnderSampler(random_state=42)
    x_data_under, y_data_under = under_sampling.fit_resample(x_data, y_data)
    print('\nResampled dataset shape %s \n' % Counter(y_data_under))  # RandomUnderSampler or Tomelink
    return [x_data_under, y_data_under]


# ============================= preprocessing ============================================
def my_robust_scaler(x_train: pd.DataFrame, x_test: pd.DataFrame) -> [np.array, np.array]:
    sc = RobustScaler()
    fit_x_train = sc.fit_transform(x_train)
    fit_x_test = sc.transform(x_test)
    return fit_x_train, fit_x_test


def my_scaler_normalizer(x_train: pd.DataFrame, x_test: pd.DataFrame) -> [np.array, np.array]:
    sc = Normalizer()
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


@deprecated("확장가능한 함수를 만들었습니다.")
def my_scaler(x_train: pd.DataFrame, x_test: pd.DataFrame) -> [np.array, np.array]:
    sc = StandardScaler()
    fit_x_train = sc.fit_transform(x_train)
    fit_x_test = sc.transform(x_test)
    return fit_x_train, fit_x_test


def model_train(train_loader, model, epoch, cost_func, optimizer):
    global device
    loss_list = []  # loss 값을 저장할 리스트
    acc_list = []  # acc 값을 저장할 리스트
    f1_list = []  # f1 score 값을 저장할 리스트
    score_dict = {
        "acc_data": 0,
        "loss_data": 0,
        "f1_data": 0,
        "recall_data": 0,
        "precision_data": 0,
        "auc_data": 0
    }
    model.to(device)  #
    for ep in range(1, epoch + 1):
        for key_, _ in score_dict.items():
            score_dict[key_] = 0
        # score_dict[ACC_DATA] = 0  # 1 epoch accuracy
        # score_dict[LOSS_DATA] = 0  # 1 epoch loss
        # score_dict[F1_DATA] = 0  # 1 epoch f1 score
        # score_dict[RECALL_DATA] = 0  # 1 epoch recall
        # score_dict[PRECISION_DATA] = 0  # 1 epoch precision
        # score_dict[AUC_DATA] = 0
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
            score_dict[ACC_DATA] += accuracy_score(y_true, y_pred)
            # accuracy 를 구하기 위해 전체 데이터 사이즈 더함
            total += y.size(0)
            # loss 값 더함
            score_dict[LOSS_DATA] += loss
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            score_dict[PRECISION_DATA] += precision
            score_dict[RECALL_DATA] += recall
            score_dict[F1_DATA] += f1_score(y_true, y_pred, average="weighted", zero_division=0)
            f1_total += 1
            # print(conf_matrix)
            conf_matrix = confusion_matrix(y_true, y_pred)
            TN = conf_matrix[0, 0]  # 정상인 기계를 정상이라 예측
            FP = conf_matrix[0, 1]
            FN = conf_matrix[1, 0]
            TP = conf_matrix[1, 1]  # 고장난 것을 고장이라 예측
            auc_score = roc_auc_score(y_true, y_pred)  # roc 커브
            score_dict[AUC_DATA] += auc_score

        for key, value in score_dict.items():
            score_dict[key] = (100 * value / f1_total)
        # acc_list.append(acc)
        # loss_list.append(loss_data)
        # f1_list.append(f1_data)
        if ep % 100 == 0:
            print(
                f"epoch : {ep}/{epoch}\t\tloss:{score_dict[LOSS_DATA]}\t\t accuracy:{score_dict[ACC_DATA]:.3f} \t\t "
                f"recall:{score_dict[RECALL_DATA]:.3f} \t\t precision:{score_dict[PRECISION_DATA]:.3f} \t\t "
                f"f1 score:{score_dict[F1_DATA]:.3f} \t\t auc score :{score_dict[AUC_DATA]:.3f} \t\t "
                f"TP:{TP} FN:{FN} FP:{FP} TN:{TN}")


def model_test():
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


def my_svm_model(x_train, y_train, x_test, kernal="rbf"):
    clf = SVC(kernel=kernal)  # 분류는 rbf 또는 poly
    clf.fit(x_train, y_train)
    return clf.predict(x_test), clf


def show_rating(y_pred, y_true):
    print(f"accuracy : {accuracy_score(y_true, y_pred)}")
    print(f"f1 score : {f1_score(y_true, y_pred)}")
    print(f"precision : {precision_score(y_true, y_pred)}")
    print(f"recall : {recall_score(y_true, y_pred)}")
    print(f"roc : {roc_auc_score(y_true, y_pred)}")
    print(f"confusion matrix :{confusion_matrix(y_true, y_pred)}")
    print(f"roc :{roc_auc_score(y_true, y_pred)}")


if __name__ == "__main__":
    if config.is_model_save:
        model_name = input("모델 이름을 입력하세요")

    # 1. 파일 읽기, x_data y_data 로 나눔
    df = read_file(config.dataset_file_path)
    x_data = df.iloc[:, :-2]  # y 값 제외하고 슬라이싱 , 고장단계도 제외(1,2,3 은 고장 유무가 0이고 4,5는 고장 유무가 1이기때문에 제거)
    y_data = df.iloc[:, -1]  # y 값 만 슬라이싱

    # df.info()  # type 및 자료 확인
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
    #  5. scaler
    x_train, x_test = my_robust_scaler(x_train, x_test)
    # ============================== make model =================================
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
    pred,clf = my_svm_model(x_train,y_train,x_test)
    show_rating(pred,y_test)
    epoch = 2000
    lr = 0.1
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    cost_func = nn.BCELoss()  # 이진분류기 때문에 binary cross entropy 사용
    model_train(train_loader=train_loader, model=model, epoch=epoch, optimizer=optimizer, cost_func=cost_func)
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
