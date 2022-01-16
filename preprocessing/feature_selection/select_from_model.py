import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


class BaseSelectFromModel:
    def __init__(self, x_data: pd.DataFrame, y_data: pd.DataFrame, n_estimators: int):
        self.x_data = x_data
        self.y_data = y_data
        self.sel = SelectFromModel(RandomForestClassifier(n_estimators=n_estimators))
        self.sel.fit(x_data, y_data)

    def get_feature_size(self):
        print("AD",self.x_data.columns[self.sel.get_support()])
        return len(list(self.x_data.columns[self.sel.get_support()]))

    def get_data(self):
        columns = self.x_data.columns[self.sel.get_support()]
        print(columns)
        return self.x_data.loc[:, columns]
