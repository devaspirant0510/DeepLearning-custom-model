from enum import Enum, auto
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE


class Constant(Enum):
    RANDOM_UNDER_SAMPLING = auto()
    RANDOM_OVER_SAMPLING = auto()
    SMOOTE = auto()
    ADASYN = auto()
    TOME_LINK = auto()


class BaseSampling:
    def __init__(self, x_data, y_data, model_type, random_state):
        self.x_data = x_data
        self.y_data = y_data
        self.x_result = None
        self.y_result = None
        self.run_sampling(model_type, random_state)

    def run_sampling(self, model_type, random_state):
        if model_type == Constant.RANDOM_OVER_SAMPLING:
            self.x_result, self.y_result = RandomOverSampler(random_state=random_state).fit_resample(self.x_data,
                                                                                                     self.y_data)
        elif model_type == Constant.RANDOM_UNDER_SAMPLING:
            self.x_result, self.y_result = RandomUnderSampler(random_state=random_state).fit_resample(self.x_data,
                                                                                                      self.y_data)
        elif model_type == Constant.SMOOTE:
            self.x_result, self.y_result = SMOTE(random_state=random_state).fit_resample(self.x_data, self.y_data)
        elif model_type == Constant.ADASYN:
            self.x_result, self.y_result = ADASYN(random_state=random_state).fit_resample(self.x_data, self.y_data)
        elif model_type == Constant.TOME_LINK:
            self.x_result, self.y_result = TomekLinks().fit_resample(self.x_data, self.y_data)
        else:
            raise Exception("해당 모델이 존재하지 않습니다.")

    def get_data(self):
        return self.x_result, self.y_result
