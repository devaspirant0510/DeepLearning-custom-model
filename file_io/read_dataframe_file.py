import pandas as pd
import os.path
from enum import Enum, auto


class Constant(Enum):
    CSV_FILE = auto()
    EXCEL_FILE = auto()


class ReadDataframeFile:
    def __init__(self, file_path: str, file_type: Constant, encoding=None):
        self.dataframe = None
        if os.path.isfile(file_path):
            self.dataframe = self.__read_dataframe(file_path, file_type, encoding)
        else:
            raise Exception("해당 파일이 존재하지 않습니다.")

    def __read_dataframe(self, path:str, file_type:Constant, encoding=None) -> pd.DataFrame:
        if file_type == Constant.CSV_FILE:
            return pd.read_csv(path, encoding=encoding)
        elif file_type == Constant.EXCEL_FILE:
            return pd.read_excel(path)
        else:
            raise Exception("존재하지 않는 file_type 입니다")

    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe
