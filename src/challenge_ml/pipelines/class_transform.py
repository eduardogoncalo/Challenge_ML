import logging
import pandas as pd
from typing import List, Any
from sklearn.base import BaseEstimator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CastFloat(BaseEstimator):
    type_transform = "float"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        for col in X.columns:
            X[col] = X[col].astype(self.type_transform)

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.transform(X)


class ReplaceStr(BaseEstimator):
    def __init__(self, inicial_value: str, final_value: any) -> None:
        self.inicial_value = inicial_value
        self.final_value = final_value

    def replace_any_value(self, X: pd.DataFrame) -> pd.DataFrame:

        for col in X.columns:
            X[col] = X[col].replace(self.inicial_value, self.final_value)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.replace_any_value(X)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.transform(X)
