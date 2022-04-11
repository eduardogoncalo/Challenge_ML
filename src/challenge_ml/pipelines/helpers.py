import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator


def cost_function_model(y, y_pred):
    """Used in RandomizedSearchCV at 'make_score'"""
    tp = np.where((y_pred == 1) & (y == 1), 10, 0)
    fp = np.where((y_pred == 1) & (y == 0), 10, 0)
    fn = np.where((y_pred == 0) & (y == 1), 500, 0)
    return np.sum([tp, fp, fn])  # add metric to model


def convert(y):
    """Used in RandomizedSearchCV at 'make_score'"""
    y_pred_converted = np.where((y == "pos"), 1, 0)
    return y_pred_converted


def true_falses_metrics_mlflow(y_test, y_pred_tunned):
    """tn, fp, fn, tp"""

    tf_np_list = confusion_matrix(y_test, y_pred_tunned).ravel().tolist()


class RandomForestSelctionFeature(BaseEstimator):
    def __init__(self, RandomForest: RandomForestClassifier):
        self.RandomForest = RandomForest

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        return self.RandomForest.feature_importances_


def feature_importance(model, X: pd.DataFrame):

    importances = pd.Series(data=model.feature_importances_, index=data.feature_names)

    # sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importância de cada feature')

