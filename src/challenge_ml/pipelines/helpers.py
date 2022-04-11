import numpy as np


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
