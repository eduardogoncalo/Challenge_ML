import pandas as pd
from imblearn.pipeline import Pipeline


def predict_validation_dataset(
    model: Pipeline, data_validation: pd.DataFrame, target_feature: str
):

    y_real = data_validation[target_feature]
    y_pred = model.predict(data_validation)

    return y_real, y_pred
