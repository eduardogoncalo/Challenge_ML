from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ..class_transform import ReplaceStr, CastFloat
from imblearn.under_sampling import RandomUnderSampler
from ..helpers import cost_function_model, convert
from sklearn.model_selection import KFold, RandomizedSearchCV
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    confusion_matrix,
    make_scorer,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import PolynomialFeatures


def create_ibmpipeline(
    data_train: pd.DataFrame, target_feature: str, feature_list: list
):
    """In this function will be create a pipeline from iblearn that will be fit. Using the proprietis of the pipeline
    it's possible build personalized steps. Mode details at README"""

    X = data_train[feature_list]
    y = data_train[target_feature]

    numeric_features = X.columns

    numeric_transformer = Pipeline(
        steps=[
            ("replace_str", ReplaceStr(inicial_value="na", final_value=np.nan)),
            ("cast_feature", CastFloat()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("rus", RandomUnderSampler()),
            ("poly", PolynomialFeatures()),
            ("classifier", GradientBoostingClassifier()),
        ]
    )

    return X, y, pipe


def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size_params: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_params, random_state=42
    )

    return X_train, X_test, y_train, y_test


def fit_model(
    pre_trained_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame
):
    return pre_trained_pipeline.fit(X_train, y_train)


def randomizesearch_kfold(
    trained_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_inter_params: int,
    error_score_params: int,
    verbose_params: int,
    n_jobs_params: int,
):

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # TO do
    model_params = {
        "classifier__n_estimators": list(
            map(int, np.linspace(start=10, stop=1000, num=500))
        ),
        "classifier__max_depth": list(map(int, np.linspace(start=1, stop=100, num=50))),
        "classifier__learning_rate": stats.loguniform(0.05, 0.3),
        "preprocessor__num__imputer__strategy": ["mean", "median"],
    }

    model_otimizated = RandomizedSearchCV(
        trained_pipeline,
        param_distributions=model_params,
        cv=kfold,
        n_iter=n_inter_params,
        scoring=make_scorer(cost_function_model, greater_is_better=False),
        error_score=error_score_params,
        verbose=verbose_params,
        n_jobs=-n_jobs_params,
    )

    return model_otimizated.fit(X_train, y_train)


def predict_model(pipeline_tunned: Pipeline, X_test: pd.DataFrame):

    y_pred_tunned = pipeline_tunned.predict(X_test)

    return y_pred_tunned


def model_performim_mlflow(y_test: pd.DataFrame, y_pred_tunned: pd.DataFrame):

    mlflow_metrics = {
        "accuracy": {
            "value": accuracy_score(convert(y_test), convert(y_pred_tunned)),
            "step": 1,
        },
        "recall": {
            "value": recall_score(convert(y_test), convert(y_pred_tunned)),
            "step": 1,
        },
        "total_costs": {
            "value": cost_function_model(convert(y_test), convert(y_pred_tunned)),
            "step": 1,
        }
    }

    return mlflow_metrics


def plot_matrix_confusion(y_test, y_pred_tunned, cmap_="viridis"):

    cm = confusion_matrix(y_test, y_pred_tunned)
    cmd = ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"])
    cmd.plot(cmap=cmap_)

    return cmd.figure_
