from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from ..class_transform import ReplaceStr, CastFloat
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector


def create_ibmpipeline_featureselector(
    data_train: pd.DataFrame,
    target_feature: str,
    kfold,
    n_features_to_select_params: int,
):
    """In this function will be create a pipeline from iblearn that will be fit. Using the proprietis of the pipeline
    it's possible build personalized steps. Mode details at README"""

    X = data_train.drop(columns=target_feature)
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
            (
                "classifier",
                SequentialFeatureSelector(
                    RandomForestClassifier(),
                    n_features_to_select=n_features_to_select_params,
                    cv=kfold,
                ),
            ),
        ]
    )

    return X, y, pipe


def get_best_features(
    trained_pipeline_FeatureSelector: Pipeline, classifier_FeatureSelector: str
) -> list:
    features_bool = trained_pipeline_FeatureSelector.named_steps[
        classifier_FeatureSelector
    ].support_
    features_trainned = trained_pipeline_FeatureSelector.feature_names_in_

    return list(features_trainned[features_bool])
