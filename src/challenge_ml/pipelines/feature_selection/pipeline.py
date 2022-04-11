from kedro.pipeline import Pipeline, node
from .nodes import create_ibmpipeline_featureselector, get_best_features
from ..classifier_ml.nodes import split_train_test, fit_model, randomizesearch_kfold


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_ibmpipeline_featureselector,
                inputs=[
                    "data_pre_2020",
                    "params:target_feature",
                    "params:kfold_FeatureSelector",
                    "params:features_to_get_params",
                ],
                outputs=["X_features", "y_features", "pre_pipe_FeatureSelector"],
                name="create_ibm_pipeline_feature_selector",
            ),
            node(
                func=split_train_test,
                inputs=[
                    "X_features",
                    "y_features",
                    "params:test_size_params",
                ],
                outputs=[
                    "X_train_features",
                    "X_test_features",
                    "y_train_features",
                    "y_test_features",
                ],
                name="split_test_train_dataset_feature_selector",
            ),
            node(
                func=fit_model,
                inputs=[
                    "pre_pipe_FeatureSelector",
                    "X_train_features",
                    "y_train_features",
                ],
                outputs="trained_pipeline_FeatureSelector",
                name="fit_pre_trained_pipe_FeatureSelector",
            ),
            node(
                func=get_best_features,
                inputs=[
                    "trained_pipeline_FeatureSelector",
                    "params:classifier_FeatureSelector",
                ],
                outputs="feature_list",
                name="get_feature_list",
            ),
        ]
    )
