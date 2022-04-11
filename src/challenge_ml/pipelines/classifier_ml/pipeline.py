from kedro.pipeline import Pipeline, node
from .nodes import (
    create_ibmpipeline,
    split_train_test,
    fit_model,
    randomizesearch_kfold,
    model_performim_mlflow,
    plot_matrix_confusion,
    predict_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_ibmpipeline,
                inputs=["data_pre_2020", "params:target_feature", "feature_list"],
                outputs=["X", "y", "pre_trained_pipeline"],
                name="create_ibm_pipeline",
            ),
            node(
                func=split_train_test,
                inputs=[
                    "X",
                    "y",
                    "params:test_size_params",
                ],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                ],
                name="split_test_train_dataset",
            ),
            node(
                func=fit_model,
                inputs=[
                    "pre_trained_pipeline",
                    "X_train",
                    "y_train",
                ],
                outputs="trained_pipeline",
                name="fit_pre_trained_pipeline",
            ),
            node(
                func=randomizesearch_kfold,
                inputs=[
                    "trained_pipeline",
                    "X_train",
                    "y_train",
                    "params:n_inter_params",
                    "params:error_score_params",
                    "params:verbose_params",
                    "params:n_jobs_params",
                ],
                outputs="pipeline_tunned",
                name="improve_pipeline_randomizesearch",
            ),
            node(
                func=predict_model,
                inputs=[
                    "pipeline_tunned",
                    "X_test",
                ],
                outputs="y_pred_tunned",
                name="predict_tunned_model",
            ),
            node(
                func=model_performim_mlflow,
                inputs=[
                    "y_test",
                    "y_pred_tunned",
                ],
                outputs="mlflow_classify",
                name="mlflow_metrics",
            ),
            node(
                func=plot_matrix_confusion,
                inputs=["y_test", "y_pred_tunned", "params:cmap_test"],
                outputs="confusion_matrix",
                name="confusion_matrix_mlflow",
            ),
        ]
    )
