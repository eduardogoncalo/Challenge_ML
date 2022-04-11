from kedro.pipeline import Pipeline, node
from ..classifier_ml.nodes import (
    model_performim_mlflow,
    plot_matrix_confusion,
)
from .nodes import predict_validation_dataset


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=predict_validation_dataset,
                inputs=[
                    "pipeline_tunned",
                    "data_2020",
                    "params:target_feature",
                ],
                outputs=[
                    "y_real_validation",
                    "y_pred_validation",
                ],
                name="predict_validation_dataset_",
            ),
            node(
                func=model_performim_mlflow,
                inputs=[
                    "y_real_validation",
                    "y_pred_validation",
                ],
                outputs="mlflow_classify_validation",
                name="mlflow_metrics_validation",
            ),
            node(
                func=plot_matrix_confusion,
                inputs=[
                    "y_real_validation",
                    "y_pred_validation",
                    "params:cmap_validation",
                ],
                outputs="confusion_matrix_validation",
                name="confusion_matrix_mlflow_validation",
            ),
        ]
    )
