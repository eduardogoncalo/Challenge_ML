"""Project pipelines."""
"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from challenge_ml.pipelines import classifier_ml as cml
from challenge_ml.pipelines import feature_selection as fs
from challenge_ml.pipelines import validation_results as vr


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    feature_selection = fs.create_pipeline()
    classifier_ml = cml.create_pipeline()
    validation_results = vr.create_pipeline()

    return {
        "fs": feature_selection,
        "cml": classifier_ml,
        "vr": validation_results,
        "full_train": feature_selection + classifier_ml,
        "full_pipeline_train_validation": feature_selection
        + classifier_ml
        + validation_results,
    }
