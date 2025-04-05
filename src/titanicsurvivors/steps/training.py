import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from typing_extensions import Annotated
from zenml import step
import xgboost as xgb

from titanicsurvivors.utils.experiment_tracking import get_experiment_tracker_name

load_dotenv()


@step(experiment_tracker=get_experiment_tracker_name())
def train_xgb_classifier(
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    max_depth: int = 6,
    eta: float = 0.1,
    objective: str = "binary:logistic",
    eval_metric: str = "error",
) -> Annotated[
    xgb.Booster, f"<model_artifact_name>_{os.getenv('GROUP_NAME', 'Default')}"
]:  # ... Please add the name of the trained model artifact.
    # ... Please enable mlflow autologging for the training process.

    # ... Please add the provided training script.
