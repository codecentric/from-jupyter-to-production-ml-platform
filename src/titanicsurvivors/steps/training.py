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
    xgb.Booster, f"xgb_model_{os.getenv('GROUP_NAME', 'Default')}"
]:  # ... Please add the name of the trained model artifact.
    # ... Please enable mlflow autologging for the training process.
    mlflow.autolog()

    # ... Please add the provided training script.
    dtrain = xgb.DMatrix(inputs, label=targets)
    params = {
        "max_depth": max_depth,
        "eta": eta,
        "objective": objective,
        "eval_metric": eval_metric,
    }
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        nfold=5,
        metrics=["error"],
        early_stopping_rounds=10,
        stratified=True,
    )

    best_iteration = cv_results["test-error-mean"].idxmin()
    best_model = xgb.train(params, dtrain, num_boost_round=best_iteration + 1)
    return best_model
