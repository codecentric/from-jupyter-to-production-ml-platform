import mlflow
import pandas as pd
from dotenv import load_dotenv
from xgboost import Booster
from zenml import step
import xgboost as xgb

from titanicsurvivors.utils.experiment_tracking import get_experiment_tracker_name

load_dotenv()


@step(experiment_tracker=get_experiment_tracker_name())
def validate_xgb_model(model: Booster, inputs: pd.DataFrame, targets: pd.DataFrame):
    dtest = xgb.DMatrix(inputs)
    predictions = model.predict(dtest)

    predictions = [round(value) for value in predictions]

    # ... Please insert the calculation of the metrics here.

    print("Test accuracy:", accuracy)
    print("Test precision:", precision)
    print("Test recall:", recall)
    print("Test f1:", f1)
    mlflow.log_metrics(
        {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    )
