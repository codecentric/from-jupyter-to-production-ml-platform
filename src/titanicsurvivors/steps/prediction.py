from typing import Annotated

import pandas as pd
import xgboost as xgb
from zenml import step


@step
def predict_survival(
    model: xgb.Booster,
    features: dict[str, float],
) -> Annotated[str, "prediction"]:
    """Return the Titanic survival prediction for one encoded passenger."""
    # TODO: Convert features to an XGBoost DMatrix and calculate the prediction.
    # TODO: Return "Survived" or "Died" based on the model probability.
    _ = (model, features, pd, xgb)
    raise NotImplementedError("Implement the prediction step.")
