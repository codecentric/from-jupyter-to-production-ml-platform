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
    inputs = xgb.DMatrix(pd.DataFrame([features]))
    probability = float(model.predict(inputs)[0])
    return "Survived" if probability >= 0.5 else "Died"
