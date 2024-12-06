import logging
from typing import Optional, Annotated, Tuple
from xgboost import XGBClassifier

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from zenml import step, get_step_context, log_metadata

from titanicsurvivors.models import titanic_random_forest, titanic_xgboost


@step(model=titanic_random_forest)
def score_random_forest(
    test_input: pd.DataFrame,
    test_target: pd.Series,
    random_forest: Optional[RandomForestClassifier] = None,
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
]:
    model = get_step_context().model
    if random_forest is None:
        random_forest = model.get_model_artifact("random_forest").load()

    logging.info(f"Score: {random_forest.score(test_input, test_target)}")
    predicted_targets = random_forest.predict(test_input)

    accuracy = float(accuracy_score(test_target, predicted_targets))
    precision = float(precision_score(test_target, predicted_targets))
    recall = float(recall_score(test_target, predicted_targets))

    logging.info(f"accuracy: {accuracy}")
    logging.info(f"precision: {precision}")
    logging.info(f"recall: {recall}")

    log_metadata(
        model_name=model.name,
        model_version=model.version,
        metadata={"accuracy": accuracy, "precision": precision, "recall": recall},
    )
    return accuracy, precision, recall


@step(model=titanic_xgboost)
def score_xgb_classifier(
    test_input: pd.DataFrame,
    test_target: pd.Series,
    xgb_classifier: Optional[XGBClassifier] = None,
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
]:
    model = get_step_context().model
    if xgb_classifier is None:
        xgb_classifier = model.get_model_artifact("xgboost_classifier").load()

    logging.info(f"Score: {xgb_classifier.score(test_input, test_target)}")
    predicted_targets = xgb_classifier.predict(test_input)

    accuracy = float(accuracy_score(test_target, predicted_targets))
    precision = float(precision_score(test_target, predicted_targets))
    recall = float(recall_score(test_target, predicted_targets))

    logging.info(f"accuracy: {accuracy}")
    logging.info(f"precision: {precision}")
    logging.info(f"recall: {recall}")
    log_metadata(
        model_name=model.name,
        model_version=model.version,
        metadata={"accuracy": accuracy, "precision": precision, "recall": recall},
    )
    return accuracy, precision, recall
