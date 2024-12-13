from random import sample
from typing import Annotated

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from zenml import step, ArtifactConfig, get_step_context, log_metadata
from zenml.integrations.sklearn.materializers import SklearnMaterializer
from zenml.client import Client

import mlflow

from titanicsurvivors.models import titanic_random_forest, titanic_xgboost
from xgboost import XGBClassifier

experiment_tracker = Client().active_stack.experiment_tracker


@step(
    model=titanic_random_forest,
    output_materializers={"random_forest": SklearnMaterializer},
    enable_cache=False,
    experiment_tracker="mlflow",
)
def train_random_forest(
    train_input: pd.DataFrame,
    train_target: pd.Series,
) -> Annotated[
    RandomForestClassifier,
    ArtifactConfig(name="random_forest", artifact_type="ModelArtifact"),
]:
    mlflow.sklearn.autolog(log_models=False)
    param_dist = {
        "n_estimators": sample(range(50, 100), 5),
        "max_depth": sample(range(1, 10), 5),
        "bootstrap": [True, False],
    }
    random_forest = RandomForestClassifier()
    rand_search = RandomizedSearchCV(
        random_forest, param_distributions=param_dist, n_iter=5
    )
    model = get_step_context().model

    rand_search.fit(train_input, train_target)
    best_random_forest = rand_search.best_estimator_
    feature_importance = pd.Series(
        best_random_forest.feature_importances_, index=train_input.columns
    ).sort_values(ascending=False)

    log_metadata(
        model_name=model.name,
        model_version=model.version,
        metadata={
            "feature_importance": feature_importance.to_dict(),
            "Best params": rand_search.best_params_,
        },
    )
    mlflow.sklearn.log_model(random_forest, artifact_path=model.name)

    return best_random_forest


@step(model=titanic_xgboost, experiment_tracker="mlflow", enable_cache=False)
def train_xgboost(
    train_input: pd.DataFrame,
    train_target: pd.Series,
) -> Annotated[
    XGBClassifier,
    ArtifactConfig(name="xgboost_classifier", artifact_type="ModelArtifact"),
]:
    mlflow.xgboost.autolog(log_models=False)

    param_dist = {
        "n_estimators": range(8, 50),
        "max_depth": range(6, 20),
        "learning_rate": [0.4, 0.45, 0.5, 0.55, 0.6],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1],
    }
    xgb_classifier = XGBClassifier()
    rand_search = RandomizedSearchCV(
        param_distributions=param_dist,
        estimator=xgb_classifier,
        scoring="accuracy",
        verbose=0,
        n_iter=5,
        cv=4,
    )
    model = get_step_context().model

    rand_search.fit(train_input, train_target)
    best_xgb_classifier = rand_search.best_estimator_
    feature_importance = pd.Series(
        best_xgb_classifier.feature_importances_, index=train_input.columns
    ).sort_values(ascending=False)

    log_metadata(
        model_name=model.name,
        model_version=model.version,
        metadata={
            "feature_importance": feature_importance.to_dict(),
            "Best params": rand_search.best_params_,
        },
    )
    mlflow.xgboost.log_model(best_xgb_classifier, artifact_path=model.name)

    return best_xgb_classifier
