from zenml import pipeline
from zenml.client import Client

from titanicsurvivors.models import titanic_xgboost
from titanicsurvivors.steps.analysis import score_xgb_classifier
from titanicsurvivors.steps.training import train_xgboost
from dotenv import load_dotenv
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)

load_dotenv()

pipeline_settings = {
    "experiment_tracker": MLFlowExperimentTrackerSettings(
        experiment_name="TrainXGBClassifier", nested=True
    ),
}


@pipeline(
    model=titanic_xgboost,
    name="Train XGBClassifier for Titanic Survivors",
    settings=pipeline_settings,
)
def train():
    client = Client()

    train_input = client.get_artifact_version(name_id_or_prefix="train_input")
    test_input = client.get_artifact_version(name_id_or_prefix="test_input")
    train_target = client.get_artifact_version(name_id_or_prefix="train_target")
    test_target = client.get_artifact_version(name_id_or_prefix="test_target")

    xgb_classifier = train_xgboost(
        train_input=train_input,
        train_target=train_target,
    )
    score_xgb_classifier(
        test_input=test_input, test_target=test_target, xgb_classifier=xgb_classifier
    )


if __name__ == "__main__":
    train()
