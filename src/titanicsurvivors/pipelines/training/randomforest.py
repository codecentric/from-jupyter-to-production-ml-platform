from zenml import pipeline
from zenml.client import Client

from titanicsurvivors.models import titanic_random_forest
from titanicsurvivors.steps.analysis import score_random_forest
from titanicsurvivors.steps.training import train_random_forest
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

pipeline_settings = {
    "experiment_tracker": MLFlowExperimentTrackerSettings(
        experiment_name="Train RandomForest Classifier"
    ),
}


@pipeline(
    model=titanic_random_forest,
    name="Train_RandomForestClassifier_for_Titanic_survivors",
    settings=pipeline_settings,
    enable_cache=False,
)
def train():
    client = Client()

    train_input = client.get_artifact_version(name_id_or_prefix="train_input")
    test_input = client.get_artifact_version(name_id_or_prefix="test_input")
    train_target = client.get_artifact_version(name_id_or_prefix="train_target")
    test_target = client.get_artifact_version(name_id_or_prefix="test_target")

    random_forest = train_random_forest(
        train_input=train_input,
        train_target=train_target,
    )
    score_random_forest(
        test_input=test_input, test_target=test_target, random_forest=random_forest
    )


if __name__ == "__main__":
    train()
