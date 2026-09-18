import os

from dotenv import load_dotenv
from zenml import pipeline
from zenml.client import Client
from zenml.config import ResourceSettings
from zenml.integrations.gcp.flavors.gcp_deployer_flavor import GCPDeployerSettings

from titanicsurvivors.models import titanic_xgboost
from titanicsurvivors.steps.prediction import predict_survival


load_dotenv()

DEFAULT_FEATURES = {
    "Age": 2.0,
    "Deck_0": 1.0,
    "Deck_1": 0.0,
    "Deck_2": 0.0,
    "Deck_3": 0.0,
    "Embarked_0": 1.0,
    "Embarked_1": 0.0,
    "Embarked_2": 0.0,
    "Family_Size_Grouped_0": 0.0,
    "Family_Size_Grouped_1": 0.0,
    "Family_Size_Grouped_2": 1.0,
    "Fare": 2.0,
    "Parch": 1.0,
    "Pclass_1": 1.0,
    "Pclass_2": 0.0,
    "Pclass_3": 0.0,
    "Sex_0": 1.0,
    "Sex_1": 0.0,
    "SibSp": 0.0,
    "Ticket_Frequency": 2.0,
    "Title_0": 1.0,
    "Title_1": 0.0,
    "Title_2": 0.0,
    "Title_3": 0.0,
    "is_married": 1.0,
}


@pipeline(
    model=titanic_xgboost,
    enable_cache=False,
    settings={
        "deployer": GCPDeployerSettings(use_secret_manager=False),
        "resources": ResourceSettings(cpu_count=1, memory="2GiB"),
    },
)
def model_deployment_pipeline(features: dict[str, float] = DEFAULT_FEATURES) -> str:
    model_artifact = Client().get_artifact_version(
        f"<model_artifact>_{os.getenv('GROUP_NAME', 'Default')}"
    )  # TODO: Use the model artifact name produced by the training pipeline.

    # TODO: Pass the artifact and request features to the prediction step.
    # TODO: Return the prediction so Cloud Run exposes it in the response.
    _ = (model_artifact, features, predict_survival)
    raise NotImplementedError("Complete the model deployment pipeline.")


if __name__ == "__main__":
    model_deployment_pipeline()
