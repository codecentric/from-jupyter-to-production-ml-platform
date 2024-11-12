from zenml import pipeline
from zenml.client import Client

from titanicsurvivors.models import titanic_random_forest
from titanicsurvivors.steps.analysis import score_random_forest
from titanicsurvivors.steps.training import train_random_forest
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from dotenv import load_dotenv

from zenml.integrations.bentoml.steps import (
    bento_builder_step,
    bentoml_model_deployer_step,
)
import zenml

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
    bento = bento_builder_step(
        model=random_forest,
        model_name=titanic_random_forest.name,  # Name of the model
        model_type="sklearn",  # Type of the model (pytorch, tensorflow, sklearn, xgboost..)
        service="deployment:TitanicRFService",  # Path to the service file within zenml repo
        labels={  # Labels to be added to the bento bundle
            "framework": "sklearn",
            "dataset": "titanic",
            "zenml_version": zenml.__version__,
        },
        exclude=["data"],  # Exclude files from the bento bundle
        python={
            "packages": ["zenml", "scikit-learn"],
        },  # Python package requirements of the model
    )

    _ = bentoml_model_deployer_step(
        bento=bento,
        model_name=titanic_random_forest.name,
        port=3000,  # Name of the model
        deployment_type="container",
        image="titanicsurvivors/rfclassifier",
        image_tag=titanic_random_forest.version,
        platform="linux/amd64",
    )
    score_random_forest(
        test_input=test_input, test_target=test_target, random_forest=random_forest
    )


if __name__ == "__main__":
    train()
