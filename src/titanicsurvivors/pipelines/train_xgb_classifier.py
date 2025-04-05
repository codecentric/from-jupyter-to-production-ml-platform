
    
import os

from dotenv import load_dotenv
from zenml import pipeline
from zenml.client import Client

from titanicsurvivors.models import titanic_xgboost
from titanicsurvivors.steps.dataset import (
    feature_transformation,
    split_data_into_subset,
)
from titanicsurvivors.steps.training import train_xgb_classifier
from titanicsurvivors.steps.validation import validate_xgb_model
from titanicsurvivors.settings import docker_settings, mlflow_settings

load_dotenv()


@pipeline(
    model=titanic_xgboost,
    settings={"docker": docker_settings, "experiment_tracker": mlflow_settings},
    name=f"Train_Model_{os.getenv('GROUP_NAME', 'Default')}",
)
def train_xgb():
    client = Client()

    # ... Please add the name of the artifact, that we want to use here
    data_w_features = client.get_artifact_version(
        name_id_or_prefix=f"combined_features_{os.getenv('GROUP_NAME', 'Default')}"
    )

    encoded_data = feature_transformation(data_w_features=data_w_features)
    train_input, test_input, train_target, test_target = split_data_into_subset(
        data=encoded_data
    )
    xgb_model = train_xgb_classifier(
        inputs=train_input,
        targets=train_target,
        max_depth=50,
        eta=0.1,
        objective="binary:logistic",
        eval_metric="error",
    )
    validate_xgb_model(model=xgb_model, inputs=test_input, targets=test_target)


if __name__ == "__main__":
    train_xgb()
