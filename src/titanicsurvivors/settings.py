import os

from zenml.config import DockerSettings
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)

from dotenv import load_dotenv

load_dotenv()

docker_settings = DockerSettings(
    build_context_root=".",
    dockerfile="./Dockerfile",
    allow_including_files_in_images=False,
    allow_download_from_code_repository=False,
    allow_download_from_artifact_store=False,
    environment={"GROUP_NAME": os.getenv("GROUP_NAME", "Default")},
)


mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=f"Titanic Survivors {os.getenv('GROUP_NAME', 'Default')}",
    nested=True,
)
