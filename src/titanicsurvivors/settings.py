import os

from zenml.config import DockerSettings
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)

from titanicsurvivors.utils.experiment_tracking import get_mlflow_azure_blob_env_secret
from dotenv import load_dotenv

load_dotenv()

docker_settings = DockerSettings(
    build_context_root=".",
    dockerfile="./Dockerfile",
    allow_including_files_in_images=True,
    install_stack_requirements=False,
    allow_download_from_code_repository=False,
    allow_download_from_artifact_store=False,
    environment={
        "AZURE_STORAGE_ACCESS_KEY": get_mlflow_azure_blob_env_secret(),
        "GROUP_NAME": os.getenv('GROUP_NAME', 'Default')
    },
    apt_packages=["git"],
)


mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=f"Titanic Survivors {os.getenv('GROUP_NAME', 'Default')}",
    nested=True,
)
