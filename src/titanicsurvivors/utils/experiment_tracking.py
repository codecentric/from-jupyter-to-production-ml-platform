import os
import tempfile

import mlflow
from zenml.client import Client


def get_experiment_tracker_name() -> str | None:
    experiment_tracker = Client().active_stack.experiment_tracker
    orchestrator = Client().active_stack.orchestrator

    if experiment_tracker is not None:
        if orchestrator.flavor == "default":
            os.environ["AZURE_STORAGE_ACCESS_KEY"] = get_mlflow_azure_blob_env_secret()
        return experiment_tracker.name
    return None


def get_mlflow_azure_blob_env_secret() -> str:
    secret = Client().get_secret("azure_blob_access_key")
    return secret.secret_values["access_key"]
