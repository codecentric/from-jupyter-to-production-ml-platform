import os

from dotenv import load_dotenv
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)
from zenml.config import DockerSettings


load_dotenv()

docker_orchestrator_settings = LocalDockerOrchestratorSettings(
    run_args={"network_mode": "host"}
)

docker_settings = DockerSettings(
    # replicate_local_python_environment="pip_freeze",
    environment={
        "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ACCESS_KEY"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_SECRET_KEY"),
        "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        "ZENML_STORE_URL": "http://localhost:8080",
    },
    apt_packages=["git"],
)
