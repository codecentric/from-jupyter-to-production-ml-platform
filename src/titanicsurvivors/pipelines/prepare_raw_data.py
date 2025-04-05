import os

from dotenv import load_dotenv
from zenml import pipeline

from titanicsurvivors.settings import docker_settings, mlflow_settings
from titanicsurvivors.steps.raw_data import load_raw_data

load_dotenv()


@pipeline(
    settings={"docker": docker_settings, "experiment_tracker": mlflow_settings},
    name=f"Load_Raw_Data_{os.getenv('GROUP_NAME', 'Default')}",
)
def prepare_raw_data():
    raw_data_files = ["./data/train.csv"]
    load_raw_data(raw_data_files=raw_data_files)


if __name__ == "__main__":
    prepare_raw_data()
