import os
import tempfile
from os.path import join

import mlflow
import pandas as pd
from dotenv import load_dotenv
from typing_extensions import Annotated
from zenml import step

from titanicsurvivors.utils.experiment_tracking import get_experiment_tracker_name

load_dotenv()


@step(experiment_tracker=get_experiment_tracker_name(), enable_cache=False)
def load_raw_data(
    raw_data_files: list[str],
) -> Annotated[pd.DataFrame, f"raw_data_{os.getenv('GROUP_NAME', 'Default')}"]:
    raw_data_df = pd.DataFrame()
    for raw_data_file in raw_data_files:
        df = pd.read_csv(raw_data_file, header=0, sep=",")
        raw_data_df = pd.concat([raw_data_df, df], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_data_path = join(tmp_dir, "raw_data.csv")
        raw_data_df.to_csv(tmp_data_path, index=False)
        mlflow.log_param("number_of_rows", len(raw_data_df))
        mlflow.log_artifact(local_path=tmp_data_path, artifact_path="data")
    return raw_data_df
