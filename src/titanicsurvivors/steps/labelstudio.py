from datetime import datetime
from typing import Annotated, List

import pandas as pd
from zenml import step
from zenml.client import Client

from titanicsurvivors.utils.data import DataFrameColumns


@step(enable_cache=False)
def fetch_raw_titanic_data() -> Annotated[pd.DataFrame, "titanic_raw"]:
    annotator = Client().active_stack.annotator
    from zenml.integrations.label_studio.annotators.label_studio_annotator import (
        LabelStudioAnnotator,
    )

    if not isinstance(annotator, LabelStudioAnnotator):
        raise TypeError("This step can only be used with the Label Studio annotator.")
    dataset = annotator.get_dataset(dataset_name="titanic")
    data = label_to_dataframe(dataset.tasks)
    return data


def label_to_dataframe(tasks: List[dict]) -> pd.DataFrame:
    passenger_data = []
    for task in tasks:
        task_data = task["data"]
        passenger = {DataFrameColumns.PASSENGER_ID.value: task_data["ID"]}
        passenger |= task_data["p_data"]
        latest_annotation = get_latest_annotation(task["annotations"])
        passenger[DataFrameColumns.SURVIVED.value] = (
            1 if latest_annotation["result"][0]["value"]["choices"] == "Survived" else 0
        )
        for key, value in passenger.items():
            if value == -1:
                passenger[key] = None
        passenger_data.append(passenger)
    return pd.DataFrame(passenger_data)


def get_latest_annotation(annotations: List[dict]) -> dict:
    latest_annotation = None
    datetime_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    for annotation in annotations:
        if latest_annotation is None:
            latest_annotation = annotation
            continue
        if datetime.strptime(
            annotation["updated_at"], datetime_format
        ) > datetime.strptime(latest_annotation["updated_at"], datetime_format):
            latest_annotation = annotation
    return latest_annotation
