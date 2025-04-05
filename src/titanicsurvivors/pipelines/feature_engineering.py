import os

from dotenv import load_dotenv
from zenml import pipeline
from zenml.client import Client

from titanicsurvivors.steps.data_cleaning import handle_missing_values
from titanicsurvivors.steps.feature_engineering.age import divide_age_in_bins
from titanicsurvivors.steps.feature_engineering.common import combine_features
from titanicsurvivors.steps.feature_engineering.family_size import (
    add_family_size_feature,
)
from titanicsurvivors.steps.feature_engineering.fare import divide_fare_in_bins
from titanicsurvivors.steps.feature_engineering.ticket import (
    add_ticket_frequency_feature,
)
from titanicsurvivors.steps.feature_engineering.title import add_title_feature
from titanicsurvivors.settings import docker_settings, mlflow_settings

load_dotenv()


@pipeline(
    settings={"docker": docker_settings, "experiment_tracker": mlflow_settings},
    name=f"Feature_Engineering_{os.getenv('GROUP_NAME', 'Default')}",
)
def add_features_to_dataset():
    client = Client()
    raw_data = client.get_artifact_version(
        f"raw_data_{os.getenv('GROUP_NAME', 'Default')}"
    )
    data_without_missing_data = handle_missing_values(raw_data)
    data_binned_age, age_categories = divide_age_in_bins(data=data_without_missing_data)
    data_family_size = add_family_size_feature(data=data_without_missing_data)
    data_binned_fare, fare_categories = divide_fare_in_bins(
        data=data_without_missing_data
    )
    data_w_ticket_frequency = add_ticket_frequency_feature(
        data=data_without_missing_data
    )
    data_w_title = add_title_feature(data=data_without_missing_data)
    combine_features(
        age_data=data_binned_age,
        family_data=data_family_size,
        fare_data=data_binned_fare,
        ticket_data=data_w_ticket_frequency,
        title_data=data_w_title,
    )


if __name__ == "__main__":
    add_features_to_dataset()
