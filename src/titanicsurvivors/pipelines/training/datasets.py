import os

from zenml import pipeline
from zenml.client import Client

from titanicsurvivors.steps.datapreparation import feature_transformation, split_data
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pipeline
def create_subsets_from_artifact(
    data_artifact_name: str = "data_w_features_feature_store",
):
    client = Client()
    data_w_features = client.get_artifact_version(name_id_or_prefix=data_artifact_name)
    encoded_data = feature_transformation(data_w_features=data_w_features)
    split_data(data=encoded_data)


if __name__ == "__main__":
    pipeline_args = {"config_path": os.path.join("./configs", "create_subsets.yaml")}
    create_subsets_from_artifact.with_options(**pipeline_args)()
