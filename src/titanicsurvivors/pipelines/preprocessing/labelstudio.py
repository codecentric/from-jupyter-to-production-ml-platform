from zenml import pipeline

from titanicsurvivors.steps.datacleaning import handle_missing_values
from titanicsurvivors.steps.features import add_features
from titanicsurvivors.steps.labelstudio import fetch_raw_titanic_data


@pipeline(name="Load and prepare dataset from LabelStudio")
def prepare_label_studio_data():
    raw_data = fetch_raw_titanic_data()
    filled_data = handle_missing_values(data=raw_data)
    add_features(data=filled_data)


if __name__ == "__main__":
    prepare_label_studio_data()
