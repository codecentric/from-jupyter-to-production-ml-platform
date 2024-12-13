from dotenv import load_dotenv
from zenml import pipeline

from titanicsurvivors.steps.featurestore import (
    get_historical_features_from_feature_store,
)

load_dotenv()

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pipeline()
def data_from_feature_store():
    get_historical_features_from_feature_store()


if __name__ == "__main__":
    data_from_feature_store()
