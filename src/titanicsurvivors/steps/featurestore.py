from datetime import datetime
from typing import Annotated, Optional

import pandas as pd
from feast import FeatureService
from zenml import step
from zenml.client import Client
from zenml.integrations.feast.feature_stores import FeastFeatureStore

from titanicsurvivors.utils.data import DataFrameColumns


@step()
def get_historical_features_from_feature_store(
    full_feature_names: bool = False,
) -> Annotated[pd.DataFrame, "data_w_features_feature_store"]:
    entity_dict = {
        DataFrameColumns.PASSENGER_ID.value: [index for index in range(1, 800)],
        DataFrameColumns.EVENT_TIMESTAMP.value: [datetime.now() for _ in range(1, 800)],
    }

    feature_store = Client().active_stack.feature_store
    if not feature_store:
        raise AttributeError(
            "The Feast feature store component is not available. "
            "Please make sure that the Feast stack component is registered as part of your current active stack."
        )
    if not isinstance(feature_store, FeastFeatureStore):
        raise ValueError("This pipeline step only supports one Feast FeatureStore.")

    feature_services = feature_store.get_feature_services()
    titanic_feature_service: Optional[FeatureService] = None
    for feature_service in feature_services:
        if feature_service.name == "titanic_classifier":
            titanic_feature_service = feature_service
            break

    if titanic_feature_service is None:
        raise ValueError(
            "The expected FeatureService “titanic_classifier” is not available in "
            "the Feast Feature Store."
        )
    entity_df = pd.DataFrame.from_dict(entity_dict)

    data = feature_store.get_historical_features(
        entity_df=entity_df,
        features=titanic_feature_service,
        full_feature_names=full_feature_names,
    )
    data[DataFrameColumns.AGE.value] = data[DataFrameColumns.AGE.value]
    data[DataFrameColumns.FARE.value] = data[DataFrameColumns.FARE_CATEGORY.value]
    data = data.drop(
        columns=[
            DataFrameColumns.FARE_CATEGORY.value,
            DataFrameColumns.AGE_CATEGORY.value,
        ]
    )
    return data
