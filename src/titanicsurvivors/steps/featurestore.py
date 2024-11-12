from datetime import datetime
from typing import List, Annotated

from titanicsurvivors.utils.data import DataFrameColumns
from zenml import step
from zenml.client import Client
import pandas as pd

features_of_service: List[str] = [
    "titanic_passenger_stats:Pclass",
    "titanic_passenger_stats:Name",
    "titanic_passenger_stats:Sex",
    "titanic_passenger_stats:Age",
    "titanic_passenger_stats:SibSp",
    "titanic_passenger_stats:Parch",
    "titanic_passenger_stats:Ticket",
    "titanic_passenger_stats:Fare",
    "titanic_passenger_stats:Deck",
    "titanic_passenger_stats:Embarked",
    "titanic_passenger_stats:Survived",
    "additional_and_grouped_features:Fare_Category",
    "additional_and_grouped_features:Age_Category",
    "additional_and_grouped_features:Family_Size_Grouped",
    "additional_and_grouped_features:Ticket_Frequency",
    "additional_and_grouped_features:Title",
    "additional_and_grouped_features:is_married",
]


@step()
def get_historical_features_from_feature_store(
    full_feature_names: bool = False,
) -> Annotated[pd.DataFrame, "data_w_features_feature_store"]:
    entity_dict = {
        DataFrameColumns.PASSENGER_ID.value: [index for index in range(1, 800)],
        DataFrameColumns.EVENT_TIMESTAMP.value: [datetime.now() for _ in range(1, 800)],
    }

    feature_store = Client().active_stack.feature_store
    entity_df = pd.DataFrame.from_dict(entity_dict)

    data = feature_store.get_historical_features(
        entity_df=entity_df,
        features=features_of_service,
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
