# from datetime import timedelta
# from typing import Any
#
# import numpy as np
#
# import pandas as pd
# from feast import (
#     Entity,
#     FeatureView,
#     Project,
#     Field,
#     FeatureService,
# )
# from feast.on_demand_feature_view import on_demand_feature_view
#
# from feast.types import String, Int32, Float32, Int64, Bool
# from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
#     PostgreSQLSource,
# )
# from titanicsurvivors.steps.features import (
#     bin_fare,
#     bin_age,
#     add_family_size,
#     group_family_size,
#     add_ticket_frequency,
#     add_title,
#     add_is_married,
#     group_titles,
# )
#
# from titanicsurvivors.utils.data import DataFrameColumns
#
# project = Project(
#     name="titanicsurvivors",
#     description="This project creates fraud detection for credit card transactions.",
# )
#
# passenger_entity = Entity(
#     name="passenger",
#     join_keys=["PassengerId"],
# )
#
#
# titanic_train_data_source = PostgreSQLSource(
#     name="titanic_train_data_source",
#     query="SELECT * FROM titanic",
#     timestamp_field="event_timestamp",
#     created_timestamp_column="created",
# )
#
#
# titanic_passenger_stats = FeatureView(
#     # The unique name of this feature view. Two feature views in a single
#     # project cannot have the same name
#     name="titanic_passenger_stats",
#     entities=[passenger_entity],
#     ttl=timedelta(days=1),
#     # The list of features defined below act as a schema to both define features
#     # for both materialization of features into a store, and are used as references
#     # during retrieval for building a training dataset or serving features
#     schema=[
#         Field(name=DataFrameColumns.TICKET_CLASS.value, dtype=Int32),
#         Field(name=DataFrameColumns.NAME.value, dtype=String),
#         Field(name=DataFrameColumns.SEX.value, dtype=String),
#         Field(name=DataFrameColumns.AGE.value, dtype=Float32),
#         Field(name=DataFrameColumns.NUM_OF_SIBLINGS_OR_SPOUSES.value, dtype=Int32),
#         Field(name=DataFrameColumns.NUM_OF_PARENTS_OR_CHILDREN.value, dtype=Int32),
#         Field(name=DataFrameColumns.TICKET_NUMBER.value, dtype=String),
#         Field(name=DataFrameColumns.FARE.value, dtype=Float32),
#         Field(name=DataFrameColumns.DECK.value, dtype=String),
#         Field(name=DataFrameColumns.PORT_OF_EMBARKATION.value, dtype=String),
#         Field(name=DataFrameColumns.SURVIVED.value, dtype=Bool),
#     ],
#     online=False,
#     source=titanic_train_data_source,
#     # Tags are user defined key/value pairs that are attached to each
#     # feature view
#     tags={"team": "titanic"},
# )
#
#
# @on_demand_feature_view(
#     sources=[titanic_passenger_stats],
#     schema=[
#         Field(name=DataFrameColumns.FARE_CATEGORY.value, dtype=Int64),
#         Field(name=DataFrameColumns.AGE_CATEGORY.value, dtype=Int64),
#         Field(name=DataFrameColumns.FAMILY_SIZE_GROUPED.value, dtype=String),
#         Field(name=DataFrameColumns.TICKET_FREQUENCY.value, dtype=Int64),
#         Field(name=DataFrameColumns.TITLE.value, dtype=String),
#         Field(name=DataFrameColumns.IS_MARRIED.value, dtype=Int64),
#     ],
#     mode="python",
# )
# def additional_and_grouped_features(inputs: pd.DataFrame) -> dict[str, Any]:
#     store_init = False
#     inputs = pd.DataFrame(inputs)
#     if len(inputs) == 1 and inputs[DataFrameColumns.NAME.value][0] == "hello world":
#         store_init = True  #
#     df = pd.DataFrame()
#     binned_fare, fare_categories = bin_fare(data=inputs, store_init=store_init)
#     df[DataFrameColumns.FARE_CATEGORY.value] = binned_fare.cat.codes
#     binned_age, age_categories = bin_age(data=inputs, store_init=store_init)
#     df[DataFrameColumns.AGE_CATEGORY.value] = binned_age.cat.codes
#
#     df[DataFrameColumns.FAMILY_SIZE.value] = add_family_size(data=inputs)
#     df[DataFrameColumns.FAMILY_SIZE_GROUPED.value] = group_family_size(data=df)
#     df = df.drop(columns=[DataFrameColumns.FAMILY_SIZE.value])
#     df[DataFrameColumns.TICKET_FREQUENCY.value] = add_ticket_frequency(data=inputs)
#     df[DataFrameColumns.TITLE.value] = add_title(data=inputs, store_init=store_init)
#     df[DataFrameColumns.IS_MARRIED.value] = add_is_married(data=df)
#     df[DataFrameColumns.IS_MARRIED.value] = df[
#         DataFrameColumns.IS_MARRIED.value
#     ].astype(np.int64)
#     df[DataFrameColumns.TITLE.value] = group_titles(data=df)
#     return df.to_dict(orient="list")
#
#
# feature_service = FeatureService(
#     name="titanic_classifier",
#     features=[titanic_passenger_stats, additional_and_grouped_features],
# )


from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from feast import Entity, FeatureView, Project, Field, FeatureService
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import String, Int32, Float32, Int64, Bool
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from titanicsurvivors.steps.features import (
    bin_fare,
    bin_age,
    add_family_size,
    group_family_size,
    add_ticket_frequency,
    add_title,
    add_is_married,
    group_titles,
)
from titanicsurvivors.utils.data import DataFrameColumns


def create_project() -> Project:
    """Create and return a Feast project for Titanic survivors.

    Returns:
        Feast project instance.
    """
    return Project(
        name="titanicsurvivors",
        description="This project creates fraud detection for credit card transactions.",
    )


def create_passenger_entity() -> Entity:
    """Create and return a Feast entity for Titanic passengers.

    Returns:
        Feast entity instance.
    """
    return Entity(
        name="passenger",
        join_keys=["PassengerId"],
    )


def create_postgresql_source() -> PostgreSQLSource:
    """Create and return a PostgreSQL source for Titanic data.

    Returns:
        The PostgreSQL source instance.
    """
    return PostgreSQLSource(
        name="titanic_train_data_source",
        query="SELECT * FROM titanic",
        timestamp_field=DataFrameColumns.EVENT_TIMESTAMP.value,
        created_timestamp_column="created",
    )


def create_titanic_passenger_stats(
    entity: Entity, source: PostgreSQLSource
) -> FeatureView:
    """Create and return a Titanic passenger stats feature view.

    Args:
        entity: The entity representing a passenger.
        source: The PostgreSQL source for Titanic data.

    Returns:
        Feast feature view instance.
    """
    return FeatureView(
        name="titanic_passenger_stats",
        entities=[entity],
        ttl=timedelta(days=1),
        schema=[
            Field(name=DataFrameColumns.TICKET_CLASS.value, dtype=Int32),
            Field(name=DataFrameColumns.NAME.value, dtype=String),
            Field(name=DataFrameColumns.SEX.value, dtype=String),
            Field(name=DataFrameColumns.AGE.value, dtype=Float32),
            Field(name=DataFrameColumns.NUM_OF_SIBLINGS_OR_SPOUSES.value, dtype=Int32),
            Field(name=DataFrameColumns.NUM_OF_PARENTS_OR_CHILDREN.value, dtype=Int32),
            Field(name=DataFrameColumns.TICKET_NUMBER.value, dtype=String),
            Field(name=DataFrameColumns.FARE.value, dtype=Float32),
            Field(name=DataFrameColumns.DECK.value, dtype=String),
            Field(name=DataFrameColumns.PORT_OF_EMBARKATION.value, dtype=String),
            Field(name=DataFrameColumns.SURVIVED.value, dtype=Bool),
        ],
        online=False,
        source=source,
        tags={"team": "titanic"},
    )


@on_demand_feature_view(
    sources=[
        create_titanic_passenger_stats(
            create_passenger_entity(), create_postgresql_source()
        )
    ],
    schema=[
        Field(name=DataFrameColumns.FARE_CATEGORY.value, dtype=Int64),
        Field(name=DataFrameColumns.AGE_CATEGORY.value, dtype=Int64),
        Field(name=DataFrameColumns.FAMILY_SIZE_GROUPED.value, dtype=String),
        Field(name=DataFrameColumns.TICKET_FREQUENCY.value, dtype=Int64),
        Field(name=DataFrameColumns.TITLE.value, dtype=String),
        Field(name=DataFrameColumns.IS_MARRIED.value, dtype=Int64),
    ],
    mode="python",
)
def additional_and_grouped_features(inputs: pd.DataFrame) -> dict[str, Any]:
    """Calculate additional and grouped features for Titanic passengers.

    Args:
        inputs: Input dataframe with passenger data.

    Returns:
        Dictionary of added features.
    """
    inputs = pd.DataFrame(inputs)

    store_init = (
        len(inputs) == 1 and inputs[DataFrameColumns.NAME.value][0] == "hello world"
    )

    df = pd.DataFrame()

    df[DataFrameColumns.FARE_CATEGORY.value] = bin_fare(
        data=inputs, store_init=store_init
    )[0].cat.codes
    df[DataFrameColumns.AGE_CATEGORY.value] = bin_age(
        data=inputs, store_init=store_init
    )[0].cat.codes
    df[DataFrameColumns.FAMILY_SIZE.value] = add_family_size(data=inputs)
    df[DataFrameColumns.FAMILY_SIZE_GROUPED.value] = group_family_size(data=df)
    df.drop(columns=[DataFrameColumns.FAMILY_SIZE.value], inplace=True)
    df[DataFrameColumns.TICKET_FREQUENCY.value] = add_ticket_frequency(data=inputs)
    df[DataFrameColumns.TITLE.value] = add_title(data=inputs, store_init=store_init)
    df[DataFrameColumns.IS_MARRIED.value] = add_is_married(data=df).astype(np.int64)
    df[DataFrameColumns.TITLE.value] = group_titles(data=df)

    return df.to_dict(orient="list")


def create_feature_service() -> FeatureService:
    """Create and return a feature service for Titanic classifier.

    Returns:
        Feast feature service instance.
    """
    return FeatureService(
        name="titanic_classifier",
        features=[titanic_passenger_stats, additional_and_grouped_features],
    )


project = create_project()
passenger_entity = create_passenger_entity()
titanic_train_data_source = create_postgresql_source()
titanic_passenger_stats = create_titanic_passenger_stats(
    passenger_entity, titanic_train_data_source
)
feature_service = create_feature_service()
