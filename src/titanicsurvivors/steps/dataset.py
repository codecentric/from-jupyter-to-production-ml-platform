import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing_extensions import Annotated
from zenml import step

from titanicsurvivors.utils.data import DataFrameColumns

load_dotenv()


@step
def split_data_into_subset(
    data: pd.DataFrame, test_split: float = 0.2
) -> tuple[
    Annotated[pd.DataFrame, f"train_input_{os.getenv('GROUP_NAME', 'Default')}"],
    Annotated[pd.DataFrame, f"test_input_{os.getenv('GROUP_NAME', 'Default')}"],
    Annotated[pd.Series, f"train_target_{os.getenv('GROUP_NAME', 'Default')}"],
    Annotated[pd.Series, f"test_target_{os.getenv('GROUP_NAME', 'Default')}"],
]:
    inputs = data[data.columns.difference([DataFrameColumns.SURVIVED.value])]
    targets = data[DataFrameColumns.SURVIVED]
    train_input, test_input, train_target, test_target = train_test_split(
        inputs, targets, test_size=test_split, shuffle=True, stratify=targets
    )
    return train_input, test_input, train_target, test_target


@step()
def feature_transformation(
    data_w_features: pd.DataFrame,
) -> Annotated[pd.DataFrame, "encoded_data"]:
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder()
    encoded_non_numerical = encode_non_numerical(
        data=data_w_features, label_encoder=label_encoder
    )
    encoded_data = encode_categorical(
        data=encoded_non_numerical, one_hot_encoder=one_hot_encoder
    )
    encoded_data = encoded_data.drop(
        columns=[
            column
            for column in [
                DataFrameColumns.NAME.value,
                DataFrameColumns.TICKET_NUMBER.value,
                DataFrameColumns.FAMILY_SIZE.value,
                DataFrameColumns.PASSENGER_ID.value,
                "event_timestamp",
            ]
            if column in encoded_data.columns
        ]
    )
    return encoded_data


def encode_non_numerical(
    data: pd.DataFrame,
    label_encoder: LabelEncoder,
    non_numeric_features: list[str] | None = None,
) -> pd.DataFrame:
    non_numeric_features = non_numeric_features or [
        DataFrameColumns.PORT_OF_EMBARKATION.value,
        DataFrameColumns.SEX.value,
        DataFrameColumns.DECK.value,
        DataFrameColumns.TITLE.value,
        DataFrameColumns.FAMILY_SIZE_GROUPED.value,
    ]
    for feature in non_numeric_features:
        data[feature] = label_encoder.fit_transform(data[feature])
    return data


def encode_categorical(
    data: pd.DataFrame,
    one_hot_encoder: OneHotEncoder,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    categorical_features = categorical_features or [
        DataFrameColumns.TICKET_CLASS.value,
        DataFrameColumns.SEX.value,
        DataFrameColumns.DECK.value,
        DataFrameColumns.PORT_OF_EMBARKATION.value,
        DataFrameColumns.TITLE.value,
        DataFrameColumns.FAMILY_SIZE_GROUPED.value,
    ]
    one_hot_encoder.fit(data[categorical_features])
    encoded_array = one_hot_encoder.transform(data[categorical_features])

    encoded_df = pd.DataFrame(
        encoded_array.toarray(),
        columns=one_hot_encoder.get_feature_names_out(
            input_features=categorical_features
        ),
    )
    df_encoded = pd.concat(
        [
            data.drop(columns=categorical_features),
            encoded_df,
        ],
        axis=1,
    )
    return df_encoded
