from typing import Annotated, Tuple

import pandas as pd
from zenml import step
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from titanicsurvivors.utils.data import DataFrameColumns


@step()
def split_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "train_input"],
    Annotated[pd.DataFrame, "test_input"],
    Annotated[pd.Series, "train_target"],
    Annotated[pd.Series, "test_target"],
]:
    inputs = data[data.columns.difference([DataFrameColumns.SURVIVED.value])]
    targets = data[DataFrameColumns.SURVIVED]
    train_input, test_input, train_target, test_target = train_test_split(
        inputs, targets, test_size=0.3, random_state=5, stratify=targets
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
    data: pd.DataFrame, label_encoder: LabelEncoder
) -> pd.DataFrame:
    non_numeric_features = [
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
    data: pd.DataFrame, one_hot_encoder: OneHotEncoder
) -> pd.DataFrame:
    categorical_features = [
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
