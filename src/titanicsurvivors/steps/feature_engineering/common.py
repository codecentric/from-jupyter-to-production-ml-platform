import os

from dotenv import load_dotenv
from typing_extensions import Annotated

import pandas as pd
from zenml import step

from titanicsurvivors.utils.data import DataFrameColumns

load_dotenv()


@step
def combine_features(
    age_data: pd.DataFrame,
    family_data: pd.DataFrame,
    fare_data: pd.DataFrame,
    ticket_data: pd.DataFrame,
    title_data: pd.DataFrame,
) -> Annotated[pd.DataFrame, f"combined_features_{os.getenv('GROUP_NAME', 'Default')}"]:
    age_data[DataFrameColumns.FAMILY_SIZE_GROUPED.value] = family_data[
        DataFrameColumns.FAMILY_SIZE_GROUPED.value
    ]
    age_data[DataFrameColumns.FARE.value] = fare_data[
        DataFrameColumns.FARE_CATEGORY.value
    ]
    age_data[DataFrameColumns.TICKET_CLASS.value] = ticket_data[
        DataFrameColumns.TICKET_CLASS.value
    ]
    age_data[DataFrameColumns.TICKET_FREQUENCY.value] = ticket_data[
        DataFrameColumns.TICKET_FREQUENCY.value
    ]
    age_data[DataFrameColumns.TITLE.value] = title_data[DataFrameColumns.TITLE.value]
    age_data[DataFrameColumns.IS_MARRIED.value] = title_data[
        DataFrameColumns.IS_MARRIED.value
    ]

    return age_data
