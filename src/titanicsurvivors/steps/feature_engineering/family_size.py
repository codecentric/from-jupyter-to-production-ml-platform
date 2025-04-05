import os

from dotenv import load_dotenv
from typing_extensions import Annotated

import pandas as pd
from zenml import step

from titanicsurvivors.utils.data import DataFrameColumns

load_dotenv()


@step
def add_family_size_feature(
    data: pd.DataFrame,
) -> Annotated[
    pd.DataFrame, f"data_w_family_size_{os.getenv('GROUP_NAME', 'Default')}"
]:
    family_size = add_family_size(data=data)
    data.loc[:, DataFrameColumns.FAMILY_SIZE.value] = family_size
    data.loc[:, DataFrameColumns.FAMILY_SIZE_GROUPED.value] = group_family_size(
        data=data
    )
    return data


def group_family_size(data: pd.DataFrame) -> pd.Series:
    return data[DataFrameColumns.FAMILY_SIZE.value].apply(get_family_size_group)


def get_family_size_group(family_size: int) -> str:
    if family_size == 1:
        return "Alone"
    if 1 < family_size < 5:
        return "Small"
    if family_size >= 5:
        return "Large"


def add_family_size(data: pd.DataFrame) -> pd.Series:
    """
    Calculates the family size of each passenger and returns it as a Series.

    The family size is determined by summing the number of siblings/spouses and parents/children
    for each passenger, and adding one to include the passenger themselves. This provides an
    indication of the total family members each passenger is traveling with, which could be a useful
    feature for analysis, especially in understanding survival rates.

    Args:
        data: The titanic DataFrame containing the columns for number of siblings/spouses
                             and number of parents/children.

    Returns:
        A Series representing the family size of each passenger.
    """

    return (
        data[DataFrameColumns.NUM_OF_SIBLINGS_OR_SPOUSES.value]
        + data[DataFrameColumns.NUM_OF_PARENTS_OR_CHILDREN.value]
        + 1
    )
