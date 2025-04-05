import os

from dotenv import load_dotenv
from typing_extensions import Annotated

import numpy as np
import pandas as pd
from zenml import step

from titanicsurvivors.utils.data import DataFrameColumns

load_dotenv()


@step
def add_title_feature(
    data: pd.DataFrame,
) -> Annotated[pd.DataFrame, f"data_w_title_{os.getenv('GROUP_NAME', 'Default')}"]:
    title = add_title(data=data)
    data.loc[:, DataFrameColumns.TITLE.value] = title
    grouped_title = group_titles(data=data)
    is_married = add_is_married(data=data)

    data.loc[:, DataFrameColumns.IS_MARRIED.value] = is_married
    data.loc[:, DataFrameColumns.TITLE.value] = grouped_title
    return data


def add_title(data: pd.DataFrame, store_init: bool = False) -> pd.Series:
    """
    Extracts and returns the title from the 'Name' column of data.

    Observations:
    - There are many titles that occur rarely and some that seem incorrect.
    - The 'Master' title is unique and typically given to male passengers below age 26, who have the highest
      survival rate among all males.

    Args:
        data: The titanic DataFrame containing the 'Name' column.

    Returns:
        A Series representing the extracted titles from the 'Name' column.
    """
    if store_init:
        return data[DataFrameColumns.NAME.value]
    return (
        data[DataFrameColumns.NAME.value]
        .str.split(", ", expand=True)[1]
        .str.split(".", expand=True)[0]
    )


def add_is_married(data: pd.DataFrame) -> pd.Series:
    """
    Adds a binary feature indicating whether a passenger is married, based on the 'Mrs' title.

    The 'Mrs' title has the highest survival rate among female titles. Since all female titles
    are grouped together in other features, the 'Is_Married' feature separates those with the 'Mrs'
    title to retain this significant information.

    Args:
        data: The titanic DataFrame containing the 'Title' column.

    Returns:
        A Series representing the binary 'Is_Married' feature, where 1 indicates the 'Mrs' title
                   and 0 indicates otherwise.
    """

    is_married = pd.Series(np.zeros(len(data)))
    is_married.loc[data[DataFrameColumns.TITLE.value] == "Mrs"] = 1
    return is_married


def group_titles(data: pd.DataFrame) -> pd.DataFrame:
    """
    Groups less common titles in the 'Title' column of the DataFrame into more general categories.

    The 'Title' feature created from the 'Name' column has various titles with low frequencies
    and some incorrect entries. This function replaces specific titles with broader categories
    for simplification and to reflect similar characteristics among passengers.

    Replacement rules:
    - Female-specific titles (Miss, Mrs, Ms, Mlle, Lady, Mme, the Countess, Dona) are grouped
        as 'Miss/Mrs/Ms'. Note: Titles like Mlle, Mme, and Dona are actual names mistaken as titles
        due to splitting.
    - Titles indicating professional or social status
        (Dr, Col, Major, Jonkheer, Capt, Sir, Don, Rev) are grouped as 'Dr/Military/Noble/Clergy'.

    Args:
        data: The titanic DataFrame containing the 'Title' column.

    Returns:
         A Series representing the grouped titles.
    """

    return data[DataFrameColumns.TITLE.value].apply(get_title_group)


def get_title_group(title: str) -> str:
    """
    Groups a given title into a more general category based on predefined rules.

    Replacement rules:
    - Female-specific titles (Miss, Mrs, Ms, Mlle, Lady, Mme, the Countess, Dona) are replaced
        with 'Miss/Mrs/Ms'. Note: Titles like Mlle, Mme, and Dona are actual names mistaken as
        titles due to splitting.
    - Titles indicating professional or social status
    (Dr, Col, Major, Jonkheer, Capt, Sir, Don, Rev) are replaced with 'Dr/Military/Noble/Clergy'.

    Args:
        title: The original title string.

    Returns:
        The grouped title string based on the predefined rules.
    """

    if title in ["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the Countess", "Dona"]:
        return "Miss/Mrs/Ms"
    if title in ["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"]:
        return "Dr/Military/Noble/Clergy"

    return title
