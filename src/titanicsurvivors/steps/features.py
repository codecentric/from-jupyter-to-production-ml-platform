from typing import Tuple, Annotated

import numpy as np
import pandas as pd

from titanicsurvivors.utils.data import DataFrameColumns

from zenml import step


from titanicsurvivors.utils.materializers import CategoricalMaterializer


@step(
    output_materializers={
        "passenger_fare_categories": CategoricalMaterializer,
        "passenger_age_categories": CategoricalMaterializer,
    }
)
def add_features(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "data_w_features_label_studio"],
    Annotated[pd.Categorical, "passenger_fare_categories"],
    Annotated[pd.Categorical, "passenger_age_categories"],
]:
    binned_fare, fare_categories = bin_fare(data=data)
    data[DataFrameColumns.FARE.value] = binned_fare.cat.codes
    binned_age, age_categories = bin_age(data=data)
    data[DataFrameColumns.AGE.value] = binned_age.cat.codes
    data[DataFrameColumns.FAMILY_SIZE.value] = add_family_size(data=data)
    data[DataFrameColumns.FAMILY_SIZE_GROUPED.value] = group_family_size(data=data)
    data[DataFrameColumns.TICKET_FREQUENCY.value] = add_ticket_frequency(data=data)
    data[DataFrameColumns.TITLE.value] = add_title(data=data)
    data[DataFrameColumns.IS_MARRIED.value] = add_is_married(data=data)
    data[DataFrameColumns.TITLE.value] = group_titles(data=data)

    return data, fare_categories, age_categories


def bin_fare(
    data: pd.DataFrame, store_init: bool = False
) -> Tuple[pd.Series, pd.Categorical]:
    """
    Bins the 'Fare' column of data into 13 quantile-based categories and returns the binned fares
    and their unique categories.

    The 'Fare' feature in the dataset is positively skewed, with extremely high survival rates at
    the higher ends of the fare distribution. To capture this pattern and the survival rates
    associated with different fare ranges, the fare values are divided into 13 quantile-based bins.
    This method, while producing a relatively high number of bins, provides significant information
    gain, particularly revealing groups with distinct survival rates.

    Observations:
    - Groups at the lower end of the fare distribution tend to have the lowest survival rates.
    - Groups at the higher end of the fare distribution have the highest survival rates.
    - An unusual group in the middle range (15.742, 23.25] is identified with a notably high
    survival rate.

    Args:
        data: The titanic DataFrame containing the 'Fare' column.

    Returns:
        A tuple containing:
            - A Series representing the binned fare values.
            - A Categorical representation of the unique fare bins.
    """

    binned_fare = pd.qcut(
        data[DataFrameColumns.FARE.value],
        13,
        duplicates="drop" if store_init else "raise",
    )
    return binned_fare, binned_fare.unique()


def bin_age(
    data: pd.DataFrame, store_init: bool = False
) -> Tuple[pd.Series, pd.Categorical]:
    """
    Bins the 'Age' column of a data into 10 quantile-based categories and returns the binned ages
    and their unique categories.

    The 'Age' feature in the dataset has a normal distribution with some noticeable spikes
    and bumps. To better capture the survival rate patterns associated with different age ranges,
    the age values are divided into 10 quantile-based bins. This method helps to identify groups
    with distinct survival rates.

    Observations:
    - The first bin, corresponding to the youngest ages, has the highest survival rate.
    - The fourth bin from the distribution has the lowest survival rate.
    - An unusual group in the mid-range (34.0, 40.0] is identified with a notably high
    survival rate.

    Args:
        data: The titanic DataFrame containing the 'Age' column.

    Returns:
        A tuple containing:
            - A Series representing the binned age values.
            - A Categorical representation of the unique age bins.
    """

    binned_fare = pd.qcut(
        data[DataFrameColumns.AGE.value],
        10,
        duplicates="drop" if store_init else "raise",
    )
    return binned_fare, binned_fare.unique()


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


def group_family_size(data: pd.DataFrame) -> pd.Series:
    return data[DataFrameColumns.FAMILY_SIZE.value].apply(get_family_size_group)


def get_family_size_group(family_size: int) -> str:
    if family_size == 1:
        return "Alone"
    if 1 < family_size < 5:
        return "Small"
    if family_size >= 5:
        return "Large"


def add_ticket_frequency(data: pd.DataFrame) -> pd.Series:
    """
    Calculates the frequency of each ticket number in the dataset and returns it as a Series.

    The 'Ticket' feature contains many unique values, making direct analysis difficult.
    By grouping tickets by their frequencies, we can simplify the analysis and gain useful insights
    about the passengers. This feature helps to identify groups of passengers traveling together who
    are not accounted for by the 'Family_Size' feature, such as friends, nannies, maids, etc.,
    who share a ticket.

    Ticket frequency is different from family size because it captures groups of travelers who are
    not related but used the same ticket.

    Observations:
    - Groups with 2, 3, and 4 members have a higher survival rate.
    - Solo travelers have the lowest survival rate.
    - After 4 group members, the survival rate decreases drastically.

    Ticket frequency values help capture nuances similar to 'Family_Size' but with differences in
    group composition, providing additional information.

    Args:
        data: The titanic DataFrame containing the 'Ticket' column.

    Returns:
        A Series representing the frequency of each ticket number in the dataset.
    """

    return data.groupby(DataFrameColumns.TICKET_NUMBER.value)[
        DataFrameColumns.TICKET_NUMBER.value
    ].transform("count")


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
