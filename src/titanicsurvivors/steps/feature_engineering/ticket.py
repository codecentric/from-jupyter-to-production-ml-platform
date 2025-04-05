import os

from dotenv import load_dotenv
from typing_extensions import Annotated

import pandas as pd
from zenml import step

from titanicsurvivors.utils.data import DataFrameColumns

load_dotenv()


@step
def add_ticket_frequency_feature(
    data: pd.DataFrame,
) -> Annotated[
    pd.DataFrame, f"data_w_ticket_frequency_{os.getenv('GROUP_NAME', 'Default')}"
]:
    frequency = add_ticket_frequency(data=data)
    data.loc[:, DataFrameColumns.TICKET_FREQUENCY.value] = frequency
    return data


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
