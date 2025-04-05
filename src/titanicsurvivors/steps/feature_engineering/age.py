import os

import pandas as pd
from dotenv import load_dotenv
from pandas import Categorical
from typing_extensions import Annotated
from zenml import step

from titanicsurvivors.utils.data import DataFrameColumns

load_dotenv()


@step(enable_cache=False)
def divide_age_in_bins(
    data: pd.DataFrame, store_init: bool = False
) -> tuple[
    Annotated[pd.DataFrame, f"data_w_binned_age_{os.getenv('GROUP_NAME', 'Default')}"],
    Annotated[Categorical, f"age_categories_{os.getenv('GROUP_NAME', 'Default')}"],
]:
    binned_age_data, age_category_values = bin_age(data=data, store_init=store_init)
    data.loc[:, DataFrameColumns.AGE.value] = binned_age_data.cat.codes
    return data, age_category_values


def bin_age(
    data: pd.DataFrame, store_init: bool = False
) -> tuple[pd.Series, pd.Categorical]:
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
