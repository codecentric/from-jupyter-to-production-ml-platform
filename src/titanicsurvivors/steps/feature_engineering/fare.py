import os

from dotenv import load_dotenv
from typing_extensions import Annotated

import pandas as pd
from pandas import Categorical
from zenml import step

from titanicsurvivors.utils.data import DataFrameColumns

load_dotenv()


@step
def divide_fare_in_bins(
    data: pd.DataFrame, store_init: bool = False
) -> tuple[
    Annotated[pd.DataFrame, f"data_w_binned_fare_{os.getenv('GROUP_NAME', 'Default')}"],
    Annotated[Categorical, f"fare_categories_{os.getenv('GROUP_NAME', 'Default')}"],
]:
    binned_fare_data, fare_category_values = bin_fare(data=data, store_init=store_init)
    data.loc[:, DataFrameColumns.FARE_CATEGORY.value] = binned_fare_data.cat.codes
    return data, fare_category_values


def bin_fare(
    data: pd.DataFrame, store_init: bool = False
) -> tuple[pd.Series, pd.Categorical]:
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
