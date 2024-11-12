from typing import Annotated

import pandas as pd

from zenml import step

from titanicsurvivors.utils.data import DataFrameColumns


@step()
def handle_missing_values(
    data: pd.DataFrame,
) -> Annotated[pd.DataFrame, "data_without_missing_data"]:
    data[DataFrameColumns.AGE.value] = fill_missing_age(data=data)
    data[DataFrameColumns.PORT_OF_EMBARKATION.value] = fill_missing_embarked(data=data)
    data[DataFrameColumns.FARE.value] = fill_missing_fare(data=data)
    data[DataFrameColumns.DECK.value] = replace_cabin_w_deck(data=data)
    data = data.drop(columns=[DataFrameColumns.CABIN_NUMBER.value])

    return data


def fill_missing_age(data: pd.DataFrame) -> pd.Series:
    """
    Fills missing values in the 'Age' column of data filled with the median age based on
    'Sex' and 'Pclass' groups.

    Missing values in the 'Age' column are filled using the median age of groups defined by
    'Sex' and 'Pclass'. This approach is chosen because the 'Pclass' feature has a high correlation
    with 'Age' (0.408106) and 'Survived' (0.338481), making it a logical choice to group by.
    Additionally, 'Sex' is used as the second level of grouping to improve accuracy.
    Typically, median ages for 'Pclass' and 'Sex' combinations are distinct, with higher classes
    associated with higher median ages and females generally having slightly lower median ages
    than males.

    Args:
        data: The titanic DataFrame containing the columns 'Sex', 'Pclass',
                and 'Age'.

    Returns:
        A Series representing the 'Age' column with missing values filled based on
        group medians.
    """
    return data.groupby(
        [DataFrameColumns.SEX.value, DataFrameColumns.TICKET_CLASS.value]
    )[DataFrameColumns.AGE.value].transform(lambda x: x.fillna(x.median()))


def fill_missing_embarked(data: pd.DataFrame) -> pd.Series:
    """
    Fills missing values in the 'Embarked' column of data filled with 'S' (Southampton)
    based on background information.

    'Embarked' is a categorical feature, and there are only 2 missing values in the entire data set.
    Both of these passengers are female, upper class, and have the same ticket number, indicating
    they knew each other and embarked from the same port. While the mode 'Embarked' value for an
    upper class female passenger is 'C' (Cherbourg), further research reveals that
    Mrs. George Nelson Stone and her maid Amelie Icard boarded the Titanic in Southampton.
    Therefore, the missing values in 'Embarked' are filled with 'S' (Southampton) based
    on this information.

    Args:
        data: The input titanic DataFrame containing the 'Embarked' column.

    Returns:
        A Series representing the 'Embarked' column with missing values filled with 'S'
        (Southampton).
    """

    return data[DataFrameColumns.PORT_OF_EMBARKATION.value].fillna("S")


def fill_missing_fare(data: pd.DataFrame) -> pd.Series:
    """
    Fills missing values in the 'Fare' column of the data filled with the median Fare of a
    third class passenger traveling alone.

    There is only one passenger with a missing Fare value. The Fare is assumed to be related to
    family size (Parch and SibSp) and Pclass features. The median Fare value of a male with a
    third class ticket and no family members (Parch = 0 and SibSp = 0) is chosen as a logical
    replacement for the missing value. This approach is used to estimate the missing Fare in a
    manner aligned with typical fare patterns for such passengers.

    Args:
        data: The titanic DataFrame containing the columns 'Pclass', 'Parch', 'SibSp', and 'Fare'.

    Returns:
        A Series representing the 'Fare' column with the missing value filled based on the
        median Fare of a third class passenger traveling alone.
    """
    med_fare = data.groupby(
        [
            DataFrameColumns.TICKET_CLASS.value,
            DataFrameColumns.NUM_OF_PARENTS_OR_CHILDREN.value,
            DataFrameColumns.NUM_OF_SIBLINGS_OR_SPOUSES.value,
        ]
    )[DataFrameColumns.FARE.value].transform("median")

    return data[DataFrameColumns.FARE.value].fillna(med_fare)


def replace_cabin_w_deck(data: pd.DataFrame) -> pd.Series:
    """
    Creates the 'Deck' column for data by grouping similar decks and handling missing values.

    The 'Cabin' feature contains important information about passenger survival rates, as it
    relates to the deck on which the cabin is located. The first letter of the 'Cabin' values
    represents the deck. This function addresses the deck information by grouping them based on
    passenger class distribution and survival rates, and by handling specific cases such as
    the 'T' deck.

    The grouping is done as follows:
    - Decks 'A', 'B', and 'C' are grouped together as 'ABC' since they only contain 1st
        class passengers.
    - Decks 'D' and 'E' are grouped together as 'DE' due to their similar passenger class
        distribution and survival rates.
    - Decks 'F' and 'G' are grouped together as 'FG' for the same reasons.
    - The 'T' deck, which only has one 1st class passenger, is grouped with 'A' as it closely
        resembles the A deck.
    - 'M' is used to represent missing 'Cabin' values and is treated as a separate deck due to its
        distinct characteristics and lower survival rates.

    Args:
        data: The titanic DataFrame containing the 'Deck' column.

    Returns:
        A Series representing the optimized 'Deck' column with grouped deck values.
    """

    data[DataFrameColumns.DECK.value] = data[DataFrameColumns.CABIN_NUMBER.value].apply(
        lambda cabin_number: cabin_number[0] if pd.notnull(cabin_number) else "M"
    )

    idx = data[data[DataFrameColumns.DECK.value] == "T"].index
    data.loc[idx, DataFrameColumns.DECK.value] = "A"
    data[DataFrameColumns.DECK.value] = data[DataFrameColumns.DECK.value].replace(
        ["A", "B", "C"], "ABC"
    )
    data[DataFrameColumns.DECK.value] = data[DataFrameColumns.DECK.value].replace(
        ["D", "E"], "DE"
    )
    data[DataFrameColumns.DECK.value] = data[DataFrameColumns.DECK.value].replace(
        ["F", "G"], "FG"
    )
    return data[DataFrameColumns.DECK.value]
