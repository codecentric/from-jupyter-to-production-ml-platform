from enum import Enum


class DataFrameColumns(str, Enum):
    PASSENGER_ID: str = "PassengerId"
    NAME: str = "Name"
    TICKET_CLASS: str = "Pclass"
    SEX: str = "Sex"
    AGE: str = "Age"
    AGE_CATEGORY: str = "Age_Category"
    TITLE: str = "Title"
    NUM_OF_SIBLINGS_OR_SPOUSES: str = "SibSp"
    NUM_OF_PARENTS_OR_CHILDREN: str = "Parch"
    TICKET_NUMBER: str = "Ticket"
    TICKET_FREQUENCY: str = "Ticket_Frequency"
    FARE: str = "Fare"
    FARE_CATEGORY: str = "Fare_Category"
    CABIN_NUMBER: str = "Cabin"
    PORT_OF_EMBARKATION: str = "Embarked"
    SURVIVED: str = "Survived"
    DECK: str = "Deck"
    FAMILY_SIZE: str = "Family_Size"
    FAMILY_SIZE_GROUPED: str = "Family_Size_Grouped"
    IS_MARRIED: str = "is_married"
    EVENT_TIMESTAMP: str = "event_timestamp"


PASSENGER_ID = DataFrameColumns.PASSENGER_ID
