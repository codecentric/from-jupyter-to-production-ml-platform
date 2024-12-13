import os
from datetime import datetime

from zenml import step
import pandas as pd
from sqlalchemy import create_engine


@step(enable_cache=False)
def insert_data_into_database(data: pd.DataFrame, table_name: str = "titanic"):
    engine = create_engine(os.getenv("POSTGRES_CONNECTION_STRING"))
    data["event_timestamp"] = [datetime.now() for _ in range(len(data))]
    data["created"] = [datetime.now() for _ in range(len(data))]
    data.to_sql(table_name, engine, if_exists="replace", index=False)
