from zenml import pipeline
from zenml.client import Client

from titanicsurvivors.steps.database import insert_data_into_database

from dotenv import load_dotenv

load_dotenv()


@pipeline()
def add_data_into_database():
    client = Client()
    data_without_missing_data = client.get_artifact_version(
        name_id_or_prefix="data_without_missing_data"
    )
    insert_data_into_database(data=data_without_missing_data)


if __name__ == "__main__":
    add_data_into_database()
