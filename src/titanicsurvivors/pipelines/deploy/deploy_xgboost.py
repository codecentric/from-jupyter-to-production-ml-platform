import os

from dotenv import load_dotenv
from zenml import pipeline
from zenml.client import Client
from zenml.integrations.bentoml.steps import (
    bento_builder_step,
    bentoml_model_deployer_step,
)

from titanicsurvivors.models import titanic_xgboost


load_dotenv()


@pipeline(model=titanic_xgboost, enable_cache=False)
def bento_deployment_pipeline():
    model_artifact = Client().get_artifact_version(
        f"xgb_model_{os.getenv('GROUP_NAME', 'Default')}"
    )  # ... Change the name of the model artifact. Exchange with a Datat Scientist what name was given to the model artifact.

    bento = bento_builder_step(
        model=model_artifact,
        model_name="titanic-classifier",  # ... Add a model name. This must match the one in the service.
        model_type="xgboost",
        service="service.py:TitanicService",
        labels={
            "framework": "xgboost",
            "dataset": "titanic",
        },
        exclude=["data"],
        python={
            "packages": ["zenml", "xgboost", "pandas", "scikit-learn"],
        },
    )

    # bentoml_model_deployer_step(
    #     bento=bento,
    #     model_name="titanic-classifier",  # Name of the model
    #     port=3001,  # Port to be used by the http server
    #     deployment_type="container",
    #     image="mlopsworkshop/titanicsurvivors",
    #     image_tag=titanicsurvivors.__version__,
    #     platform="linux/amd64",
    # )

    deployed_model = bentoml_model_deployer_step(
        bento=bento,
        model_name="titanic-classifier",  # Name of the model
        port=3001,  # Port to be used by the http server
    )


if __name__ == "__main__":
    bento_deployment_pipeline()
