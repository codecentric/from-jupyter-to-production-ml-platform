from zenml import pipeline
from zenml.client import Client
import zenml

from zenml.integrations.bentoml.steps import (
    bento_builder_step,
    bentoml_model_deployer_step,
)

from titanicsurvivors.models import titanic_random_forest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pipeline(
    model=titanic_random_forest,
)
def deploy_model(random_forest_artifact_name="random_forest"):
    random_forest = Client().get_artifact_version(
        name_id_or_prefix=random_forest_artifact_name
    )

    bento = bento_builder_step(
        model=random_forest,
        model_name=titanic_random_forest.name,
        model_type="sklearn",
        service="services:TitanicRfService",
        labels={
            "framework": "sklearn",
            "dataset": "titanic",
            "zenml_version": zenml.__version__,
        },
        exclude=["data"],
        python={
            "packages": ["zenml", "scikit-learn", "pandas", "numpy"],
        },
    )

    bentoml_model_deployer_step(
        bento=bento,
        deployment_type="local",
        model_name=titanic_random_forest.name,
        port=3000,
        image="titanicsurvivors/rfclassifier",
        image_tag=titanic_random_forest.version,
        platform="linux/amd64",
    )


if __name__ == "__main__":
    deploy_model()
