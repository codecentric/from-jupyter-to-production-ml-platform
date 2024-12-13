#!make
include .env

zenml-create-service-account:
	zenml service-account create <serivce_name>

zenml-login:
	zenml login http://localhost:8080 --api-key

zenml-create-artifact-store-minio:
	zenml integration install -y s3

	zenml secret create minio_secret \
    --aws_access_key_id=${AWS_ACCESS_KEY_ID} \
    --aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}

	zenml artifact-store register minio_store -f s3 \
    --path='s3://zenml' \
    --authentication_secret=minio_secret \
    --client_kwargs='{"endpoint_url": "http://localhost:9000", "region_name": "eu-east-1"}'

zenml-create-container-registry:
	zenml container-registry register local-docker-registry \
    --flavor=default \
    --uri=localhost:5000

	zenml service-connector register local-docker-registry-service-connector --type docker --username=${DOCKER_REGISTRY_USER} --password=${DOCKER_REGISTRY_PASSWORD} --registry=localhost:5000

	zenml container-registry connect local-docker-registry --connector=local-docker-registry-service-connector

zenml-register-docker-orchestrator:
	zenml orchestrator register docker \
    --flavor=local_docker

zenml-local-image-builder:
	zenml image-builder register local-image-builder --flavor=local


zenml-connect-github-repo:
	zenml code-repository register ${REPO_NAME} --type=github \
	--url=https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git \
	--owner=${GITHUB_USERNAME} --repository=${REPO_NAME} \
	--token=${GITHUB_TOKEN}


zenml-create-exp-tracker:
	zenml experiment-tracker register mlflow \
    --flavor=mlflow \
    --tracking_uri=http://localhost:5001 \
    --tracking_username="admin" \
    --tracking_password="password"

zenml-create-model-registry:
	zenml model-registry register mlflow_model_registry --flavor=mlflow

zenml-add-annotation:
	zenml secret create label_studio_secrets --api_key=${LABEL_STUDIO_ACCESS_KEY}
	zenml annotator register label_studio --flavor label_studio --authentication_secret=label_studio_secrets --instance_url="http://localhost" --port=8081

zenml-add-feast:
	zenml feature-store register feast_store --flavor=feast --feast_repo="./src/titanicsurvivors/feature_repo"

zenml-add-model-deployer:
	zenml model-deployer register bentoml_deployer --flavor=bentoml

zenml-register-stack:
	zenml stack register docker-compose -o docker -a minio_store -e mlflow -r mlflow_model_registry -an label_studio -f feast_store -c local-docker-registry -d bentoml_deployer -i local-image-builder

zenml-create-components:
	zenml integration install -y s3

	zenml secret create minio_secret \
    --aws_access_key_id=${AWS_ACCESS_KEY_ID} \
    --aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}

	zenml artifact-store register minio_store -f s3 \
    --path='s3://zenml' \
    --authentication_secret=minio_secret \
    --client_kwargs='{"endpoint_url": "http://localhost:9000", "region_name": "eu-east-1"}'

	zenml container-registry register local-docker-registry \
    --flavor=default \
    --uri=localhost:5000

	zenml service-connector register local-docker-registry-service-connector --type docker --username=${DOCKER_REGISTRY_USER} --password=${DOCKER_REGISTRY_PASSWORD} --registry=localhost:5000

	zenml container-registry connect local-docker-registry --connector=local-docker-registry-service-connector

	zenml experiment-tracker register mlflow \
    --flavor=mlflow \
    --tracking_uri=http://localhost:5001 \
    --tracking_username="admin" \
    --tracking_password="password"

	zenml model-registry register mlflow_model_registry --flavor=mlflow

	zenml secret create label_studio_secrets --api_key=${LABEL_STUDIO_ACCESS_KEY}
	zenml annotator register label_studio --flavor label_studio --authentication_secret=label_studio_secrets --instance_url="http://localhost" --port=8081

	zenml feature-store register feast_store --flavor=feast --feast_repo="./src/titanicsurvivors/feature_repo"

	zenml model-deployer register bentoml_deployer --flavor=bentoml

