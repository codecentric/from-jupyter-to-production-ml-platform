#!/bin/sh

# Hole die Prozess ID (PID) des MinIO-Servers

# Warte, bis der MinIO-Server auf Port 9000 verfügbar ist
until curl --output /dev/null --silent --head --fail http://localhost:9001; do
  echo "Wait for MinIO..."
  sleep 1
done

# Setze den MinIO-Client alias
mc alias set myminio http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD


# Erstelle den Bucket 'zenml', wenn er noch nicht existiert
if ! mc ls myminio/zenml > /dev/null 2>&1; then
  mc mb myminio/zenml
fi
echo "Bucket 'zenml' have been successfully created or already exist.."

# Erstelle den Bucket 'zenml', wenn er noch nicht existiert
if ! mc ls myminio/mlflow > /dev/null 2>&1; then
  mc mb myminio/mlflow
fi
echo "Bucket 'mllfow' have been successfully created or already exist.."

# Erstelle den Bucket 'zenml', wenn er noch nicht existiert
if ! mc ls myminio/labelstudio > /dev/null 2>&1; then
  mc mb myminio/labelstudio
  mc cp --recursive /init-data/tasks myminio/labelstudio
fi
echo "Bucket 'labelstudio' have been successfully created or already exist.."

if ! mc ls myminio/data > /dev/null 2>&1; then
  mc mb myminio/data
fi
echo "Bucket 'data' have been successfully created or already exist.."




