#!/bin/bash

while ! mysqladmin ping -h"localhost" --silent; do
    echo "Wait to start MySQL ..."
    sleep 3
done

mysql -u root -p"$MYSQL_ROOT_PASSWORD" <<-EOSQL
CREATE DATABASE IF NOT EXISTS mlflow;
CREATE USER IF NOT EXISTS '$MYSQL_MLFLOW_USER'@'%' IDENTIFIED BY '$MYSQL_MLFLOW_PASSWORD';
GRANT ALL PRIVILEGES ON mlflow.* TO '$MYSQL_MLFLOW_USER'@'%';
FLUSH PRIVILEGES;
EOSQL

echo "The database 'mlflow' have been successfully created or already exist."

mysql -u root -p"$MYSQL_ROOT_PASSWORD" <<-EOSQL
CREATE DATABASE IF NOT EXISTS zenml;
CREATE USER IF NOT EXISTS '$MYSQL_ZENML_USER'@'%' IDENTIFIED BY '$MYSQL_ZENML_PASSWORD';
GRANT ALL PRIVILEGES ON zenml.* TO '$MYSQL_ZENML_USER'@'%';
FLUSH PRIVILEGES;
EOSQL

echo "The database 'zenml' have been successfully created or already exist."

mysql -u root -p"$MYSQL_ROOT_PASSWORD" <<-EOSQL
CREATE DATABASE IF NOT EXISTS feast;
CREATE USER IF NOT EXISTS '$MYSQL_FEAST_USER'@'%' IDENTIFIED BY '$MYSQL_FEAST_PASSWORD';
GRANT ALL PRIVILEGES ON feast.* TO '$MYSQL_FEAST_USER'@'%';
FLUSH PRIVILEGES;
EOSQL

echo "The database 'zenml' have been successfully created or already exist."
