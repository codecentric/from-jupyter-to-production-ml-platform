from zenml import Model

titanic_random_forest = Model(
    # The name uniquely identifies this model
    # It usually represents the business use case
    name="titanic_random_forest",
    # The version specifies the version
    # If None or an unseen version is specified, it will be created
    # Otherwise, a version will be fetched
    # Some other properties may be specified
    license="Apache 2.0",
    description="A random forest classifier for the titanic dataset.",
)

titanic_xgboost = Model(
    # The name uniquely identifies this model
    # It usually represents the business use case
    name="titanic_xgboost",
    # The version specifies the version
    # If None or an unseen version is specified, it will be created
    # Otherwise, a version will be fetched
    # Some other properties may be specified
    license="Apache 2.0",
    description="A xgboost classifier for the titanic dataset.",
)
