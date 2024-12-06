from zenml import Model

titanic_random_forest = Model(
    name="titanic_random_forest",
    license="Apache 2.0",
    description="A random forest classifier for the titanic dataset.",
)

titanic_xgboost = Model(
    name="titanic_xgboost",
    license="Apache 2.0",
    description="A xgboost classifier for the titanic dataset.",
)
