from zenml import Model

titanic_xgboost = Model(
    name="titanic_xgboost",
    license="Apache 2.0",
    description="A xgboost classifier for the titanic dataset.",
)
