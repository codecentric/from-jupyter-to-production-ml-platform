from typing import Annotated

import bentoml
from bentoml.validators import DType, Shape
import numpy as np


@bentoml.service(
    name="TitanicRFService",
)
class TitanicRFService:
    def __init__(self):
        # load model
        self.model = bentoml.sklearn.load_model("titanic_random_forest")

    @bentoml.api()
    async def predict_ndarray(
        self, inp: Annotated[np.ndarray, DType("float32"), Shape((1, 25))]
    ) -> np.ndarray:
        output_tensor = await self.model.predict(inp)
        return output_tensor
