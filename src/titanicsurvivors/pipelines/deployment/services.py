from typing import List

import bentoml
import numpy as np
import pandas as pd


@bentoml.service(
    name="TitanicRfService",
)
class TitanicRfService:
    def __init__(self):
        # load model
        self.model = bentoml.sklearn.load_model("titanic_random_forest")

    @bentoml.api()
    async def predict(self, inputs: List[dict]) -> List[str]:
        output_tensor = self.model.predict(pd.DataFrame(inputs))
        return np.where(output_tensor == 1, "Survived", "Died").tolist()
