from typing import List

import bentoml
import numpy as np
import pandas as pd
import xgboost as xgb


@bentoml.service(
    name="TitanicService",
)
class TitanicService:
    def __init__(self):
        self.model = bentoml.xgboost.load_model("titanic-classifier")

    @bentoml.api()
    async def predict(self, inputs: List[dict]) -> List[str]:
        dinput = xgb.DMatrix(pd.DataFrame(inputs))
        output_tensor = self.model.predict(dinput)
        print(output_tensor)

        return np.where(
            np.array([round(output) for output in output_tensor]) == 1,
            "Survived",
            "Died",
        ).tolist()
