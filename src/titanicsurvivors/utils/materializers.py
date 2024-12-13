import json
import os
from typing import Type, ClassVar, Tuple, Any
import pandas as pd
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType


class CategoricalMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (pd.Categorical,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.DATA

    def load(self, data_type: Type[pd.Categorical]) -> pd.Categorical:
        """Read from artifact store."""
        with self.artifact_store.open(os.path.join(self.uri, "data.json"), "r") as file:
            cat_dict = json.load(file)

        categories = [
            pd.Interval(left, right) for left, right in cat_dict["categories"]
        ]
        return pd.Categorical.from_codes(
            codes=cat_dict["codes"], categories=categories, ordered=cat_dict["ordered"]
        )

    def save(self, categorical: pd.Categorical) -> None:
        """Write to artifact store."""
        with self.artifact_store.open(os.path.join(self.uri, "data.json"), "w") as file:
            cat_dict = {
                "codes": categorical.codes.tolist(),
                "categories": [
                    (interval.left, interval.right)
                    for interval in categorical.categories
                ],
                "ordered": categorical.ordered,
            }
            json.dump(cat_dict, file)
