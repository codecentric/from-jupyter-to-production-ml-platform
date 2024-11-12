import json


from fastparquet import ParquetFile

data = ParquetFile("./data/train.parquet").to_pandas()
data = data.fillna(-1)
labelled_data_list = []
records = data.to_dict(orient="records")
for idx, row in enumerate(records):
    label = row.pop("Survived")
    id = row.pop("PassengerId")

    record = {
        "predictions": [],
        "data": {
            "ID": id,
            "p_data": row,
        },
        "annotations": [
            {
                "result": [
                    {
                        "from_name": "sentiment_class",
                        "to_name": "message",
                        "type": "choices",
                        "readonly": False,
                        "hidden": False,
                        "value": {"choices": "Survived" if label else "Died"},
                    }
                ]
            }
        ],
    }
    # Save JSON list to a file
    with open(f"./data/tasks/label_task_{idx}.json", "w") as f:
        json.dump(record, f, indent=2)
