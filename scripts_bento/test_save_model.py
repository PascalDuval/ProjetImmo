import bentoml
import pandas as pd

model_ref = bentoml.sklearn.get("random_forest_energy:latest")
model = bentoml.sklearn.load_model("random_forest_energy:latest")
features = model_ref.custom_objects["features"]

# exemple de test
sample = pd.DataFrame([{
    "BuildingAge": 25,
    "log_surface": 3.7,
    "has_parking": 1,
    "Use_Office": 0,
    "Use_Other": 0,
    "Use_Retail": 1,
    "Use_Warehouse": 0,
    "Use_Unknown": 0
}])

sample = sample.reindex(columns=features, fill_value=0)
print("🔮 Prediction:", model.predict(sample)[0])
