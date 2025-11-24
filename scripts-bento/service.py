# scripts-bento/service.py

import bentoml
from pydantic import BaseModel
from typing import List
import pandas as pd

# Charger le modèle BentoML
model_ref = bentoml.sklearn.get("random_forest_energy_model:latest")
model = model_ref.to_runner()
features = model_ref.custom_objects["features"]

# Définition du service BentoML
svc = bentoml.Service("rf_energy_service", runners=[model])

# Modèle de validation avec Pydantic
class EnergyInput(BaseModel):
    BuildingAge: float
    log_surface: float
    efficiency_score: float
    surface_per_floor: float
    parking_ratio: float
    Usage_Office: float = 0.0
    Usage_Retail: float = 0.0
    Usage_Education: float = 0.0
    Usage_Residential: float = 0.0
    Usage_Storage: float = 0.0
    Usage_Unknown: float = 0.0

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def predict(input_data: List[EnergyInput]):
    df = pd.DataFrame([data.dict() for data in input_data])
    df = df.reindex(columns=features, fill_value=0.0)  # alignement des colonnes
    predictions = model.predict.run(df)
    return predictions.tolist()
