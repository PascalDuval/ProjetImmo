import pandas as pd
import bentoml
from pydantic import BaseModel

# Charger le modèle
model_ref = bentoml.sklearn.get("random_forest_energy:latest")
model = bentoml.sklearn.load_model("random_forest_energy:latest")
features = model_ref.custom_objects["features"]

# Schéma d’entrée
class EnergyInput(BaseModel):
    BuildingAge: float
    log_surface: float
    has_parking: int
    Use_Office: int = 0
    Use_Other: int = 0
    Use_Retail: int = 0
    Use_Warehouse: int = 0
    Use_Unknown: int = 0


# ✅ Nouveau style de déclaration (annotations plutôt que input/output)
@bentoml.service(name="random_forest_energy_service")
class EnergyService:
    def predict(self, input_data: EnergyInput) -> dict:
        """
        Endpoint REST : /predict
        """
        df = pd.DataFrame([input_data.dict()])
        df = df.reindex(columns=features, fill_value=0)
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}
