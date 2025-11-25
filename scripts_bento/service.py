import pandas as pd
import bentoml
from pydantic import BaseModel, field_validator, model_validator
import pandera.pandas as pa
from pandera import Column, Check

# Charger le modèle BentoML
model_ref = bentoml.sklearn.get("random_forest_energy:latest")
model = bentoml.sklearn.load_model("random_forest_energy:latest")
features = model_ref.custom_objects["features"]

# ✅ Schéma Pydantic — Validation des valeurs reçues
class EnergyInput(BaseModel):
    BuildingAge: float
    log_surface: float
    has_parking: int
    Use_Office: int = 0
    Use_Other: int = 0
    Use_Retail: int = 0
    Use_Warehouse: int = 0
    Use_Unknown: int = 0

    @field_validator("BuildingAge", "log_surface")
    def must_be_positive(cls, v, field):
        if v < 0:
            raise ValueError(f"{field.name} doit être positif.")
        return v

    @field_validator("has_parking")
    def must_be_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError("has_parking doit valoir 0 ou 1.")
        return v

    @model_validator(mode="after")
    def check_usage_fields(self):
        if sum([
            self.Use_Office,
            self.Use_Other,
            self.Use_Retail,
            self.Use_Warehouse,
            self.Use_Unknown,
        ]) == 0:
            raise ValueError("Au moins un champ 'Use_*' doit être égal à 1.")
        return self


# ✅ Schéma Pandera — validation uniquement sur les features d’entrée utilisateur
energy_schema = pa.DataFrameSchema({
    "BuildingAge": Column(float, Check.ge(0), nullable=False),
    "log_surface": Column(float, Check.ge(0), nullable=False),
    "has_parking": Column(int, Check.isin([0, 1]), nullable=False),
    "Use_Office": Column(int, Check.isin([0, 1]), nullable=False),
    "Use_Other": Column(int, Check.isin([0, 1]), nullable=False),
    "Use_Retail": Column(int, Check.isin([0, 1]), nullable=False),
    "Use_Warehouse": Column(int, Check.isin([0, 1]), nullable=False),
    "Use_Unknown": Column(int, Check.isin([0, 1]), nullable=False),
}, coerce=True, strict=False)


# ✅ Mapping pour aligner les colonnes du modèle
COLUMN_MAPPING = {
    "Use_Retail": "Use_Retail Store",
    "Use_Warehouse": "Use_Non-Refrigerated Warehouse",
    "Use_Office": "Use_Office",
    "Use_Other": "Use_Other",
    "Use_Unknown": "Use_Unknown",
}


@bentoml.service(name="random_forest_energy_service")
class EnergyService:

    @bentoml.api(route="/predict", name="predict")
    def predict(self, data: EnergyInput) -> dict:
        """
        Endpoint POST /predict
        JSON attendu :
        {
          "data": {
            "BuildingAge": 10,
            "log_surface": 3.6,
            "has_parking": 1,
            "Use_Retail": 1
          }
        }
        """
        df = pd.DataFrame([data.dict()])

        # ✅ Validation Pandera avant renommage
        try:
            energy_schema.validate(df)
        except pa.errors.SchemaError as e:
            return {"error": f"Erreur de validation Pandera : {str(e)}"}

        # ✅ Puis on renomme pour le modèle
        df.rename(columns=COLUMN_MAPPING, inplace=True)
        df = df.reindex(columns=features, fill_value=0)

        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}

    @bentoml.api(route="/ping", name="ping")
    def ping(self) -> dict:
        return {"status": "alive 🚀"}

    @bentoml.api(route="/model_info", name="model_info")
    def model_info(self) -> dict:
        return {
            "model_name": model_ref.tag.name,
            "model_version": model_ref.tag.version,
            "n_features": len(features),
            "feature_names": features,
        }
