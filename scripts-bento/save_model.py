# scripts-bento/save_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import bentoml

# Chargement des données
df = pd.read_csv("data/feature_engineered_2016_energy1.csv")

# Sélection des variables
target = 'SiteEnergyUse(kBtu)'
features_to_use = [
    'BuildingAge', 'log_surface', 'has_parking'
] + [col for col in df.columns if col.startswith('Usage_')]

df_model = df[features_to_use + [target]].dropna()
X = df_model[features_to_use]
y = df_model[target]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde avec BentoML
bentoml.sklearn.save_model(
    "random_forest_energy_model",
    model,
    signatures={"predict": {"batchable": True}},
    custom_objects={"features": features_to_use}
)
