# scripts-bento/save_model.py

import pandas as pd
import numpy as np
import bentoml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 📂 Chargement des données nettoyées
df = pd.read_csv("data/feature_engineered_cleaned_for_bento.csv")

# 🎯 Définition de la cible
target = "SiteEnergyUse(kBtu)"

# 🧹 Suppression des lignes où la cible est manquante
df = df.dropna(subset=[target])

# 🎯 Séparation X / y
X = df.drop(columns=[target])
y = df[target]

# ✂️ Split train/test (même si on ne s'en sert pas ici)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 Sauvegarde avec les features utilisées
bentoml.sklearn.save_model(
    "random_forest_energy",
    model,
    custom_objects={"features": list(X.columns)}
)

print("✅ Modèle entraîné et sauvegardé avec succès.")
