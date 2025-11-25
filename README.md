# Projet 6 - Conso Batiment (Seattle 2016)

## Objectif
Predire la consommation energetique `SiteEnergyUse(kBtu)` des batiments du benchmark public de Seattle 2016 en partant d un jeu de donnees nettoye et en exposant un service BentoML.

## Organisation des fichiers
- `analyse_exploratoire.ipynb` : exploration initiale, nettoyage de base et premiers graphiques.
- `feature andMore.ipynb` : feature engineering (encodages, normalisation, traitement des valeurs extremes) et export des jeux prets pour le ML.
- `modeles.ipynb` : essais et comparaison de modeles (regressions, XGBoost, reseaux de neurones), selection du RandomForest final.
- `data/` : jeux de donnees. `2016_Building_Energy_Benchmarking.csv` (brut), versions purgees/ML/intermediaires, `feature_engineered_cleaned_for_bento.csv` (jeu final consomme par BentoML), fichiers d outliers (`top_extreme_intensity_outliers.csv`, `outliers_surface_energy.csv`).
- `scripts_bento/save_model.py` : entraine un `RandomForestRegressor` sur `feature_engineered_cleaned_for_bento.csv` et sauvegarde le modele dans le store BentoML avec la liste des features.
- `scripts_bento/service.py` : service BentoML 1.4.29 expose `/predict`, `/ping` et `/model_info` avec validation Pydantic + Pandera et mapping des colonnes avant inference.
- `scripts_bento/service - back.py` : version minimale de service (legacy) gardee en reference.
- `scripts_bento/test_save_model.py` : smoke test local pour charger le modele sauvegarde et generer une prediction a partir d un DataFrame.
- `scripts_bento/test_api.py` : appel HTTP exemple sur `http://127.0.0.1:3000/predict` (format JSON BentoML attendu `{"data": {...}}`).
- `bentofile.yaml` : configuration BentoML (service cible, dependances et fichiers embarques) utilisee par `bentoml build`.
- `pyproject.toml` / `poetry.lock` : dependances du projet (Python 3.11, scikit-learn, pandas, pandera, bentoml, etc.).

## Prerequis et installation
- Python 3.11 et Poetry installes localement.
- Recuperer les donnees deja presentes dans `data/` (pas d appel reseau necessaire).
- Installer l environnement :
  ```bash
  poetry install
  poetry shell   # ou prefixer chaque commande par "poetry run"
  ```

## Entrainement et sauvegarde du modele BentoML
1) Verifier/mettre a jour les features via les notebooks si besoin (les sorties ecrivent dans `data/`).
2) Enregistrer le modele RandomForest :
   ```bash
   poetry run python scripts_bento/save_model.py
   ```
   Le modele est stocke dans le Bento store sous le tag `random_forest_energy` avec les noms de colonnes utilises.

## Service BentoML
- Lancer le serveur local avec rechargement :
  ```bash
  poetry run bentoml serve scripts_bento.service:EnergyService --reload
  ```
- Endpoints exposes :
  - `POST /predict` : JSON avec la cle `data`. Exemple :
    ```json
    {
      "data": {
        "BuildingAge": 10,
        "log_surface": 3.6,
        "has_parking": 1,
        "Use_Retail": 1,
        "Use_Office": 0,
        "Use_Other": 0,
        "Use_Warehouse": 0,
        "Use_Unknown": 0
      }
    }
    ```
    Retour : `{ "prediction": <float> }` apres validation Pydantic/Pandera et mapping des colonnes (`Use_Retail` -> `Use_Retail Store`, etc.).
  - `GET /ping` : verifie que le service repond.
  - `GET /model_info` : nom/version du modele et liste des features attendues.
- Construire un bento reproductible : `poetry run bentoml build` (utilise `bentofile.yaml`).

## Tests rapides
- Verifier la prediction hors API : `poetry run python scripts_bento/test_save_model.py`.
- Tester l endpoint `/predict` une fois le serveur lance : `poetry run python scripts_bento/test_api.py`.

## Notes
- Les scripts supposent que le fichier `data/feature_engineered_cleaned_for_bento.csv` est present. Regenerer via les notebooks si besoin.
- L environnement virtuel `.venv/` est optionnel si vous utilisez Poetry; sinon installez les dependances via `pip` en lisant `pyproject.toml`.
