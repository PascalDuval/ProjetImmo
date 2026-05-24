# Conso Batiment (Seattle 2016)

Guide tres explicite pour tester le repo de bout en bout avec Poetry:
1) execution des notebooks dans l ordre,
2) entrainement/sauvegarde du modele,
3) lancement API BentoML,
4) test HTTP.

## 1) Objectif du projet

Predire la consommation energetique SiteEnergyUse(kBtu) des batiments de Seattle (benchmark public 2016), avec un pipeline ML complet et une exposition via API.

## 2) Prerequis

- Python 3.11
- Git
- Poetry
- Jupyter (installe via Poetry)

Verification rapide:

```bash
python --version
git --version
poetry --version
```

## 3) Clonage et installation

```bash
git clone https://github.com/PascalDuval/ForecastingEC4Seattle.git
cd ForecastingEC4Seattle/old-projet
poetry env use 3.11
poetry install
```

Execution des commandes:
- soit avec poetry shell
- soit (recommande) avec poetry run

## 4) Arborescence claire (renommee)

```text
old-projet/
├─ data/
│  ├─ 2016_Building_Energy_Benchmarking.csv
│  ├─ feature_engineered_2016_energySpec.csv
│  └─ ... jeux intermediaires + exports
├─ scripts_bento/
│  ├─ save_model.py
│  ├─ service.py
│  └─ test_api.py
├─ ml01_exploration_donnees.ipynb
├─ ml02_feature_engineering.ipynb
├─ ml03_modelisation_evaluation.ipynb
├─ bentofile.yaml
├─ pyproject.toml
└─ README.md
```

## 5) Mode de test repo: enchainement des notebooks

Objectif: rejouer le workflow data/ML dans un ordre deterministic.

Ordre obligatoire:
1. ml01_exploration_donnees.ipynb
2. ml02_feature_engineering.ipynb
3. ml03_modelisation_evaluation.ipynb

### Option A: execution manuelle (Jupyter)

```bash
poetry run jupyter notebook
```

Puis ouvrir chaque notebook dans l ordre 01 -> 02 -> 03 et faire Run All.

### Option B: execution automatisee (recommandee pour test)

Depuis old-projet:

```bash
poetry run jupyter nbconvert --to notebook --execute --inplace ml01_exploration_donnees.ipynb
poetry run jupyter nbconvert --to notebook --execute --inplace ml02_feature_engineering.ipynb
poetry run jupyter nbconvert --to notebook --execute --inplace ml03_modelisation_evaluation.ipynb
```

Ce mode est utile pour valider rapidement qu un clone propre peut reproduire le pipeline sans intervention manuelle.

## 6) Ce que fait chaque notebook (detail explicite)

### ml01_exploration_donnees.ipynb

Ce notebook fait l EDA:
1. charge le jeu brut Seattle 2016,
2. inspecte qualite des donnees (missing/NaN/null),
3. etudie distributions de variables,
4. analyse correlation/corr et heatmap,
5. repere des outliers (boxplot, distribution),
6. prepare les premieres decisions de nettoyage.

Sortie principale: comprehension du dataset et regles de preparation a appliquer ensuite.

### ml02_feature_engineering.ipynb

Ce notebook transforme les donnees:
1. nettoyage complementaire (dropna et filtres),
2. creation de variables derivees (ex: log_surface),
3. encodage categoriel (get_dummies),
4. traitement outliers (methodes de type IQR),
5. normalisation/encodage selon les besoins,
6. export des jeux prets pour modelisation.

Sortie principale: fichiers feature_engineered utilises par l entrainement.

### ml03_modelisation_evaluation.ipynb

Ce notebook compare plusieurs modeles et mesure leurs performances.

Algorithmes mentionnes/utilises dans le notebook:
- LinearRegression
- Ridge
- ElasticNet
- RandomForest
- GradientBoosting
- XGBoost
- LightGBM

Metriques d evaluation:
- R2
- MAE
- RMSE

Sortie principale: choix du modele final et des hyperparametres pertinents.

## 7) Ce que font les scripts Python

### scripts_bento/save_model.py

1. Charge data/feature_engineered_2016_energySpec.csv.
2. Garde les variables numeriques et supprime lignes incompletes.
3. Separe target SiteEnergyUse(kBtu) et variables explicatives.
4. Fait train/test split.
5. Lance RandomizedSearchCV sur RandomForestRegressor.
6. Evalue train/test (R2, MAE, RMSE).
7. Sauvegarde dans BentoML sous random_forest_energy.
8. Exporte aussi random_forest_optimized_model.joblib.

Commande:

```bash
poetry run python scripts_bento/save_model.py
```

### scripts_bento/service.py

Expose une API BentoML:
- GET /ping
- POST /predict

Le service:
1. charge le modele BentoML,
2. valide l entree (Pydantic + Pandera),
3. reindexe les colonnes selon les features du modele,
4. retourne prediction_kBtu.

Commande:

```bash
poetry run bentoml serve scripts_bento.service:EnergyService --reload
```

### scripts_bento/test_api.py

Envoie une requete HTTP de test sur localhost:3000/predict et affiche status + reponse JSON.

Commande:

```bash
poetry run python scripts_bento/test_api.py
```

## 8) Procedure complete de validation (checklist)

1. Rejouer notebooks 01 -> 02 -> 03.
2. Entrainer/sauvegarder le modele:

```bash
poetry run python scripts_bento/save_model.py
```

3. Demarrer API:

```bash
poetry run bentoml serve scripts_bento.service:EnergyService --reload
```

4. Dans un second terminal, tester API:

```bash
poetry run python scripts_bento/test_api.py
```

## 9) Payload attendu pour /predict

```json
{
  "log_surface": 3.8,
  "PourcentElec": 45.0,
  "Use_Office": 1,
  "Use_Other": 0,
  "Use_Retail": 0,
  "Use_Warehouse": 0,
  "Use_Unknown": 0
}
```

## 10) Focus explicite sur les algorithmes ML

Pourquoi plusieurs algorithmes:
- etablir une baseline lineaire (LinearRegression),
- tester regularisation (Ridge, ElasticNet),
- tester modeles non lineaires ensemblistes (RandomForest, GradientBoosting),
- comparer des boosters performants sur tabulaire (XGBoost, LightGBM).

Lecture pratique des metriques:
- R2 plus eleve -> meilleure variance expliquee,
- MAE plus faible -> erreur absolue moyenne plus faible,
- RMSE plus faible -> penalise plus fortement les grosses erreurs.

Decision recommandee:
- choisir le meilleur compromis RMSE/MAE/R2 sur validation,
- verifier l ecart train vs test pour eviter le surapprentissage,
- conserver la version retenue dans BentoML pour inference stable.

## 11) Erreurs frequentes

- random_forest_energy:latest introuvable:
  - executer scripts_bento/save_model.py avant de lancer l API.

- connection refusee sur /predict:
  - verifier que bentoml serve tourne bien en parallele.

- erreur de version Python:
  - reexecuter poetry env use 3.11 puis poetry install.
