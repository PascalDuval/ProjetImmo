# Conso Batiment (Seattle 2016)

README de prise en main tres explicite pour executer ce projet avec Poetry, du clonage au test de l API BentoML.

## 1) Ce que fait ce projet

Objectif ML: predire la consommation energetique SiteEnergyUse(kBtu) des batiments de Seattle (jeu public 2016), puis exposer la prediction via une API BentoML.

Pipeline global:
1. Preparer/transformer les donnees (notebooks).
2. Entrainer un RandomForest avec recherche d hyperparametres.
3. Sauvegarder le modele dans BentoML.
4. Demarrer une API locale et tester une prediction.

## 2) Prerequis

- Windows PowerShell, macOS Terminal ou Linux shell.
- Python 3.11 installe.
- Git installe.
- Poetry installe.

Verification rapide:

```bash
python --version
git --version
poetry --version
```

## 3) Cloner le projet et se placer dans old-projet

Si vous n avez pas encore clone le depot:

```bash
git clone https://github.com/PascalDuval/ForecastingEC4Seattle.git
cd ForecastingEC4Seattle/old-projet
```

Si le depot est deja clone, placez-vous simplement dans le dossier old-projet.

## 4) Installer l environnement avec Poetry

Depuis le dossier old-projet:

```bash
poetry env use 3.11
poetry install
```

Ensuite, deux facons de lancer les commandes:

- Soit ouvrir un shell Poetry:

```bash
poetry shell
```

- Soit prefixer chaque commande avec poetry run (methode la plus robuste):

```bash
poetry run python --version
```

## 5) Arborescence utile (version simplifiee)

```text
old-projet/
├─ data/
│  ├─ 2016_Building_Energy_Benchmarking.csv
│  ├─ feature_engineered_2016_energySpec.csv
│  └─ ... autres jeux intermediaires et exports
├─ scripts_bento/
│  ├─ save_model.py
│  ├─ service.py
│  └─ test_api.py
├─ analyse_exploratoire.ipynb
├─ feature andMore.ipynb
├─ modeles.ipynb
├─ bentofile.yaml
├─ pyproject.toml
└─ README.md
```

## 6) Ce que fait chaque script

### scripts_bento/save_model.py

Ce script:
1. Charge data/feature_engineered_2016_energySpec.csv.
2. Garde les colonnes numeriques et supprime les lignes incompletes.
3. Separe la cible SiteEnergyUse(kBtu) du reste des variables.
4. Fait un split train/test (80/20).
5. Lance une RandomizedSearchCV sur RandomForestRegressor.
6. Evalue le modele (R2, MAE, RMSE) sur train et test.
7. Sauvegarde le modele dans BentoML sous le tag random_forest_energy.
8. Sauvegarde aussi un fichier local random_forest_optimized_model.joblib.

Commande:

```bash
poetry run python scripts_bento/save_model.py
```

### scripts_bento/service.py

Ce script definit le service BentoML:
1. Charge le modele random_forest_energy:latest depuis le store BentoML.
2. Defini le schema d entree (Pydantic + Pandera).
3. Expose les endpoints:
   - GET /ping
   - POST /predict
4. Aligne les colonnes d entree avec les features du modele.
5. Retourne prediction_kBtu.

Commande de lancement:

```bash
poetry run bentoml serve scripts_bento.service:EnergyService --reload
```

### scripts_bento/test_api.py

Ce script envoie une requete HTTP de test vers http://127.0.0.1:3000/predict et affiche:
- le status code,
- la reponse JSON.

Commande:

```bash
poetry run python scripts_bento/test_api.py
```

Important: ce test fonctionne seulement si le service BentoML tourne deja.

## 7) Procedure complete recommandee

Depuis old-projet:

1. Installer les dependances:

```bash
poetry install
```

2. Entrainer et sauvegarder le modele:

```bash
poetry run python scripts_bento/save_model.py
```

3. Demarrer l API (laisser ce terminal ouvert):

```bash
poetry run bentoml serve scripts_bento.service:EnergyService --reload
```

4. Dans un second terminal, tester l API:

```bash
poetry run python scripts_bento/test_api.py
```

## 8) Exemple de payload attendu par /predict

Le endpoint /predict attend un JSON plat (pas de cle data imbriquee), par exemple:

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

## 9) Construire un Bento (optionnel)

Une fois le modele sauvegarde:

```bash
poetry run bentoml build
```

Le fichier bentofile.yaml controle le service cible, les modeles et les fichiers inclus.

## 10) Probleme frequents et corrections rapides

- Erreur random_forest_energy:latest introuvable:
  - Relancer poetry run python scripts_bento/save_model.py.

- Erreur connexion refusee sur /predict:
  - Verifier que poetry run bentoml serve scripts_bento.service:EnergyService --reload tourne dans un autre terminal.

- Python pas en 3.11:
  - Forcer poetry env use 3.11 puis poetry install.

- Donnees manquantes:
  - Verifier la presence de data/feature_engineered_2016_energySpec.csv.
