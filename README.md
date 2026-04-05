# Olist Customer Segmentation

<p align="center">
  <img src="assets/images/olist_logo.png" alt="Olist Logo" width="300"/>
</p>

> Segmentation non supervisée des clients d'Olist afin de permettre
> à l'équipe marketing de personnaliser ses campagnes de communication
> selon le profil comportemental de chaque groupe de clients.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.14-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Tests](https://img.shields.io/badge/Tests-49%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-100%25%20logic-brightgreen)
![Segments](https://img.shields.io/badge/Segments-4-blue)
![Clients](https://img.shields.io/badge/Clients-93%20358-blue)
![CI/CD](https://github.com/thiernodaoudaly/olist-customer-segmentation/actions/workflows/ci.yml/badge.svg)

</div>

---

## Table des matières

1. [Contexte business](#contexte-business)
2. [Architecture du projet](#architecture-du-projet)
3. [Dataset Olist](#dataset-olist)
4. [Installation pas à pas](#installation-pas-à-pas)
5. [Lancement de l'environnement](#lancement-de-lenvironnement)
6. [Description de chaque fichier](#description-de-chaque-fichier)
7. [Pipeline MLOps — ordre d'exécution](#pipeline-mlops--ordre-dexécution)
8. [Résultats](#résultats)
9. [Segmentation clients](#segmentation-clients)
10. [Tests unitaires](#tests-unitaires)
11. [CI/CD GitHub Actions](#cicd-github-actions)
12. [API FastAPI](#api-fastapi)
13. [Dashboard marketing](#dashboard-marketing)
14. [Contrat de maintenance](#contrat-de-maintenance)
15. [Auteur](#auteur)

---

## Contexte business

Olist est une entreprise brésilienne fondée en 2015 qui connecte
les petits marchands aux grandes marketplaces en ligne
(Amazon, Mercado Livre, Americanas, etc.) via une solution SaaS.
Valorisée à 1,5 milliard de dollars en 2021, elle gère plus de
90 000 clients actifs aux comportements très variés.

**Problème :** L'équipe marketing ne peut pas appliquer une stratégie
de communication unique à tous les clients. Un client qui commande
tous les mois à 300 BRL n'a pas le même profil qu'un acheteur unique
à 50 BRL insatisfait.

**Solution :** Ce projet vise à identifier automatiquement des groupes
de clients homogènes à partir de leurs données comportementales
(historique d'achats, satisfaction, délais de livraison) pour permettre
des campagnes ciblées et mesurables.

**4 segments identifiés :**

| Segment | Nom | Taille | Stratégie |
|---|---|---|---|
| 0 | Nouveaux satisfaits | 49% | Fidélisation |
| 1 | Clients dormants | 33% | Réactivation |
| 2 | Clients insatisfaits | 15% | Récupération |
| 3 | VIP multi-acheteurs | 3% | Programme VIP |

---

## Architecture du projet

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE MLOPS                        │
│                                                          │
│  data/raw/        →   src/data/load_data.py              │
│  (CSV Olist)          Chargement des 5 tables            │
│      ↓                                                   │
│  src/features/    →   build_features.py                  │
│  build_features       RFM enrichi (récence, fréquence,   │
│                       montant, satisfaction, livraison)   │
│      ↓                                                   │
│  src/models/      →   train.py                           │
│  train                K-Means K=4 + MLflow logging       │
│      ↓                                                   │
│  models/          →   kmeans_model.pkl + scaler.pkl      │
│  artefacts            Sauvegardés localement + MLflow    │
│      ↓                                                   │
│  src/api/         →   main.py (FastAPI)                  │
│  main                 3 endpoints REST + Swagger         │
│      ↓                                                   │
│  dashboard/       →   index.html                         │
│  index.html           Interface marketing                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUALITÉ & CI/CD                       │
│                                                          │
│  tests/           →   49 tests unitaires Pytest          │
│                       couverture 100% logique métier     │
│      ↓                                                   │
│  .github/         →   GitHub Actions                     │
│  workflows/ci.yml     Black + Flake8 + Pytest            │
│                       exécutés à chaque git push         │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset Olist

Les données comprennent l'historique des commandes de 2016 à 2018,
avec 100 000 commandes réelles anonymisées, gracieusement fournies
par Olist sous licence CC BY-NC-SA 4.0 et disponibles sur Kaggle.

**Lien :** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

<p align="center">
  <img src="assets/images/olist_database_schema.webp"
       alt="Schéma BDD Olist" width="700"/>
</p>

### Les 5 tables utilisées

| Fichier CSV | Lignes | Colonnes | Rôle dans le projet |
|---|---|---|---|
| `olist_orders_dataset.csv` | 99 441 | 8 | Table centrale — statuts et dates |
| `olist_order_items_dataset.csv` | 112 650 | 7 | Prix et frais de port par article |
| `olist_order_payments_dataset.csv` | 103 886 | 5 | Mode et montant des paiements |
| `olist_order_reviews_dataset.csv` | 99 224 | 7 | Notes et commentaires clients |
| `olist_customers_dataset.csv` | 99 441 | 5 | Localisation des clients |

### Statistiques clés du dataset

- **Période couverte :** septembre 2016 → août 2018
- **Commandes livrées :** 96 478 (97%)
- **Clients uniques :** 93 358 (après feature engineering)
- **97% des clients** n'ont commandé qu'une seule fois
- **Score satisfaction moyen :** 4.09/5
- **92% des livraisons** arrivent en avance (moyenne -12 jours)
- **Montant moyen par commande :** 159 BRL (médiane : 105 BRL)

---

## Installation pas à pas

### Prérequis système

Avant de commencer, vérifiez que ces outils sont installés :

```bash
python --version       # >= 3.13 recommandé
git --version          # >= 2.0
docker --version       # >= 24.0
docker compose version # >= 2.0
```

### Étape 1 — Cloner le repository

```bash
git clone https://github.com/thiernodaoudaly/olist-customer-segmentation.git
cd olist-customer-segmentation
```

### Étape 2 — Créer l'environnement virtuel Python

```bash
# Windows
py -3.13 -m venv venv
venv\Scripts\activate

# Linux / macOS
python3.13 -m venv venv
source venv/bin/activate
```

Votre prompt doit afficher `(venv)` au début.

### Étape 3 — Installer les dépendances

```bash
# Mise à jour pip en premier
python -m pip install --upgrade pip

# Installation de toutes les dépendances
pip install -r requirements.txt

# Installation du package src/ en mode editable
# Permet d'importer src.data, src.features, etc. dans les notebooks
pip install -e .
```

### Étape 4 — Vérifier l'installation

```bash
python scripts/check_env.py
```

Résultat attendu : tous les packages marqués OK.

### Étape 5 — Télécharger le dataset Olist

Téléchargez le dataset depuis Kaggle :
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Extrayez tous les fichiers CSV dans `data/raw/` :

```
data/raw/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_customers_dataset.csv
├── olist_sellers_dataset.csv
├── olist_products_dataset.csv
├── olist_geolocation_dataset.csv
└── product_category_name_translation.csv
```

### Étape 6 — Initialiser DVC

DVC (Data Version Control) versionne les fichiers de données
trop lourds pour Git (CSV, modèles .pkl).

```bash
dvc init
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc/ .dvcignore
git commit -m "data: ajout dataset Olist via DVC"
```

### Étape 7 — Lancer le pipeline d'entraînement

```bash
python src/models/train.py
```

Ce script exécute tout le pipeline en une commande :
chargement → features → scaling → K-Means → MLflow logging.

### Étape 8 — Lancer l'API

```bash
# Terminal 1 — API FastAPI
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Interface MLflow (optionnel)
mlflow ui --port 5000
```

### Étape 9 — Ouvrir le dashboard marketing

Ouvrez `dashboard/index.html` dans votre navigateur.
L'API doit tourner sur localhost:8000 pour que le dashboard fonctionne.

---

## Lancement de l'environnement

### Option A — Avec Docker (recommandé pour la production)

Docker garantit la reproductibilité totale — même résultat sur
n'importe quelle machine, quel que soit l'OS.

```bash
# Construire et lancer tous les services
docker-compose up --build
```

| Service | URL | Description |
|---|---|---|
| API FastAPI | http://localhost:8000 | Endpoint de segmentation |
| Swagger UI | http://localhost:8000/docs | Documentation interactive |
| MLflow UI | http://localhost:5000 | Tracking des expériences |

Pour arrêter :

```bash
docker-compose down
```

### Option B — Sans Docker (développement local)

```bash
# Activer le venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # Linux/macOS

# Lancer l'API
uvicorn src.api.main:app --reload --port 8000

# Lancer MLflow dans un terminal séparé
mlflow ui --port 5000
```

### Option C — Reproduire le pipeline avec DVC

DVC garantit que le pipeline est exécuté dans le bon ordre
et ne rejoue que les étapes modifiées.

```bash
dvc repro
```

---

## Description de chaque fichier

### Notebooks

#### `notebooks/segmentation_01_eda.ipynb`
**Rôle :** Analyse exploratoire des données (EDA).

Ce notebook explore les 5 tables du dataset Olist pour comprendre
la structure des données avant toute modélisation. Il répond aux
questions suivantes : quelles sont les valeurs manquantes ? Quelles
sont les distributions des variables clés ? Quelles transformations
sont nécessaires avant le clustering ?

**Ce qu'il produit :**
- Distribution des statuts de commandes
- Analyse de la fréquence d'achat (97% acheteurs uniques)
- Distribution des montants (asymétrique → log-transformation)
- Score de satisfaction (bimodal : pics en 1 et 5)
- Délais de livraison (92% en avance, moyenne -12 jours)
- Décision finale : features RFM enrichies à construire

**Pour l'exécuter :**

```bash
jupyter notebook notebooks/segmentation_01_eda.ipynb
```

---

#### `notebooks/segmentation_02_modeling.ipynb`
**Rôle :** Feature engineering, modélisation et tracking MLflow.

Ce notebook construit les features RFM enrichies, teste plusieurs
valeurs de K avec K-Means, compare les résultats dans MLflow
et produit le modèle final avec les 4 segments nommés.

**Ce qu'il produit :**
- Features engineerées pour 93 358 clients
- 9 runs MLflow (K=2 à K=10) + 1 run final K=4
- Graphiques Elbow et Silhouette Score
- 4 segments nommés avec stratégies marketing
- Fichier `data/processed/rfm_features.csv`

**Pour l'exécuter :**

```bash
jupyter notebook notebooks/segmentation_02_modeling.ipynb
```

---

#### `notebooks/segmentation_03_simulation.ipynb`
**Rôle :** Simulation de la stabilité des segments dans le temps.

Ce notebook découpe les données en 7 trimestres (2017Q1 → 2018Q3),
entraîne K-Means sur chaque trimestre et mesure si les segments
restent cohérents avec le temps via l'ARI et la variance du silhouette.

**Ce qu'il produit :**
- ARI entre trimestres consécutifs (label switching détecté)
- Variance du silhouette sur 7 trimestres (std = 0.022 → stable)
- Graphique d'évolution de la taille des clusters
- Contrat de maintenance : mise à jour trimestrielle recommandée

**Pour l'exécuter :**

```bash
jupyter notebook notebooks/segmentation_03_simulation.ipynb
```

---

### Code source (`src/`)

#### `src/data/load_data.py`
**Rôle :** Chargement des tables CSV Olist en mémoire.

Ce module fournit deux fonctions :

- `load_olist(data_dir)` — charge les 5 tables nécessaires dans un
  dictionnaire `{nom_table: DataFrame}`. Lève une `FileNotFoundError`
  explicite si un fichier manque, avec le lien Kaggle pour télécharger.

- `get_merged_dataset(data_dir)` — charge et merge toutes les tables
  en un seul DataFrame via des LEFT JOIN sur `order_id` et `customer_id`.

**Pourquoi `customer_unique_id` et pas `customer_id` ?**
Chaque commande a un `customer_id` unique — un même client a donc
plusieurs `customer_id` différents pour ses différentes commandes.
`customer_unique_id` identifie le vrai client physique.

**Utilisation :**

```python
from src.data.load_data import load_olist
dfs = load_olist("data/raw")
orders = dfs["orders"]       # 99 441 lignes × 8 colonnes
customers = dfs["customers"] # 99 441 lignes × 5 colonnes
```

---

#### `src/features/build_features.py`
**Rôle :** Construction des features RFM enrichies.

Ce module contient une fonction par feature (principe de responsabilité
unique — chaque fonction est testable indépendamment) :

| Fonction | Feature produite | Transformation |
|---|---|---|
| `compute_recency()` | Jours depuis dernier achat | Aucune |
| `compute_frequency()` | Nb de commandes | log1p |
| `compute_monetary()` | Montant total (prix + frais de port) | log1p |
| `compute_satisfaction()` | Score moyen des avis | Médiane pour NaN |
| `compute_delivery()` | Délai réel vs estimé | clip(-30, +30) |
| `build_features()` | Assemble tout en un DataFrame | Toutes ci-dessus |

**Pourquoi log1p sur frequency et monetary ?**
K-Means calcule des distances euclidiennes. Sans transformation,
un montant de 13 664 BRL dominerait tous les calculs de distance.
log1p(13664) = 9.52 vs log1p(105) = 4.65 — écart ×2 au lieu de ×130.

**Pourquoi la médiane pour imputer review_score_mean ?**
La distribution des scores est bimodale (pics en 1 et 5).
La moyenne (4.09) ne représente quasiment aucun client réel.
La médiane (5.0) correspond au comportement majoritaire.

**Utilisation :**

```python
from src.features.build_features import build_features
from src.data.load_data import load_olist

dfs = load_olist("data/raw")
df_features = build_features(dfs)
# Retourne 93 358 lignes × 8 colonnes
```

---

#### `src/models/train.py`
**Rôle :** Script standalone d'entraînement du modèle complet.

Ce script exécute le pipeline complet en une commande :
`load_olist()` → `build_features()` → `StandardScaler` →
`KMeans` → `mlflow.log` → sauvegarde `.pkl`.

Il accepte des arguments CLI pour être paramétrable :

```bash
# Entraînement par défaut (K=4)
python src/models/train.py

# Avec paramètres personnalisés
python src/models/train.py --n-clusters 5 --run-name mon_run

# Avec dossier de données différent
python src/models/train.py --data-dir data/raw --random-state 0
```

**Ce qu'il sauvegarde :**
- `data/processed/rfm_features.csv` — features de tous les clients
- `models/kmeans_model.pkl` — modèle K-Means sérialisé
- `models/scaler.pkl` — StandardScaler sérialisé
- Run MLflow avec paramètres, métriques et artefacts

---

#### `src/models/predict.py`
**Rôle :** Prédiction du segment d'un client individuel.

Ce module fournit `predict_segment(client_features, kmeans_model, scaler)`
qui prend les données brutes d'un client, applique les transformations
(log1p), scale avec le StandardScaler entraîné, et retourne le segment.

**Pourquoi `scaler.transform()` et pas `scaler.fit_transform()` ?**
`fit_transform()` recalculerait mean/std sur un seul client — résultat
absurde. `transform()` utilise les paramètres appris lors de l'entraînement
sur les 93 358 clients.

**Utilisation :**

```python
from src.models.predict import predict_segment
import pickle

with open("models/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

result = predict_segment(
    client_features={
        "recency": 135,
        "frequency": 1,
        "monetary": 147.0,
        "review_score_mean": 4.5,
        "delivery_delay_mean": -12.0,
    },
    kmeans_model=kmeans,
    scaler=scaler,
)
# {"segment_id": 0, "segment_name": "Nouveaux satisfaits", "strategy": "..."}
```

---

#### `src/api/main.py`
**Rôle :** API REST FastAPI exposant le modèle de segmentation.

L'API charge automatiquement le modèle K-Means et le scaler
depuis MLflow au démarrage (via le `lifespan` context manager).

**3 endpoints disponibles :**

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check — API et modèle opérationnels ? |
| GET | `/segments` | Liste des 4 segments avec stratégies |
| POST | `/predict` | Prédit le segment d'un client |

**Validation automatique avec Pydantic :**
Si un champ manque ou a un mauvais type, FastAPI retourne
une erreur 422 avec un message clair avant même d'appeler le modèle.

**Pour lancer l'API :**

```bash
uvicorn src.api.main:app --reload --port 8000
```

**Documentation interactive :** http://localhost:8000/docs

---

### Scripts utilitaires (`scripts/`)

#### `scripts/check_env.py`
**Rôle :** Vérifie que tous les packages du requirements.txt
sont bien installés avec les bonnes versions.

```bash
python scripts/check_env.py
```

Affiche un tableau avec le statut (OK / MANQUANT) et la version
de chaque package. Pratique après une nouvelle installation
ou avant un déploiement.

---

#### `scripts/load_data.py`
**Rôle :** Téléchargement automatique du dataset Olist depuis Kaggle
via l'API Kaggle officielle.

```bash
# Installation préalable
pip install kaggle
# Placer kaggle.json dans ~/.kaggle/kaggle.json

# Téléchargement
python scripts/load_data.py
python scripts/load_data.py --data-dir data/raw
python scripts/load_data.py --force  # re-télécharge si déjà présent
```

---

### Tests (`tests/`)

#### `tests/test_sanity.py`
**Rôle :** Tests de sanité — vérifie que l'environnement de test
fonctionne et que le package `src/` est importable.

3 tests : Python fonctionne, packages installés, src/ importable.

---

#### `tests/test_load_data.py`
**Rôle :** Tests unitaires pour `src/data/load_data.py`.

10 tests qui vérifient :
- `load_olist()` retourne un dictionnaire de DataFrames
- Les 5 tables attendues sont présentes
- Les noms de tables sont simplifiés (`orders`, pas `olist_orders_dataset`)
- Une `FileNotFoundError` est levée si un fichier manque
- `get_merged_dataset()` retourne un DataFrame sans doublons

**Technique :** Les tests utilisent des fixtures Pytest avec
`tmp_path` (dossier temporaire automatiquement supprimé après)
et des mini-CSV de 3-5 lignes — pas les vrais CSV Olist.
Cela garantit des tests rapides (< 1s) et indépendants du filesystem.

---

#### `tests/test_build_features.py`
**Rôle :** Tests unitaires pour `src/features/build_features.py`.

21 tests qui vérifient chaque fonction individuellement :
- `compute_recency()` — récence basée sur la DERNIÈRE commande
- `compute_frequency()` — client avec 2 commandes → frequency=2
- `compute_monetary()` — montant inclut les frais de port
- `compute_delivery()` — délai clippé entre -30 et +30
- `build_features()` — aucun NaN, une ligne par client, log1p positif

---

#### `tests/test_predict.py`
**Rôle :** Tests unitaires pour `src/models/predict.py`.

15 tests qui vérifient la logique de `predict_segment()` :
- Retourne un dict avec les bonnes clés
- `segment_id` est un entier entre 0 et 3
- `log1p` est bien appliqué sur frequency et monetary
- Le scaler et le modèle sont appelés exactement une fois
- Les 4 segments sont tous atteignables

**Technique :** Utilise `MagicMock` de `unittest.mock` pour simuler
le modèle KMeans et le StandardScaler sans charger MLflow.
`mock_kmeans.predict.return_value = np.array([0])` → retourne
toujours le cluster 0, quelle que soit l'entrée.

---

### Configuration

#### `requirements.txt`
Liste de toutes les dépendances Python avec leurs versions minimales.
Organisé par catégorie : données, visualisation, ML, MLOps, API,
qualité de code, tests.

```bash
pip install -r requirements.txt
```

---

#### `setup.py`
Rend le dossier `src/` installable comme un package Python.
Sans ce fichier, `from src.data.load_data import load_olist`
échouerait avec `ModuleNotFoundError`.

```bash
pip install -e .   # installe src/ en mode editable
```

---

#### `pyproject.toml`
Configuration centralisée des outils de qualité de code :
- `[tool.black]` — formatage automatique, max 88 caractères par ligne
- `[tool.pytest.ini_options]` — dossier de tests, pattern de fichiers

---

#### `.flake8`
Configuration de Flake8 (linter PEP8) :
- `max-line-length = 88` — aligné sur Black
- `extend-ignore = E501` — délègue la longueur de ligne à Black
- `exclude` — ignore venv/, notebooks/, .git/

---

#### `Dockerfile`
Image Docker pour l'API FastAPI.

```
FROM python:3.13-slim      # image légère ~50Mo vs 900Mo pour python:3.13
WORKDIR /app               # répertoire de travail dans le container
COPY requirements.txt .    # copié AVANT le code → cache Docker préservé
RUN pip install ...        # installation des dépendances
COPY . .                   # code source copié APRÈS les deps
RUN pip install -e .       # installation de src/ comme package
EXPOSE 8000                # port exposé
CMD ["uvicorn", ...]       # commande de démarrage
```

---

#### `docker-compose.yml`
Orchestre deux services qui se voient sur un réseau Docker interne :
- **api** — FastAPI sur le port 8000
- **mlflow** — serveur MLflow sur le port 5000

Les `volumes` montent les dossiers locaux dans les containers pour
que les données et les runs MLflow persistent sur votre disque.

---

#### `dvc.yaml`
Définit le pipeline reproductible DVC :

```yaml
stages:
  train:
    cmd: python src/models/train.py    # commande à exécuter
    deps:                              # fichiers dont dépend l'étape
      - src/models/train.py
      - data/raw
    outs:                              # fichiers produits (trackés par DVC)
      - data/processed/rfm_features.csv
      - models/kmeans_model.pkl
```

```bash
dvc repro   # exécute le pipeline, ne rejoue que les étapes modifiées
```

---

#### `params.yaml`
Paramètres du pipeline DVC, modifiables sans toucher au code :

```yaml
train:
  n_clusters: 4
  random_state: 42
  data_dir: data/raw
```

Pour tester K=5 : modifier `n_clusters: 5` puis `dvc repro`.

---

#### `.github/workflows/ci.yml`
Pipeline CI/CD GitHub Actions exécuté à chaque `git push` :

1. Checkout du code
2. Installation Python 3.13
3. Installation des dépendances
4. Black — vérifie le formatage
5. Flake8 — vérifie les règles PEP8
6. Pytest — exécute les 49 tests avec couverture

Si une étape échoue, le badge CI/CD passe au rouge et le push
est marqué comme défaillant.

---

#### `dashboard/index.html`
Interface web pour l'équipe marketing.

Deux onglets :
- **Segments** — affiche les 4 segments chargés depuis l'API GET /segments
- **Prédire un client** — formulaire de saisie des features + appel
  POST /predict + affichage du résultat avec couleur par segment

4 profils types pré-chargés pour tester rapidement chaque segment.

**Prérequis :** uvicorn doit tourner sur localhost:8000.

---

## Pipeline MLOps — ordre d'exécution

Voici l'ordre exact pour reproduire le projet de zéro :

```bash
# 1. Cloner et installer
git clone https://github.com/thiernodaoudaly/olist-customer-segmentation.git
cd olist-customer-segmentation
py -3.13 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .

# 2. Vérifier l'installation
python scripts/check_env.py

# 3. Télécharger les données (manuellement depuis Kaggle)
# Placer les CSV dans data/raw/

# 4. Initialiser DVC
dvc init
dvc add data/raw
git add data/raw.dvc data/.gitignore
git commit -m "data: dataset Olist via DVC"

# 5. Lancer l'EDA (exploration)
jupyter notebook notebooks/segmentation_01_eda.ipynb

# 6. Lancer la modélisation
jupyter notebook notebooks/segmentation_02_modeling.ipynb

# 7. Lancer la simulation de stabilité
jupyter notebook notebooks/segmentation_03_simulation.ipynb

# 8. Entraîner le modèle final via script standalone
python src/models/train.py

# 9. Lancer les tests unitaires
pytest tests/ -v --cov=src --cov-report=term-missing

# 10. Vérifier la qualité du code
black --check src/ tests/
flake8 src/ tests/

# 11. Lancer l'API
uvicorn src.api.main:app --reload --port 8000

# 12. Ouvrir le dashboard
# Ouvrir dashboard/index.html dans le navigateur

# 13. Visualiser les expériences MLflow
mlflow ui --port 5000
# Ouvrir http://localhost:5000

# 14. Reproduire le pipeline avec DVC
dvc repro

# 15. Commiter et pusher (déclenche CI/CD)
git add .
git commit -m "feat: pipeline complet"
git push   # → GitHub Actions exécute les tests automatiquement
```

---

## Résultats

### Comparaison des modèles K-Means (K=2 à K=10)

| Modèle | n_clusters | Silhouette Score | Davies-Bouldin | Retenu |
|---|---|---|---|---|
| K-Means | 2 | 0.5909 | 0.6453 | ❌ trop grossier |
| K-Means | 3 | 0.3471 | 1.1488 | ❌ chute brutale |
| **K-Means** | **4** | **0.2521** | **1.2179** | **✅ retenu** |
| K-Means | 5 | 0.2384 | 1.1555 | ❌ |
| K-Means | 6 | 0.2469 | 1.1540 | ❌ |
| K-Means | 7 | 0.2376 | 1.1505 | ❌ |
| K-Means | 8 | 0.2250 | 1.1901 | ❌ |
| K-Means | 9 | 0.2214 | 1.2218 | ❌ |
| K-Means | 10 | 0.2189 | 1.1831 | ❌ |

**Justification du choix K=4 :**
Le silhouette optimal est obtenu à K=2 (0.59) mais 2 segments
ne permettent pas à l'équipe marketing de personnaliser ses campagnes
de façon significative. K=4 est le premier K stable sur la courbe
Elbow, avec 4 profils clients distincts et actionnables.

### Features utilisées

| Feature | Source | Transformation | Justification |
|---|---|---|---|
| `recency` | orders — purchase_date | Aucune | Jours depuis SNAPSHOT_DATE |
| `frequency_log` | orders — nb commandes | log1p | 97% des clients = 1 commande |
| `monetary_log` | order_items — prix+port | log1p | Distribution très asymétrique |
| `review_score_mean` | order_reviews | Médiane pour NaN | Score 1-5, bimodal |
| `delivery_delay_mean` | orders — réel vs estimé | clip(-30, 30) | Outliers extrêmes |

---

## Segmentation clients

### Profils des 4 segments

| Segment | Nom | Taille | Récence | Montant | Satisfaction | Délai |
|---|---|---|---|---|---|---|
| 0 | Nouveaux satisfaits | 45 997 (49%) | 135j | 147 BRL | 4.66/5 | -12.5j |
| 1 | Clients dormants | 30 556 (33%) | 404j | 168 BRL | 4.58/5 | -14.2j |
| 2 | Clients insatisfaits | 14 004 (15%) | 229j | 188 BRL | 1.60/5 | -3.7j |
| 3 | VIP multi-acheteurs | 2 801 (3%) | 222j | 309 BRL | 4.21/5 | -12.4j |

### Stratégies marketing par segment

**Segment 0 — Nouveaux satisfaits (49%)**
Ces clients ont commandé récemment (135 jours), sont très satisfaits
(4.66/5) mais n'ont commandé qu'une seule fois. C'est le segment
le plus large et le plus facile à convertir en clients fidèles.
**Stratégie :** programme de fidélisation, offre exclusive pour
le 2ème achat, email de suivi personnalisé.

**Segment 1 — Clients dormants (33%)**
Ces clients n'ont pas commandé depuis plus d'un an (404 jours)
mais étaient satisfaits lors de leur dernier achat (4.58/5).
Ils ont un fort potentiel de réactivation.
**Stratégie :** campagne "Vous nous manquez", offre de retour
avec réduction, mise en avant des nouveautés depuis leur départ.

**Segment 2 — Clients insatisfaits (15%)**
Ces clients ont un score de satisfaction de seulement 1.60/5,
malgré des livraisons globalement correctes (-3.7 jours vs -12 jours
pour les autres segments). Le problème vient du produit lui-même,
pas de la logistique.
**Stratégie :** enquête de satisfaction urgente, geste commercial
ciblé (remboursement partiel, bon d'achat), amélioration
de la qualité des produits concernés.

**Segment 3 — VIP multi-acheteurs (3%)**
Seuls 3% des clients mais les plus précieux : 2.11 commandes
en moyenne, 309 BRL de montant total (2x la moyenne),
satisfaction correcte (4.21/5).
**Stratégie :** programme VIP exclusif, accès anticipé aux
nouveautés, service client prioritaire, invitations événements.

---

## Tests unitaires

### Résultats

```
49 tests — 6.74s — 0 failures
Couverture logique métier : 100%
```

### Répartition des tests

| Fichier de test | Tests | Ce qui est vérifié |
|---|---|---|
| `test_sanity.py` | 3 | Environnement, imports, src/ |
| `test_load_data.py` | 13 | Chargement CSV, gestion erreurs |
| `test_build_features.py` | 21 | Chaque feature individuellement |
| `test_predict.py` | 15 | Logique de prédiction, mock MLflow |

### Principes appliqués

**Fixtures Pytest** — données minimales en mémoire (3-5 lignes)
plutôt que les vrais CSV Olist. Garantit des tests rapides,
indépendants du filesystem et exécutables en CI/CD.

**MagicMock** — simule KMeans et StandardScaler sans charger
MLflow. `mock.predict.return_value = np.array([0])` retourne
toujours le cluster 0, permettant de tester la logique sans
dépendance externe.

**`tmp_path`** — fixture Pytest native qui crée un dossier
temporaire unique par test, supprimé automatiquement après.

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ -v --cov=src --cov-report=html
# Rapport HTML dans htmlcov/index.html
```

---

## CI/CD GitHub Actions

Le pipeline `.github/workflows/ci.yml` s'exécute automatiquement
à chaque `git push` sur les branches `main` et `develop`.

### Étapes du pipeline

```
1. actions/checkout@v4          Récupère le code
2. actions/setup-python@v5      Installe Python 3.13
3. pip install -r requirements  Installe les dépendances
4. black --check src/ tests/    Vérifie le formatage
5. flake8 src/ tests/           Vérifie les règles PEP8
6. pytest tests/ --cov=src      Exécute les 49 tests
7. upload-artifact               Archive le rapport de couverture
```

Si une étape échoue → badge rouge + notification.
Si tout passe → badge vert.

---

## API FastAPI

### Endpoints

#### `GET /`
Health check — vérifie que l'API est opérationnelle et que
le modèle est chargé.

```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true,
  "version": "1.0.0"
}
```

#### `GET /segments`
Retourne la liste des 4 segments avec leurs stratégies marketing.

```json
[
  {
    "segment_id": 0,
    "segment_name": "Nouveaux satisfaits",
    "strategy": "Fidélisation — offres exclusives pour 2ème achat"
  },
  ...
]
```

#### `POST /predict`
Prédit le segment d'un client à partir de ses features comportementales.

**Corps de la requête :**
```json
{
  "recency": 135,
  "frequency": 1,
  "monetary": 147.0,
  "review_score_mean": 4.5,
  "delivery_delay_mean": -12.0
}
```

**Réponse :**
```json
{
  "segment_id": 0,
  "segment_name": "Nouveaux satisfaits",
  "strategy": "Fidélisation — offres exclusives pour 2ème achat",
  "input_features": { ... }
}
```

**Documentation interactive :** http://localhost:8000/docs

---

## Dashboard marketing

Interface web `dashboard/index.html` permettant à l'équipe marketing
de tester les endpoints sans connaître l'API ni le code.

**Onglet Segments :** affiche les 4 segments chargés dynamiquement
depuis GET /segments avec leurs couleurs et stratégies.

**Onglet Prédire un client :** formulaire de saisie avec 5 champs,
4 boutons de profils types pré-remplis pour tester rapidement,
résultat affiché avec badge coloré par segment.

**Prérequis :**

```bash
uvicorn src.api.main:app --reload --port 8000
# Puis ouvrir dashboard/index.html dans le navigateur
```

---

## Contrat de maintenance

### Analyse de stabilité

Simulation sur 7 trimestres (2017Q1 → 2018Q3) :

| Indicateur | Valeur | Interprétation |
|---|---|---|
| Silhouette moyen | 0.271 | Qualité du clustering correcte |
| Silhouette std | 0.022 | Très faible variation — stable |
| ARI brut moyen | 0.02 | Non interprétable (label switching) |
| CV taille clusters | 0.76 | Variation naturelle liée au RFM |

**Note méthodologique :**
L'ARI brut (0.02) est affecté par le label switching intrinsèque
à K-Means — les labels changent entre trimestres même si les profils
restent stables. L'analyse par variance de silhouette est plus robuste.

### Recommandation

**Fréquence : mise à jour trimestrielle (tous les 3 mois)**

**Déclencheur d'alerte :** silhouette < 0.20 → mise à jour urgente

**Procédure de mise à jour :**

```bash
# 1. Recalculer les features sur les 12 derniers mois
python src/models/train.py --data-dir data/raw

# 2. Comparer les nouveaux segments (silhouette vs précédent)
mlflow ui --port 5000

# 3. Si silhouette >= 0.20 → déployer le nouveau modèle
# 4. Mettre à jour le dashboard et l'API (redémarrer uvicorn)
# 5. Documenter les changements dans le README
```

---

## Auteurs

- **Thierno Daouda LY**
- **Babacar Niang**
- **Mouhamed Sarr**

Projet réalisé dans le cadre du cours MLOps — 2026.