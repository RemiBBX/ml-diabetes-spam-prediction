# 🚀 Projet Python avec uv

## Contexte

Ce projet a été réalisé dans le cadre d'un projet de groupe à [Nom de l'école / cours].  
Il combine deux tâches de classification en machine learning :

1. **Prédiction du risque de diabète** à partir de données médicales.  
2. **Détection de spam** dans des messages texte.  

Mon apport personnel : ajout d'une **analyse d'interprétabilité via SHAP** pour la prédiction du diabète, afin de comprendre quelles variables influencent le plus les prédictions du modèle.

---

## Installation

```bash
pip install uv
````
Ou regarder la [notice d'installation de uv](https://docs.astral.sh/uv/getting-started/installation/)

## Initialisation du projet

```bash
uv venv
```

```bash
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

```bash
make install
```

## Formatter le code

```bash
make
```

## Structure du dépôt

```bash
ml-diabetes-spam-prediction/
│
├── data/                   # Datasets utilisés pour les deux tâches
├── src/                    # Scripts Python
├── notebooks/              # Tous les notebooks (.ipynb)
│   ├── Diabetes_Visualisation_ML.ipynb
│   ├── SHAP_diabete_analysis.ipynb
│   └── SPAM_Visualization.ipynb
├── modelweights/           # Modèles entraînés
├── main.py                 # Script principal
├── Makefile                # Pour installation / formatage / tests
├── pyproject.toml          # Configuration des dépendances
├── uv.lock                 # Lock file pour uv
└── README.md               # Ce fichier
```

## Utilisation

### 1. Exploration des données
- Ouvrir les notebooks dans `notebooks/` pour visualiser et analyser les datasets :
  - `Diabetes_Visualisation_ML.ipynb` → exploration et visualisation des données diabète
  - `SPAM_Visualization.ipynb` → exploration et visualisation des données spam

### 2. Utilisation du modèle entraîné
- Le modèle Random Forest est sauvegardé dans `modelweights/best_random_forest_1.joblib` et peut être utilisé via `main.py`.

## Résultats clés

- Modèles performants pour la prédiction du diabète et la détection de spam.  
- SHAP permet de visualiser quelles variables influencent le plus les prédictions du diabète, améliorant l’interprétabilité.

---

## Limites

- Projet à visée pédagogique / expérimentale — **non destiné à un usage médical**.  
- Dataset et modèle peu documentés — nécessite plus de métadonnées et nettoyage pour usage sérieux.  
- L’interprétabilité via SHAP est uniquement pour la partie diabète.

---

## Perspectives / Améliorations possibles

- Documenter entièrement les datasets (sources, features, nettoyage, etc.).  
- Ajouter un pipeline ML complet avec cross-validation et métriques de performance.  
- Développer une interface pour tester facilement les modèles.  
- Étendre l’analyse SHAP à la détection de spam.  
- Ajouter tests automatisés et validation des données pour fiabiliser le projet.