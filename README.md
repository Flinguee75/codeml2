
# Priorisation énergétique des bâtiments – Montréal

## Contexte
Ce projet a été réalisé dans le cadre du **Défi CodeML (IRIU × VILLE_IA)**.  
L’objectif est d’aider la **Ville de Montréal** à identifier les bâtiments à prioriser pour la rénovation énergétique, en tenant compte à la fois de leur **vulnérabilité climatique** et de leur **dépendance énergétique**.

Le projet combine des données **climatiques, énergétiques et sociales** afin de produire un **classement actionnable** des bâtiments les plus à risque.

---

## Problématique
Les décisions de rénovation énergétique sont complexes car elles impliquent :
- des risques climatiques hétérogènes (pluies diluviennes, inondations, chaleur),
- des profils énergétiques variés,
- des enjeux sociaux et territoriaux.

Sans indicateur synthétique, il est difficile de prioriser efficacement les interventions.

---

## Approche
Le projet repose sur un **pipeline data complet**, de la fusion des données jusqu’à la visualisation finale.

### Données utilisées
- **Climatiques** : vulnérabilité structurelle, thermique, hydrique, végétation
- **Énergétiques** : consommation (électricité, gaz, mazout), émissions de GES
- **Sociales** : indices de défavorisation par quartier
- **Géographiques** : données spatiales des bâtiments

### Indices construits
**Indice de résilience**
```

Résilience = 0.5 × adaptation + 0.5 × (1 − vulnérabilité)

```

**Indice de priorité**
```

Priorité = 0.4 × (1 − résilience)
+ 0.4 × vulnérabilité climatique
+ 0.2 × dépendance énergétique

```

Ces indices permettent de produire un **classement global** des bâtiments.

---

## Résultats
- **218 bâtiments analysés**
- Classement en **3 catégories** :
  - Priorité élevée
  - Priorité moyenne
  - Bonne performance
- **Exports exploitables** (CSV, GeoJSON)
- **Carte interactive** permettant d’ajuster dynamiquement les pondérations

Le classement final est disponible dans :
```

notebooks/classement_final_batiments.csv

````

---

## Visualisation interactive
Une carte web permet :
- d’explorer les bâtiments géolocalisés,
- d’ajuster les pondérations des critères,
- de recalculer les indices en temps réel.

### Lancer la carte
```bash
python3 -m http.server 8000
````

Puis ouvrir :

```
http://localhost:8000/index.html
```

---

## Structure du projet

```
.
├── index.html                # Carte interactive
├── notebooks/
│   ├── CODEML.ipynb          # Pipeline principal
│   └── classement_final_batiments.csv
├── explorations/             # Analyses exploratoires
├── outputs/                  # Résultats générés
├── environment.yml
└── requirements.txt
```

---

## Installation rapide

### Option Conda

```bash
conda env create -f environment.yml
conda activate mlcomp
```

### Option pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Puis lancer le notebook principal :

```bash
jupyter notebook notebooks/CODEML.ipynb
```

---

## Stack technique

* **Python** (pandas, geopandas, scikit-learn)
* **Jupyter Notebook**
* **Traitement géospatial** (GeoJSON, GeoPackage)
* **Visualisation web** : Leaflet, HTML/CSS/JS

---

## Statut du projet

* **Statut** : prototype analytique fonctionnel
* **Objectif** : aide à la décision et exploration data
* **Limites** :

  * pondérations fixées manuellement,
  * pas d’automatisation temps réel,
  * données figées à l’instant T.


## Auteurs

Projet réalisé par
**Tidiane Cissé, Sarah Tabti, Yasmine Tamdrari**
Défi CodeML – IRIU × VILLE_IA
