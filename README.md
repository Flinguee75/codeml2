# Priorisation Énergétique des Bâtiments à Montréal
## *Bâtiments, énergie et résilience face aux pluies diluviennes*

## 📋 Description

Ce projet s’inscrit dans le cadre du Défi CodeML – IRIU × VILLE_IA, qui vise à aider la Ville de Montréal à cibler les bâtiments à prioriser pour la rénovation énergétique, en intégrant à la fois leur vulnérabilité climatique (pluies diluviennes, inondations, îlots de chaleur) et leur dépendance énergétique (consommation et émissions de GES).

### Objectifs
- Fusionner des données **climatiques**, **énergétiques** et **sociales**
- Créer un **indice de résilience** multi-critères pour chaque bâtiment
- Générer un **classement prioritaire** pour guider les décisions municipales
- Visualiser les résultats via une **carte interactive**

### Méthodologie

Le projet combine plusieurs dimensions d'analyse :

1. **Données climatiques** : indices structurel, thermique, hydrique et de végétation
2. **Données énergétiques** : consommation (électricité, gaz, mazout), émissions GES
3. **Données sociales** : indices de défavorisation par quartier
4. **Feature engineering** : création d'indicateurs de résilience et d'importance

**Indice de résilience** :
```
Résilience = 0.5 × score_adaptation + 0.5 × (1 - score_vulnérabilité)
```

**Indice d'importance** (pour la priorisation) :
```
Importance = 0.4 × (1 - résilience) + 0.4 × vulnérabilité_climatique + 0.2 × dépendance_fossile
```

### Résultats

- **218 bâtiments analysés** et classés par priorité
- **3 catégories** : Vulnérable – priorité élevée | Priorité moyenne | Bonne performance
- **Carte interactive** permettant d'ajuster dynamiquement les pondérations des critères
- **Classement final** exporté dans `notebooks/classement_final_batiments.csv`

## 📁 Structure du Projet

```
.
├── index.html                                    # 🗺️ Carte interactive des bâtiments
├── environment.yml                               # Configuration environnement Conda
├── requirements.txt                              # Dépendances Python alternatives
│
├── notebooks/
│   ├── CODEML.ipynb                             # 📊 Notebook principal (pipeline complet)
│   └── classement_final_batiments.csv           # 🏆 Classement priorisé des bâtiments
│
├── explorations/                                 # 🔬 Analyses exploratoires préliminaires
│   ├── climat.ipynb                             # Analyse données climatiques
│   ├── electricite.ipynb                        # Exploration consommation électrique
│   ├── electricite_1.ipynb                      # Approfondissement électricité
│   ├── inondations.ipynb                        # Risques d'inondation
│   ├── pluie.ipynb                              # Données pluviométriques
│   └── social.ipynb                             # Indices socio-économiques
│
├── data/                                         # 📦 Données brutes (non versionnées)
│
└── outputs/                                      # 💾 Résultats générés
    ├── indice_resilience.geojson                # GeoJSON pour carte interactive
    ├── indice_resilience.csv                    # Export CSV des résultats
    ├── dataset_batiments_climat_elec.gpkg       # Fusion bâtiments + climat + énergie
    ├── dataset_climatique_consolide_montreal.gpkg
    ├── dataset_electricite_nettoye.csv
    └── social_mtl_clean.geojson
```

## Installation et Configuration

### 1. Cloner le repo
```bash
git clone <votre-repo-url>
cd ml_competition
```

### 2. Créer un environnement (recommandé)

#### Option A : Avec Conda (recommandé)
Recréer l'environnement complet avec toutes les dépendances :
```bash
conda env create -f environment.yml
conda activate mlcomp
```

#### Option B : Avec pip/venv
```bash
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Vérifier l'installation
```bash
jupyter notebook --version
python -c "import geopandas; print('Environnement prêt!')"
```

### 4. Générer les données (si nécessaire)
Si les fichiers dans `outputs/` ne sont pas présents, exécutez le notebook:
```bash
jupyter notebook notebooks/CODEML.ipynb
```
Puis exécutez toutes les cellules (Cell → Run All).

## 🚀 Utilisation

### 1️⃣ Analyse complète (Notebook principal)

Le notebook `CODEML.ipynb` contient le pipeline complet :

```bash
jupyter notebook notebooks/CODEML.ipynb
```

**Sections du notebook** :
1. **Lecture et exploration** des données brutes
2. **Nettoyage et géocodage** (adresses → coordonnées GPS)
3. **Fusion spatiale** (bâtiments ⊗ climat ⊗ social)
4. **Création des indicateurs** (résilience, importance, clustering)
5. **Visualisations** et export des résultats

> 💡 **Exécuter toutes les cellules** : `Cell → Run All` (génère les fichiers dans `outputs/`)

---

### 2️⃣ Carte Interactive - Priorisation Dynamique

Visualisez et ajustez la priorisation en temps réel :

**Lancer le serveur local** :
```bash
python3 -m http.server 8000
```

**Accéder à la carte** :  
Ouvrez [http://localhost:8000/index.html](http://localhost:8000/index.html) dans votre navigateur

**Fonctionnalités** :
- 🎚️ **Curseurs interactifs** : ajustez les pondérations des 6 facteurs
  - Structurel, thermique, hydrique, végétation
  - Social (défavorisation)
  - Renouvelable (énergie propre)
- 🔄 **Recalcul dynamique** : cliquez sur "Recalculer les Indices"
- 🔍 **Détails par bâtiment** : cliquez sur les marqueurs pour voir les métriques complètes
- 🎨 **Gradient de couleur** : rouge (priorité élevée) → jaune → vert (bonne performance)

**Arrêter le serveur** : `Ctrl+C` dans le terminal

---

### 3️⃣ Explorations thématiques

Les notebooks dans `explorations/` permettent d'approfondir chaque dimension :

```bash
jupyter notebook explorations/
```

- `climat.ipynb` : Indices de risque climatique par zone
- `electricite_1.ipynb` : Analyse détaillée de la consommation énergétique
- `social.ipynb` : Cartographie des inégalités socio-économiques
- `inondations.ipynb` + `pluie.ipynb` : Risques hydriques

---

### 4️⃣ Résultats prêts à l'emploi

**Classement CSV** :
```bash
cat notebooks/classement_final_batiments.csv
```

**Colonnes principales** :
- `id`, `Adresse` : identification du bâtiment
- `indice_resilience` : capacité d'adaptation (0–1)
- `indice_importance` : priorité de rénovation (0–1)
- `categorie_finale` : Vulnérable / Priorité moyenne / Bonne performance
- `rang_final` : classement global (1 = plus prioritaire)

**📊 Top 10 des bâtiments prioritaires** :

| Rang | Adresse | Catégorie | Importance | Résilience | Vulnérabilité Climat |
|------|---------|-----------|------------|------------|---------------------|
| 🥇 1 | 505 Boulevard De Maisonneuve E | Vulnérable – priorité élevée | 1.000 | 0.402 | 0.977 |
| 🥈 2 | 7047 Rue Saint-Dominique | Vulnérable – priorité élevée | 0.960 | 0.367 | 0.915 |
| 🥉 3 | 5485 Chemin De La Côte-Saint-Paul | Vulnérable – priorité élevée | 0.897 | 0.403 | 1.000 |
| 4 | 5115 Rue Des Galets | Vulnérable – priorité élevée | 0.875 | 0.424 | 0.795 |
| 5 | 944 Rue Saint-Paul O | Vulnérable – priorité élevée | 0.853 | 0.446 | 0.994 |
| 6 | 7959 Av. 16E | Vulnérable – priorité élevée | 0.850 | 0.429 | 0.835 |
| 7 | 5 Avenue Laurier O | Vulnérable – priorité élevée | 0.840 | 0.408 | 0.785 |
| 8 | 5485 Ch. Côte-Saint-Paul | Vulnérable – priorité élevée | 0.840 | 0.438 | 1.000 |
| 9 | 7959 16E Avenue | Vulnérable – priorité élevée | 0.838 | 0.439 | 0.835 |
| 10 | 1500 Rue Des Carrières | Vulnérable – priorité élevée | 0.824 | 0.432 | 0.830 |

> 💡 **Fichier complet** : `notebooks/classement_final_batiments.csv` (218 bâtiments)

## 📊 Sources de Données

### Données utilisées
- **Climatiques** : Indices de vulnérabilité structurelle, thermique, hydrique et de végétation par secteur
- **Énergétiques** : Consommation d'électricité, gaz naturel, mazout et émissions GES par bâtiment
- **Sociales** : Indices de défavorisation socio-économique par quartier (CIUSSS)
- **Géographiques** : Géolocalisation via Google Maps Geocoding API

### Formats
- **GeoPackage** (`.gpkg`) : données géospatiales avec géométries
- **GeoJSON** : export pour visualisation web
- **CSV** : tableaux de données et classements

## 🛠️ Technologies

**Langages & Frameworks** :
- Python 3.10
- Jupyter Notebook

**Bibliothèques principales** :
- `geopandas` : traitement de données géospatiales
- `pandas` : manipulation de données tabulaires
- `matplotlib` + `seaborn` : visualisations
- `scikit-learn` : clustering et normalisation
- `shapely` : opérations géométriques
- `geopy` : géocodage d'adresses

**Frontend** :
- Leaflet.js : cartographie interactive
- HTML/CSS/JavaScript vanilla

## 🎯 Livrables

✅ **Pipeline complet** : notebook reproductible du nettoyage à la priorisation  
✅ **Classement actionnable** : 218 bâtiments catégorisés par ordre de priorité  
✅ **Carte interactive** : interface web pour explorer et ajuster les pondérations  
✅ **Exports multi-formats** : CSV, GeoJSON, GeoPackage  
✅ **Documentation** : README détaillé et code commenté

## 🔮 Améliorations Possibles

- **Optimisation bayésienne** : ajuster automatiquement les pondérations (Optuna intégré)
- **Prédiction temporelle** : anticiper l'évolution de la résilience
- **Intégration API** : pipeline automatisé avec mise à jour des données en temps réel
- **Scoring de coûts** : intégrer des estimations de coûts de rénovation
- **Dashboard BI** : tableau de bord Power BI ou Tableau pour la Ville

## 👤 Auteur
**Tidiane Cissé, Sarah Tabti, Yasmine Tamdrari**  
Projet réalisé dans le cadre du défi Bâtiments, énergie et résilience face aux pluies diluviennes – Défi CodeML – IRIU × VILLE_IA
