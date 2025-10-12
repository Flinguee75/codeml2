# Projet : Analyse Climat et Résilience

## Description
Ce projet explore les liens entre les données climatiques, énergétiques et sociales pour évaluer la résilience des bâtiments à Montréal.  
Le notebook principal `notebooks/CODEML.ipynb` contient toutes les étapes : nettoyage, fusion des données et visualisation.

## Structure
```
.
├── index.html                          # Carte interactive des bâtiments
├── notebooks/
│   └── CODEML.ipynb                   # Notebook principal d'analyse
├── data/                               # Jeux de données bruts
├── outputs/                            # Résultats et cartes générées
│   └── indice_resilience.geojson      # Données GeoJSON des bâtiments
├── requirements.txt
└── .gitignore
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

## Utilisation

### Ouvrir le notebook
```bash
jupyter notebook notebooks/CODEML.ipynb
```

### Carte Interactive - Bâtiments à Rénover
Pour visualiser la carte interactive avec les bâtiments prioritaires :

1. **Lancer un serveur HTTP local** (requis pour charger le GeoJSON) :
```bash
python3 -m http.server 8000
```

2. **Ouvrir dans votre navigateur** :
```
http://localhost:8000/index.html
```

3. **Utiliser les contrôles** :
   - Ajustez les curseurs pour modifier l'importance de chaque facteur (structurel, thermique, hydrique, végétation, social, renouvelable)
   - Cliquez sur "Recalculer les Indices" pour mettre à jour la carte
   - Cliquez sur les marqueurs pour voir les détails de chaque bâtiment

### Arrêter le serveur
Utilisez `Ctrl+C` dans le terminal.

## Auteur
Tidiane Cissé
