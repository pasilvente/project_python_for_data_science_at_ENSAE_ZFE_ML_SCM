# Projet ZFE – NO₂ à Grenoble et Paris

Ce dépôt contient le code et les données utilisées pour évaluer l’impact des Zones à Faibles Émissions (ZFE) sur les concentrations de NO₂ à **Grenoble Boulevards** et **Paris Champs-Élysées**.  
L’analyse combine :

- préparation de données de qualité de l’air,
- analyse descriptive,
- **contrôle synthétique** (SCM) pénalisé,
- modèles de **machine learning** (Random Forest et LightGBM).

L’objectif du dépôt est que l’on puisse reproduire l’intégralité des résultats du rapport à partir d’une instance vierge de VSCode-Python (par exemple sur le SSP Cloud).

## 1. Organisation du dépôt

```text
.
├── data/
│   ├── aires_clean.csv             # aires ZFE nettoyées
│   ├── aires_flat.csv
│   ├── aires.geojson
│   ├── voies_clean.csv             # tronçons routiers ZFE
│   ├── voies_flat.csv
│   ├── voies.geojson
│   ├── zfe_ids.csv, zfe_meta.csv   # métadonnées ZFE
│   ├── pollution_grenoble_no2_daily_clean.csv
│   ├── pollution_grenoble_no2_daily_imputed.csv
│   ├── pollution_paris_no2_daily_clean.csv
│   ├── pollution_paris_no2_daily_imputed.csv
│   ├── no2_donors_grenoble_daily_clean.csv
│   ├── no2_donors_grenoble_daily_imputed.csv
│   ├── no2_donors_paris_daily_clean.csv
│   ├── no2_donors_paris_daily_imputed.csv
│   ├── Export Max. journalier moy. hor. - 20251226130804 - 2017-08-17 00_00 - 2025-04-12 │00_00.csv
│   ├── Export Moy. journalière - 20251204215149 - 2016-02-05 00_00 - 2024-02-05 21_00.csv
│   ├── Export Moy. journalière - 20251205011655 - 2016-02-05 00_00 - 2024-02-05 00_00.csv
│   └── Export Moy. journalière - 20251205114649 - 2017-08-17 00_00 - 2025-12-04 11_00.csv
│
├── scripts/
│   ├── __init__.py
│   ├── zfe_data.py                 # construction tables ZFE / aires / voies
│   ├── build_pollution_data.py     # jointure pollution + ZFE, flags, etc.
│   ├── data_prep.py                # gestion des données manquantes, imputations
│   ├── scm_models.py               # modèles de contrôle synthétique
│   ├── ml_models.py                # modèles Random Forest / LightGBM
│   └── boite_a_outils_stats_desc.py# fonctions d’analyse descriptive & cartes
├── zfe-scm/                        # notebooks d’exploration (non nécessaires pour reproduire le rapport)
│   ├── grenoble_explo.ipynb
│   ├── paris_explo.ipynb
│   ├── donneurs_explo_grenoble.ipynb
│   ├── donneurs_explo_paris.ipynb
│   ├── stat_desc.ipynb
│   └── create_csv.ipynb
├── rapport_zfe.ipynb               # notebook "rapport" avec cellules vides
├── rapport_zfe_clean_outputs.ipynb # même notebook avec toutes les sorties déjà calculées
├── requirements.txt
└── README.md
``` 

Pour la reproductibilité, le notebook à exécuter est rapport_zfe_clean_outputs.ipynb (version sans sorties), à lancer de haut en bas. Le détail de cette étape est expliqué dans les sections suivantes 2, 3 et 4 qui sont à suivre pour reproduire les résultats de ce projet.

Le fichier rapport_zfe.ipynb est la même chose avec toutes les figures et tables déjà générées (utile pour la relecture mais pas nécessaire à la reproduction). Les notebooks du dossier zfe-scm/ sont des notebooks d’exploration utilisés pendant le développement et ne sont pas requis pour reproduire les résultats du rapport.

## 2. Prérequis

```markdown

- **Python** ≥ 3.11  
- **Git**
- Accès Internet (nécessaire uniquement pour télécharger les tuiles de fond de carte OpenStreetMap utilisées par `contextily`).

Toutes les bibliothèques Python nécessaires sont listées dans `requirements.txt`
(`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `geopandas`, `contextily`, etc.).
```

## 3. Installation de l’environnement

Dans un terminal (VSCode, SSP Cloud ou autre) :

1. **Cloner le dépôt**

   ```bash
   git clone <URL_DU_DEPOT_GITHUB>
   cd <nom_du_dossier_cloné>
   ```

2. **(Recommandé) Créer un environnement virtuel**

   ```bash
   python -m venv .venv
    # Sous Linux / macOS / SSP Cloud
    source .venv/bin/activate
    # Sous Windows
    .venv\Scripts\activate
    ```

3. **Installer les dépendances**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```



### 4. Reproduire les résultats du rapport

```markdown

1. Ouvrir VSCode (ou VSCode sur le SSP Cloud) dans le dossier du projet.
2. Vérifier que l’interpréteur Python utilisé est bien celui de l’environnement choisi (`.venv` ou Python système).
3. Ouvrir le notebook :

   - `rapport_zfe_clean_outputs.ipynb`

4. Vérifier que le **dossier de travail** est la racine du projet (les chemins relatifs `data/...` et `scripts/...` doivent fonctionner).

5. Exécuter toutes les cellules **dans l’ordre** (menu `Run > Run All` ou équivalent).

Le notebook rapport_zfe_clean_outputs.ipynb doit normalement :

- importer les fonctions du dossier `scripts/`,
- charger les fichiers de `data/`,
- reconstruire les tables nécessaires (aires ZFE, voies, pollution avec indicateurs),
- réaliser l’analyse descriptive,
- ajuster les SCM (mensuel et journalier),
- ajuster les modèles ML,
- calculer les ATT et produit l’ensemble des figures et tableaux utilisés dans le rapport.

Aucune variable d’environnement particulière n’est nécessaire, tous les chemins sont relatifs au dossier du projet.
```

## 5. Notes sur les données et les fichiers

Le dossier `data/` contient :

- des **tables déjà nettoyées / imputées** (fichiers `*_imputed.csv`) utilisées dans le notebook principal pour garantir la reproductibilité même si les sources en ligne évoluent ;
- des fichiers plus bruts (`*_clean.csv`, `aires_flat.csv`, `voies_flat.csv`, etc.) qui permettent, via les fonctions du dossier `scripts/`, de reconstruire les mêmes tables à partir des exports d’origine.

Le notebook `rapport_zfe.ipynb` documente explicitement quelles tables sont reconstruites à partir des bruts et lesquelles sont directement utilisées.

Les notebooks du dossier `zfe-scm/` servent uniquement à :
  - explorer les séries,
  - tester des variantes de pré-traitement,
  - vérifier les donneurs, etc.

Ils ne sont pas nécessaires pour obtenir les chiffres et graphes du rapport, mais peuvent être utiles si l’on souhaite prolonger l’analyse. Attention, le téléchargement des tuiles OpenStreetMap nécessite une connexion Internet. Cela n’empêche pas l’exécution de la suite de l’analyse, mais certaines figures géographiques peuvent être incomplètes.


## 6. Licence et contexte

Ce projet est réalisé dans le cadre d’un cours de Python pour la Data Science.  
La réutilisation du code est libre pour un usage pédagogique ou exploratoire, sous réserve de citer la source et de respecter les conditions d’utilisation des données d’origine (Air qualité, OpenStreetMap, etc.).


