# Impact de la ZFE de Grenoble sur la pollution au NO₂  
_Projet Python – Data science / contrôle synthétique_

## Objectif

Évaluer l’effet de la **Zone à Faibles Émissions (ZFE) de Grenoble** sur la qualité de l’air, en particulier sur le **NO₂**, à partir de :

- données de pollution issues du réseau Atmo (stations de Grenoble et villes voisines),
- données officielles ZFE (périmètre, dates, règles) de la **Base Nationale ZFE**.

À terme, la méthode principale sera un **Synthetic Control Method (SCM)**, avec comparaison entre Grenoble (traitée) et un groupe de villes/stations non ZFE (donneurs).

---

## Structure du projet

```text
.
├─ data/
│   ├─ aires.geojson                 # périmètres ZFE France (extrait BNZFE)
│   ├─ voies.geojson                 # tronçons routiers ZFE
│   ├─ zfe_ids.csv                   # table de correspondance ZFE / SIREN / EPCI
│   ├─ Export Moy. journalière ...   # export Atmo AURA (NO2, multi-stations, 2016–2024)
│   ├─ no2_all_stations_daily_clean.csv  # NO2 nettoyé (output)
│   ├─ no2_stations_meta.csv             # info stations + appartenance ZFE Grenoble
│   ├─ pollution_grenoble_les_frenes_monthly.csv (optionnel)
│   └─ zfe_meta.csv                  # méta ZFE (dates de début, nb d’aires, etc.)
├─ zfe-scm/
│   ├─ exploration_descriptive.ipynb # graphes avant/après, diagnostics
│   └─ create_csv.ipynb              # nettoyage / construction des CSV propres
├─ .gitignore
└─ README.md
