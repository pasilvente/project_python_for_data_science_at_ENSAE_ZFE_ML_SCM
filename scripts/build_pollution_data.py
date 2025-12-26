"""
Construction des jeux de données NO2 pour Grenoble et Paris.

Ce module reprend la logique des notebooks de préparation :

- lecture des exports bruts (ATMO / Airparif) ;
- filtrage sur le polluant NO2 ;
- mise en forme d'un jeu de données journalier propre ;
- construction d'une table de métadonnées de stations (coordonnées, type, etc.) ;
- identification des stations situées dans le périmètre de la ZFE
  à partir du fichier aires.geojson ;
- sauvegarde des fichiers nettoyés dans le dossier data/.
"""

from pathlib import Path
from typing import Tuple
import json

import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import unary_union


def build_grenoble_no2_data(
    data_dir: Path,
    raw_csv_name: str,
    aires_geojson_name: str = "aires.geojson",
    out_all_daily_name: str = "no2_all_stations_daily_clean.csv",
    out_meta_name: str = "no2_stations_meta.csv",
    out_zfe_daily_name: str = "pollution_grenoble_no2_daily_clean.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit les jeux de données NO2 pour Grenoble à partir d'un export brut ATMO.

    Étapes :
    1. Lecture du CSV brut et filtrage sur le polluant NO2.
    2. Construction d'un jeu de données journalier propre :
       - colonne de date homogène ;
       - colonnes station_id, station_name, station_env, station_influence ;
       - colonnes lat, lon ;
       - variable no2_ug_m3.
    3. Construction d'une table de métadonnées de stations (stations_meta) :
       - une ligne par station ;
       - coordonnées moyennes ;
       - type d'implantation et d'influence.
    4. Chargement de la ZFE Grenoble à partir de aires.geojson et construction
       de la géométrie agrégée.
    5. Détermination, pour chaque station, de l'appartenance ou non à la ZFE
       (colonne in_zfe_grenoble dans stations_meta).
    6. Sauvegarde :
       - no2_all_stations_daily_clean.csv : toutes les stations de l'export ;
       - no2_stations_meta.csv : métadonnées des stations avec in_zfe_grenoble ;
       - pollution_grenoble_no2_daily_clean.csv : sous-ensemble des observations
         correspondant aux stations situées dans la ZFE.

    Paramètres
    ----------
    data_dir : Path
        Dossier contenant les fichiers de données (data/).
    raw_csv_name : str
        Nom du fichier CSV brut (export ATMO) pour Grenoble.
    aires_geojson_name : str
        Nom du fichier GeoJSON des aires de ZFE.
    out_all_daily_name : str
        Nom du fichier CSV de sortie pour les données journalières toutes stations.
    out_meta_name : str
        Nom du fichier CSV de sortie pour les métadonnées de stations.
    out_zfe_daily_name : str
        Nom du fichier CSV de sortie pour les stations situées dans la ZFE Grenoble.

    Retour
    ------
    no2 : DataFrame
        Données journalières de NO2 pour l'ensemble des stations de l'export.
    stations_meta : DataFrame
        Métadonnées des stations, avec indicateur in_zfe_grenoble.
    """
    data_dir = Path(data_dir)

    poll_path = data_dir / raw_csv_name
    aires_path = data_dir / aires_geojson_name

    if not poll_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {poll_path}")
    if not aires_path.exists():
        raise FileNotFoundError(f"Fichier aires.geojson introuvable : {aires_path}")

    # Chargement du CSV brut et filtrage sur NO2
    df_raw = pd.read_csv(poll_path, sep=";", engine="python")

    # Identification de la colonne de date (contient 'Date de début')
    date_debut_col_candidates = [c for c in df_raw.columns if "Date de début" in c]
    if not date_debut_col_candidates:
        raise ValueError("Aucune colonne contenant 'Date de début' dans le fichier brut.")
    date_debut_col = date_debut_col_candidates[0]

    df_raw["date"] = pd.to_datetime(df_raw[date_debut_col])

    df_no2 = df_raw[df_raw["Polluant"] == "NO2"].copy()

    # Jeu de données journalier propre
    no2 = (
        df_no2[
            [
                "date",
                "code site",
                "nom site",
                "type d'implantation",
                "type d'influence",
                "valeur",
                "Latitude",
                "Longitude",
            ]
        ]
        .rename(
            columns={
                "code site": "station_id",
                "nom site": "station_name",
                "type d'implantation": "station_env",
                "type d'influence": "station_influence",
                "valeur": "no2_ug_m3",
                "Latitude": "lat",
                "Longitude": "lon",
            }
        )
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )

    # Table de métadonnées des stations
    stations_meta = (
        no2.groupby(
            ["station_id", "station_name", "station_env", "station_influence"]
        )[["lat", "lon"]]
        .first()
        .reset_index()
    )

    # Chargement de la ZFE Grenoble
    with aires_path.open(encoding="utf-8") as f:
        gj = json.load(f)

    grenoble_feats = [
        feat
        for feat in gj.get("features", [])
        if feat.get("publisher", {}).get("zfe_id") == "GRENOBLE"
    ]
    if not grenoble_feats:
        raise ValueError(
            "Aucune ZFE avec publisher.zfe_id == 'GRENOBLE' trouvée dans aires.geojson."
        )

    geoms = [shape(feat["geometry"]) for feat in grenoble_feats]
    zfe_grenoble = unary_union(geoms)

    # Indicateur d'appartenance à la ZFE pour chaque station
    def is_in_zfe_grenoble(row) -> bool:
        pt = Point(row["lon"], row["lat"])
        return zfe_grenoble.contains(pt)

    stations_meta["in_zfe_grenoble"] = stations_meta.apply(is_in_zfe_grenoble, axis=1)

    # Sous-ensemble des stations situées dans la ZFE
    grenoble_zfe_ids = stations_meta.loc[
        stations_meta["in_zfe_grenoble"], "station_id"
    ].unique()

    no2_zfe = no2[no2["station_id"].isin(grenoble_zfe_ids)].copy()
    no2_zfe["zone"] = "GRENOBLE"

    # Sauvegarde des jeux de données
    all_daily_path = data_dir / out_all_daily_name
    meta_path = data_dir / out_meta_name
    zfe_daily_path = data_dir / out_zfe_daily_name

    no2.to_csv(all_daily_path, index=False)
    stations_meta.to_csv(meta_path, index=False)
    no2_zfe.to_csv(zfe_daily_path, index=False)

    return no2, stations_meta


def build_paris_no2_data(
    data_dir: Path,
    raw_csv_name: str,
    aires_geojson_name: str = "aires.geojson",
    out_daily_name: str = "pollution_paris_no2_daily_clean.csv",
    out_meta_name: str = "no2_paris_stations_meta.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit les jeux de données NO2 pour Paris à partir d'un export brut Airparif.

    Étapes :
    1. Lecture du CSV brut et filtrage sur NO2.
    2. Construction d'un jeu de données journalier propre :
       - colonne de date homogène ;
       - colonnes station_id, station_name, station_env, station_influence ;
       - lat / lon en float ;
       - variable no2_ug_m3.
    3. Construction de la table de métadonnées des stations parisiennes.
    4. Chargement de la ZFE Paris à partir de aires.geojson et construction
       d'une géométrie agrégée.
    5. Détermination, pour chaque station, de l'appartenance à la ZFE (in_zfe_paris).
    6. Sauvegarde :
       - pollution_paris_no2_daily_clean.csv : données journalieres ;
       - no2_paris_stations_meta.csv : métadonnées des stations avec in_zfe_paris.

    Paramètres
    ----------
    data_dir : Path
        Dossier contenant les fichiers de données (data/).
    raw_csv_name : str
        Nom du fichier CSV brut (export Airparif) pour Paris.
    aires_geojson_name : str
        Nom du fichier GeoJSON des aires de ZFE.
    out_daily_name : str
        Nom du fichier CSV de sortie pour les données journalières propres.
    out_meta_name : str
        Nom du fichier CSV de sortie pour les métadonnées de stations.

    Retour
    ------
    no2_paris_daily : DataFrame
        Données journalières de NO2 pour les stations parisiennes.
    stations_meta : DataFrame
        Métadonnées des stations, avec indicateur in_zfe_paris.
    """
    data_dir = Path(data_dir)

    csv_path = data_dir / raw_csv_name
    aires_path = data_dir / aires_geojson_name

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {csv_path}")
    if not aires_path.exists():
        raise FileNotFoundError(f"Fichier aires.geojson introuvable : {aires_path}")

    # Chargement et nettoyage du CSV NO2 Paris
    df_raw = pd.read_csv(csv_path, sep=";", engine="python")
    df_no2 = df_raw[df_raw["Polluant"] == "NO2"].copy()

    date_col_candidates = [c for c in df_no2.columns if "Date de début" in c]
    if not date_col_candidates:
        raise ValueError("Aucune colonne contenant 'Date de début' dans le fichier brut.")
    date_col = date_col_candidates[0]
    df_no2["date"] = pd.to_datetime(df_no2[date_col])

    df_no2["lat"] = df_no2["Latitude"].astype(float)
    df_no2["lon"] = df_no2["Longitude"].astype(float)

    no2_paris_daily = (
        df_no2.rename(
            columns={
                "code site": "station_id",
                "nom site": "station_name",
                "type d'implantation": "station_env",
                "type d'influence": "station_influence",
                "valeur": "no2_ug_m3",
            }
        )[
            [
                "date",
                "station_id",
                "station_name",
                "station_env",
                "station_influence",
                "no2_ug_m3",
                "lat",
                "lon",
            ]
        ]
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )

    # Table méta des stations parisiennes
    stations_meta = (
        no2_paris_daily.groupby(
            ["station_id", "station_name", "station_env", "station_influence"]
        )[["lat", "lon"]]
        .first()
        .reset_index()
    )

    # Chargement de la ZFE Paris
    with aires_path.open(encoding="utf-8") as f:
        gj = json.load(f)

    paris_feats = [
        ft
        for ft in gj.get("features", [])
        if ft.get("publisher", {}).get("zfe_id") == "PARIS"
    ]
    if not paris_feats:
        raise ValueError(
            "Impossible de trouver une ZFE avec publisher.zfe_id == 'PARIS' dans aires.geojson."
        )

    paris_geom = unary_union([shape(ft["geometry"]) for ft in paris_feats])

    # Indicateur in_zfe_paris
    stations_meta["in_zfe_paris"] = stations_meta.apply(
        lambda row: paris_geom.contains(Point(row["lon"], row["lat"])),
        axis=1,
    )

    # Sauvegarde des CSV propres
    no2_out_path = data_dir / out_daily_name
    stations_out_path = data_dir / out_meta_name

    no2_paris_daily.to_csv(no2_out_path, index=False)
    stations_meta.to_csv(stations_out_path, index=False)

    return no2_paris_daily, stations_meta
