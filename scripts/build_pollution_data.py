"""
Construction des jeux de données NO2 pour Grenoble et Paris.

Ce module reprend la logique utilisée dans le notebook de préparation :

- lecture des exports bruts (ATMO / Airparif) ;
- filtrage sur le polluant NO2 ;
- construction d'un jeu de données journalier propre ;
- contrôle de l'appartenance des stations au périmètre de la ZFE
  à partir du fichier aires.geojson ;
- sauvegarde des fichiers nettoyés dans le dossier data/.
"""

from pathlib import Path
from typing import Tuple
import json

import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import unary_union


def build_grenoble_no2_daily(
    data_dir: Path,
    raw_csv_name: str,
    aires_geojson_name: str = "aires.geojson",
    out_daily_name: str = "pollution_grenoble_no2_daily_clean.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit le jeu de données NO2 journalier pour Grenoble à partir d'un export brut ATMO.

    Étapes :
    1. Lecture du CSV brut et filtrage sur le polluant NO2.
    2. Construction d'un jeu de données journalier propre :
       - colonne de date homogène ;
       - identifiant et nom de station ;
       - type d'implantation et d'influence ;
       - coordonnées géographiques ;
       - concentration de NO2 en µg/m³.
    3. Chargement de la ZFE Grenoble (aires.geojson) et contrôle de la position
       des stations par rapport au périmètre.
    4. Sauvegarde du jeu de données journalier dans out_daily_name.

    Le fichier de sortie contient toutes les stations présentes dans l’export.
    L’information de localisation par rapport à la ZFE est utilisée comme contrôle
    mais ne sert pas à filtrer le fichier à ce stade.

    Paramètres
    ----------
    data_dir : Path
        Dossier contenant les fichiers de données (data/).
    raw_csv_name : str
        Nom du fichier CSV brut (export ATMO) pour la zone de Grenoble.
    aires_geojson_name : str
        Nom du fichier GeoJSON décrivant les aires de ZFE.
    out_daily_name : str
        Nom du fichier CSV de sortie pour les données NO2 journalières.

    Retour
    ------
    no2 : DataFrame
        Données journalières de NO2 (toutes stations de l’export).
    stations_meta : DataFrame
        Tableau des stations avec leurs coordonnées et un indicateur in_zfe_grenoble.
    """
    data_dir = Path(data_dir)
    poll_path = data_dir / raw_csv_name
    aires_path = data_dir / aires_geojson_name

    if not poll_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {poll_path}")
    if not aires_path.exists():
        raise FileNotFoundError(f"Fichier aires.geojson introuvable : {aires_path}")

    # 1) Charger le CSV brut et filtrer NO2
    df_raw = pd.read_csv(poll_path, sep=";", engine="python")

    date_debut_col_candidates = [c for c in df_raw.columns if "Date de début" in c]
    if not date_debut_col_candidates:
        raise ValueError("Aucune colonne contenant 'Date de début' dans le fichier brut.")
    date_debut_col = date_debut_col_candidates[0]

    df_raw["date"] = pd.to_datetime(df_raw[date_debut_col])
    df_no2 = df_raw[df_raw["Polluant"] == "NO2"].copy()

    # 2) Dataset NO2 propre
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

    no2["zone"] = "GRENOBLE"

    # 3) Contrôle d'appartenance à la ZFE Grenoble
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

    # On suit la logique du notebook : un seul polygone principal
    zfe_geom = shape(grenoble_feats[0]["geometry"])

    coords = (
        no2.groupby(["station_id", "station_name"])[["lat", "lon"]]
        .first()
        .reset_index()
    )

    def in_zfe(row) -> bool:
        pt = Point(row["lon"], row["lat"])
        return zfe_geom.contains(pt)

    coords["in_zfe_grenoble"] = coords.apply(in_zfe, axis=1)

    # 4) Sauvegarde
    out_path = data_dir / out_daily_name
    no2.to_csv(out_path, index=False)

    return no2, coords


def build_paris_no2_daily(
    data_dir: Path,
    raw_csv_name: str,
    aires_geojson_name: str = "aires.geojson",
    out_daily_name: str = "pollution_paris_no2_daily_clean.csv",
    out_meta_name: str = "no2_paris_stations_meta.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit le jeu de données NO2 journalier pour Paris à partir d'un export brut Airparif.

    Étapes :
    1. Lecture du CSV brut et filtrage sur NO2.
    2. Construction d'un jeu de données journalier propre :
       - colonne de date homogène ;
       - identifiant et nom de station ;
       - type d'implantation et d'influence ;
       - coordonnées géographiques ;
       - concentration de NO2.
    3. Chargement de la ZFE Paris à partir de aires.geojson et union des polygones.
    4. Construction d'une table meta avec un indicateur d'appartenance à la ZFE (in_zfe_paris).
    5. Sauvegarde des fichiers nettoyés.

    Paramètres
    ----------
    data_dir : Path
        Dossier contenant les fichiers de données (data/).
    raw_csv_name : str
        Nom du fichier CSV brut (export Airparif) pour Paris.
    aires_geojson_name : str
        Nom du fichier GeoJSON décrivant les aires de ZFE.
    out_daily_name : str
        Nom du fichier CSV de sortie pour les données NO2 journalières.
    out_meta_name : str
        Nom du fichier CSV de sortie pour les métadonnées de stations.

    Retour
    ------
    no2_paris_daily : DataFrame
        Données journalières de NO2 pour les stations parisiennes.
    stations_meta : DataFrame
        Métadonnées des stations avec indicateur in_zfe_paris.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / raw_csv_name
    aires_path = data_dir / aires_geojson_name

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {csv_path}")
    if not aires_path.exists():
        raise FileNotFoundError(f"Fichier aires.geojson introuvable : {aires_path}")

    # 1) Charger et nettoyer le CSV NO2 Paris
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

    # 2) Charger la ZFE Paris
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

    # 3) Construire la table meta stations + in_zfe_paris
    stations_meta = (
        no2_paris_daily.groupby(
            ["station_id", "station_name", "station_env", "station_influence"]
        )[["lat", "lon"]]
        .first()
        .reset_index()
    )

    stations_meta["in_zfe_paris"] = stations_meta.apply(
        lambda row: paris_geom.contains(Point(row["lon"], row["lat"])),
        axis=1,
    )

    # 4) Sauvegarde
    no2_paris_daily.to_csv(data_dir / out_daily_name, index=False)
    stations_meta.to_csv(data_dir / out_meta_name, index=False)

    return no2_paris_daily, stations_meta
