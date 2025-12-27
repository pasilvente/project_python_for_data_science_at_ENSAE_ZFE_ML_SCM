"""
Construction d'un jeu de données NO2 journalier pour une ZFE donnée
à partir d'un export brut (ATMO, Airparif, etc.) et du fichier aires.geojson.

La fonction principale build_no2_daily_for_zfe met en forme un export déjà
restreint au polluant NO2, ajoute une colonne date et les coordonnées,
et calcule un indicateur d'appartenance des stations au périmètre de la ZFE.
"""

from pathlib import Path
from typing import Tuple
import json

import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import unary_union


def build_no2_daily_for_zfe(
    data_dir: Path,
    raw_csv_name: str,
    zfe_id: str,
    out_daily_name: str,
    aires_geojson_name: str = "aires.geojson",
    in_zfe_col: str = "in_zfe",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit un jeu de données NO2 journalier pour une ZFE donnée.

    Paramètres
    ----------
    data_dir : Path
        Dossier contenant les fichiers de données (data/).
    raw_csv_name : str
        Nom du fichier CSV brut (export ATMO / Airparif), déjà restreint à NO2.
    zfe_id : str
        Identifiant de la ZFE dans aires.geojson (par exemple "GRENOBLE" ou "PARIS").
    out_daily_name : str
        Nom du fichier CSV de sortie pour les données NO2 journalières nettoyées.
    aires_geojson_name : str
        Nom du fichier GeoJSON décrivant les aires de ZFE.
    in_zfe_col : str
        Nom de la colonne booléenne indiquant l'appartenance à la ZFE
        dans la table des stations retournée.

    Retour
    ------
    no2_daily : DataFrame
        Données journalières propres (une ligne par station et par jour).
    stations_meta : DataFrame
        Table avec une ligne par station :
        station_id, station_name, lat, lon, indicateur d'appartenance à la ZFE.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / raw_csv_name
    aires_path = data_dir / aires_geojson_name

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {csv_path}")
    if not aires_path.exists():
        raise FileNotFoundError(f"Fichier aires.geojson introuvable : {aires_path}")

    # Chargement de l'export brut
    df_raw = pd.read_csv(csv_path, sep=";", engine="python")

    # Si une colonne 'Polluant' existe, on vérifie qu'elle ne contient que NO2
    if "Polluant" in df_raw.columns:
        polluants_uniques = df_raw["Polluant"].dropna().unique()
        if len(polluants_uniques) > 1 or (
            len(polluants_uniques) == 1 and polluants_uniques[0] != "NO2"
        ):
            raise ValueError(
                f"La colonne 'Polluant' contient plusieurs valeurs : {polluants_uniques}. "
                "La fonction est prévue pour un export déjà restreint à NO2."
            )

    # Colonne de date : on repère la première colonne contenant "Date de début"
    date_cols = [c for c in df_raw.columns if "Date de début" in c]
    if not date_cols:
        raise ValueError("Aucune colonne contenant 'Date de début' dans le fichier brut.")
    date_col = date_cols[0]
    df_raw["date"] = pd.to_datetime(df_raw[date_col])

    # Conversion des coordonnées
    df_raw["lat"] = df_raw["Latitude"].astype(float)
    df_raw["lon"] = df_raw["Longitude"].astype(float)

    # Jeu de données journalier propre
    no2_daily = (
        df_raw.rename(
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

    # Étiquette de zone (utile pour la suite de l'analyse)
    no2_daily["zone"] = zfe_id

    # Construction de la géométrie de la ZFE à partir d'aires.geojson
    with aires_path.open(encoding="utf-8") as f:
        gj = json.load(f)

    feats = [
        ft
        for ft in gj.get("features", [])
        if ft.get("publisher", {}).get("zfe_id") == zfe_id
    ]
    if not feats:
        raise ValueError(
            f"Aucune ZFE avec publisher.zfe_id == '{zfe_id}' trouvée dans aires.geojson."
        )

    zfe_geom = unary_union([shape(ft["geometry"]) for ft in feats])

    # Table méta des stations et indicateur d'appartenance à la ZFE
    stations_meta = (
        no2_daily.groupby(["station_id", "station_name"])[["lat", "lon"]]
        .first()
        .reset_index()
    )

    def is_in_zfe(row) -> bool:
        pt = Point(row["lon"], row["lat"])
        return zfe_geom.contains(pt)

    stations_meta[in_zfe_col] = stations_meta.apply(is_in_zfe, axis=1)

    # Sauvegarde du daily clean
    out_path = data_dir / out_daily_name
    no2_daily.to_csv(out_path, index=False)

    return no2_daily, stations_meta
