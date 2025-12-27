# scripts/build_pollution_data.py

from pathlib import Path
from typing import Tuple
import json

import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import unary_union


def clean_no2_export(data_dir: Path, raw_csv_name: str) -> pd.DataFrame:
    """
    Nettoie un export journalier déjà restreint au polluant NO2.

    Retourne un DataFrame au format standard :

    date, station_id, station_name, station_env, station_influence,
    no2_ug_m3, lat, lon
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / raw_csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {csv_path}")

    df_raw = pd.read_csv(csv_path, sep=";", engine="python")

    # Vérification éventuelle de la colonne 'Polluant'
    if "Polluant" in df_raw.columns:
        unique_poll = df_raw["Polluant"].dropna().unique()
        if len(unique_poll) > 1 or (len(unique_poll) == 1 and unique_poll[0] != "NO2"):
            raise ValueError(
                f"La colonne 'Polluant' contient des valeurs inattendues : {unique_poll}. "
                "Cette fonction suppose un export déjà restreint à NO2."
            )

    date_cols = [c for c in df_raw.columns if "Date de début" in c]
    if not date_cols:
        raise ValueError("Aucune colonne contenant 'Date de début' dans le fichier brut.")
    date_col = date_cols[0]
    df_raw["date"] = pd.to_datetime(df_raw[date_col])

    df_raw["lat"] = df_raw["Latitude"].astype(float)
    df_raw["lon"] = df_raw["Longitude"].astype(float)

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

    return no2_daily


def build_no2_with_zfe_flag(
    data_dir: Path,
    raw_csv_name: str,
    zfe_id: str,
    out_daily_name: str,
    aires_geojson_name: str = "aires.geojson",
    in_zfe_col: str = "in_zfe",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit un jeu de données NO2 journalier et un tableau de stations
    avec un indicateur d'appartenance à une ZFE donnée.

    Paramètres
    ----------
    data_dir : Path
        Dossier contenant les fichiers de données (data/).
    raw_csv_name : str
        Nom du fichier CSV brut (export ATMO / Airparif), déjà restreint à NO2.
    zfe_id : str
        Identifiant de la ZFE dans aires.geojson (ex. "GRENOBLE" ou "PARIS").
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
        Données journalières propres (une ligne par station et par jour),
        avec une colonne 'zone' renseignée à zfe_id.
    stations_meta : DataFrame
        Table avec une ligne par station :
        station_id, station_name, station_env, station_influence, lat, lon,
        indicateur d'appartenance à la ZFE.
    """
    data_dir = Path(data_dir)
    aires_path = data_dir / aires_geojson_name

    if not aires_path.exists():
        raise FileNotFoundError(f"Fichier aires.geojson introuvable : {aires_path}")

    # Nettoyage générique de l'export NO2
    no2_daily = clean_no2_export(data_dir=data_dir, raw_csv_name=raw_csv_name)
    no2_daily["zone"] = zfe_id

    # Géométrie de la ZFE
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

    # Méta stations : une ligne par station_id
    stations_meta = (
        no2_daily
        .groupby("station_id")
        .agg(
            station_name=("station_name", lambda s: s.value_counts().idxmax()),
            station_env=("station_env", lambda s: s.value_counts().idxmax()),
            station_influence=("station_influence", lambda s: s.value_counts().idxmax()),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .reset_index()
    )

    def is_in_zfe(row) -> bool:
        pt = Point(row["lon"], row["lat"])
        return zfe_geom.contains(pt)

    stations_meta[in_zfe_col] = stations_meta.apply(is_in_zfe, axis=1)


    def is_in_zfe(row) -> bool:
        pt = Point(row["lon"], row["lat"])
        return zfe_geom.contains(pt)

    stations_meta[in_zfe_col] = stations_meta.apply(is_in_zfe, axis=1)

    # Sauvegarde du daily clean
    out_path = data_dir / out_daily_name
    no2_daily.to_csv(out_path, index=False)

    return no2_daily, stations_meta
