"""
Préparation des données ZFE à partir des fichiers GeoJSON et de la table zfe_ids.

Ce module reprend la logique du notebook de préparation initial :

1. Lecture des fichiers bruts :
   - aires.geojson
   - voies.geojson
   - zfe_ids.csv

2. Aplatissement des GeoJSON en tables tabulaires :
   - aires_flat.csv
   - voies_flat.csv

3. Construction de tables nettoyées :
   - aires_clean.csv
   - voies_clean.csv

4. Construction de la table de métadonnées zfe_meta.csv :
   - une ligne par ZFE
   - dates de début et de fin
   - nombre d'aires
   - indicateurs de restrictions VP
   - enrichissement avec zfe_ids (siren, forme juridique, etc.)
"""

from pathlib import Path
from typing import Tuple
import json

import pandas as pd


def flatten_geojson(gj: dict) -> pd.DataFrame:
    """
    Aplati un objet GeoJSON en DataFrame.

    Chaque feature est transformée en une ligne :
    - les clés du bloc "properties" deviennent des colonnes ;
    - les clés du bloc "publisher" sont ajoutées avec le préfixe "publisher_".

    Paramètres
    ----------
    gj : dict
        Contenu du GeoJSON déjà chargé en mémoire.

    Retour
    ------
    DataFrame
        Table tabulaire correspondant aux features du GeoJSON.
    """
    rows = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        publisher = feat.get("publisher", {}) or {}

        row = dict(props)
        for key, value in publisher.items():
            row[f"publisher_{key}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def build_aires_voies_flat(
    data_dir: Path,
    aires_geojson_name: str = "aires.geojson",
    voies_geojson_name: str = "voies.geojson",
    aires_flat_name: str = "aires_flat.csv",
    voies_flat_name: str = "voies_flat.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit les tables aires_flat et voies_flat à partir des GeoJSON.

    Étapes :
    - lecture des fichiers aires.geojson et voies.geojson ;
    - aplatissement via flatten_geojson ;
    - sauvegarde des résultats au format CSV dans le dossier data.

    Paramètres
    ----------
    data_dir : Path
        Dossier contenant les fichiers de données (data/).
    aires_geojson_name : str
        Nom du fichier GeoJSON d'aires.
    voies_geojson_name : str
        Nom du fichier GeoJSON de voies.
    aires_flat_name : str
        Nom du fichier CSV de sortie pour aires_flat.
    voies_flat_name : str
        Nom du fichier CSV de sortie pour voies_flat.

    Retour
    ------
    aires_df : DataFrame
        Table aplatie des aires ZFE.
    voies_df : DataFrame
        Table aplatie des voies ZFE.
    """
    data_dir = Path(data_dir)

    aires_path = data_dir / aires_geojson_name
    voies_path = data_dir / voies_geojson_name

    if not aires_path.exists():
        raise FileNotFoundError(f"Fichier GeoJSON introuvable : {aires_path}")
    if not voies_path.exists():
        raise FileNotFoundError(f"Fichier GeoJSON introuvable : {voies_path}")

    with aires_path.open(encoding="utf-8") as f:
        aires_gj = json.load(f)

    with voies_path.open(encoding="utf-8") as f:
        voies_gj = json.load(f)

    aires_df = flatten_geojson(aires_gj)
    voies_df = flatten_geojson(voies_gj)

    aires_df.to_csv(data_dir / aires_flat_name, index=False)
    voies_df.to_csv(data_dir / voies_flat_name, index=False)

    return aires_df, voies_df


def build_zfe_clean_tables(
    data_dir: Path,
    aires_flat_name: str = "aires_flat.csv",
    voies_flat_name: str = "voies_flat.csv",
    zfe_ids_name: str = "zfe_ids.csv",
    aires_clean_name: str = "aires_clean.csv",
    voies_clean_name: str = "voies_clean.csv",
    zfe_meta_name: str = "zfe_meta.csv",
    sep_zfe_ids: str = ";",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construit aires_clean, voies_clean et zfe_meta à partir de aires_flat,
    voies_flat et de la table d'identifiants zfe_ids.

    Cette fonction reprend pas à pas le code utilisé dans le notebook :

    1. Lecture des tables intermédiaires aires_flat, voies_flat et de zfe_ids.csv.
    2. Sélection des colonnes pertinentes pour aires_clean et voies_clean.
    3. Conversion des colonnes de dates en datetime.
    4. Construction de zfe_meta :
       - agrégation par ZFE (publisher_zfe_id, publisher_siren, publisher_nom),
       - calcul des dates de début et de fin,
       - nombre d'aires couvertes,
       - indicateur de présence de restriction VP.
    5. Jointure avec zfe_ids sur le SIREN pour enrichir les métadonnées.
    6. Sauvegarde des trois tables au format CSV dans le dossier data.

    Paramètres
    ----------
    data_dir : Path
        Dossier contenant les fichiers de données.
    aires_flat_name : str
        Nom du fichier CSV aires_flat.
    voies_flat_name : str
        Nom du fichier CSV voies_flat.
    zfe_ids_name : str
        Nom du fichier CSV zfe_ids.
    aires_clean_name : str
        Nom du fichier CSV de sortie pour aires_clean.
    voies_clean_name : str
        Nom du fichier CSV de sortie pour voies_clean.
    zfe_meta_name : str
        Nom du fichier CSV de sortie pour zfe_meta.
    sep_zfe_ids : str
        Séparateur utilisé dans zfe_ids.csv.

    Retour
    ------
    aires_clean : DataFrame
        Version nettoyée de aires_flat.
    voies_clean : DataFrame
        Version nettoyée de voies_flat.
    zfe_meta : DataFrame
        Table de métadonnées ZFE (une ligne par ZFE).
    """
    data_dir = Path(data_dir)

    aires = pd.read_csv(data_dir / aires_flat_name)
    voies = pd.read_csv(data_dir / voies_flat_name)
    zfe_ids = pd.read_csv(data_dir / zfe_ids_name, sep=sep_zfe_ids)

    # Colonnes retenues pour aires_clean et voies_clean
    aires_keep = [
        "publisher_zfe_id",
        "publisher_nom",
        "publisher_siren",
        "publisher_forme_juridique",
        "id",
        "date_debut",
        "date_fin",
        "vp_critair",
        "vp_horaires",
        "vul_critair",
        "vul_horaires",
        "pl_critair",
        "pl_horaires",
        "autobus_autocars_critair",
        "autobus_autocars_horaires",
        "deux_rm_critair",
        "deux_rm_horaires",
        "url_arrete",
        "url_site_information",
    ]

    voies_keep = [
        "publisher_zfe_id",
        "publisher_nom",
        "publisher_siren",
        "publisher_forme_juridique",
        "id",
        "osm_id",
        "ref",
        "one_way",
        "date_debut",
        "date_fin",
        "vp_critair",
        "vp_horaires",
        "vul_critair",
        "vul_horaires",
        "pl_critair",
        "pl_horaires",
        "autobus_autocars_critair",
        "autobus_autocars_horaires",
        "deux_rm_critair",
        "deux_rm_horaires",
        "zfe_derogation",
        "url_arrete",
        "url_site",
        "url_site_information",
    ]

    aires_clean = aires[aires_keep].copy()
    voies_clean = voies[voies_keep].copy()

    # Conversion des dates en datetime
    for df in (aires_clean, voies_clean):
        for col in ["date_debut", "date_fin"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Table méta ZFE (une ligne par ZFE)
    zfe_meta = (
        aires_clean
        .groupby(
            ["publisher_zfe_id", "publisher_siren", "publisher_nom"],
            as_index=False,
        )
        .agg(
            first_date_debut=("date_debut", "min"),
            last_date_debut=("date_debut", "max"),
            first_date_fin=(
                "date_fin",
                lambda s: s.dropna().min() if s.notna().any() else pd.NaT,
            ),
            n_aires=("id", "nunique"),
            has_vp_restriction=("vp_critair", lambda s: s.notna().any()),
        )
    )

    # Jointure avec zfe_ids pour enrichir les métadonnées (siren, forme juridique, etc.)
    zfe_ids = zfe_ids.copy()
    zfe_ids["siren"] = zfe_ids["siren"].astype(int)

    zfe_meta = zfe_meta.merge(
        zfe_ids,
        left_on="publisher_siren",
        right_on="siren",
        how="left",
    )

    # Sauvegarde des tables nettoyées
    aires_clean.to_csv(data_dir / aires_clean_name, index=False)
    voies_clean.to_csv(data_dir / voies_clean_name, index=False)
    zfe_meta.to_csv(data_dir / zfe_meta_name, index=False)

    return aires_clean, voies_clean, zfe_meta
