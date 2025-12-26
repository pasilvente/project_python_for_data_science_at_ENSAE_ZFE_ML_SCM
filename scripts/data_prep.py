"""
Fonctions de préparation des données pour le projet ZFE / NO2 :
- agrégation journalière -> mensuelle,
- diagnostic des valeurs manquantes.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def build_monthly_series(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
    group_cols=("station_id", "station_name"),
    min_days: int = 10,
) -> pd.DataFrame:
    """
    Agrège une base journalière en base mensuelle par station.

    Paramètres
    ----------
    df : DataFrame
        Données journalières, contenant au moins les colonnes `date_col`,
        `value_col` et les colonnes listées dans `group_cols`.
    value_col : str
        Nom de la colonne contenant la valeur numérique à agréger
        (par exemple "no2_ug_m3").
    date_col : str, optionnel
        Nom de la colonne de date (par défaut "date").
    group_cols : tuple de str, optionnel
        Colonnes d’identification des séries (par exemple station_id, station_name).
    min_days : int, optionnel
        Nombre minimal de jours observés dans le mois pour calculer une moyenne.
        Si ce seuil n’est pas atteint, la valeur mensuelle est mise à NaN.

    Retour
    ------
    DataFrame
        DataFrame mensuelle avec les colonnes "date", `group_cols` et `value_col`.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    group_cols = list(group_cols)

    # Agrégation journalière -> mensuelle (moyenne et nombre de jours observés)
    monthly = (
        df
        .groupby(group_cols)[value_col]
        .resample("MS")
        .agg(["mean", "count"])
        .reset_index()
    )
    # monthly contient : group_cols + ["date", "mean", "count"]

    # Nombre minimal de jours observés par mois
    mask_insufficient = monthly["count"] < min_days
    monthly.loc[mask_insufficient, "mean"] = np.nan

    monthly = monthly.rename(columns={"mean": value_col})
    monthly["date"] = pd.to_datetime(monthly["date"])

    cols = ["date"] + group_cols + [value_col]
    monthly = monthly[cols].sort_values(group_cols + ["date"])

    return monthly


def summarize_missing_daily(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
    station_col: str = "station_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produit un diagnostic des valeurs manquantes sur une base journalière.

    Retourne deux DataFrame :
    - un résumé par station (nombre de jours manquants, date min/max),
    - la liste détaillée des intervalles consécutifs de jours manquants.

    Paramètres
    ----------
    df : DataFrame
        Données journalières.
    value_col : str
        Nom de la colonne de valeur.
    date_col : str
        Nom de la colonne de date.
    station_col : str
        Nom de la colonne identifiant la station.

    Retour
    ------
    summary_df : DataFrame
        Résumé par station.
    gaps_df : DataFrame
        Intervalles consécutifs de jours manquants.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[station_col] = df[station_col].astype(str).str.strip()

    summary_rows = []
    gaps_rows = []

    for station_id, sub in df.groupby(station_col):
        sub = sub.sort_values(date_col).set_index(date_col)

        full_index = pd.date_range(start=sub.index.min(), end=sub.index.max(), freq="D")
        sub_full = sub.reindex(full_index)

        is_missing = sub_full[value_col].isna()
        total_missing = int(is_missing.sum())

        summary_rows.append(
            {
                station_col: station_id,
                "total_missing_days": total_missing,
                "date_min": sub_full.index.min(),
                "date_max": sub_full.index.max(),
            }
        )

        if not is_missing.any():
            continue

        prev_missing = is_missing.shift(1, fill_value=False)
        next_missing = is_missing.shift(-1, fill_value=False)

        start_mask = is_missing & ~prev_missing
        end_mask = is_missing & ~next_missing

        starts = sub_full.index[start_mask]
        ends = sub_full.index[end_mask]

        for start, end in zip(starts, ends):
            length = (end - start).days + 1
            gaps_rows.append(
                {
                    station_col: station_id,
                    "gap_start": start,
                    "gap_end": end,
                    "gap_length_days": length,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "total_missing_days", ascending=False
    )
    gaps_df = pd.DataFrame(gaps_rows).sort_values([station_col, "gap_start"])

    return summary_df, gaps_df
