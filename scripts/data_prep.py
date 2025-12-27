# scripts/data_prep.py

from __future__ import annotations

from typing import Tuple, List

import pandas as pd


def summarize_missing_daily(
    df: pd.DataFrame,
    id_col: str = "station_id",
    date_col: str = "date",
    value_col: str = "no2_ug_m3",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Résume les valeurs manquantes dans une série journalière par station.

    Hypothèses :
    - df contient au moins les colonnes : id_col, date_col, value_col ;
    - id_col identifie une station (station_id) ;
    - date_col est convertible en datetime et correspond à une fréquence journalière
      (ou quasi journalière) ;
    - value_col contient la concentration de NO2 (ou une autre mesure).

    Méthode :
    - pour chaque station, on reconstruit une grille journalière complète entre
      la première et la dernière date observée ;
    - on considère un jour comme manquant si value_col est absente ou NaN ;
    - on retourne :
        1) un résumé avec le nombre total de jours manquants par station ;
        2) la liste des intervalles de jours consécutifs manquants.

    Paramètres
    ----------
    df : DataFrame
        Jeu de données journalier (ou quasi journalier).
    id_col : str
        Nom de la colonne identifiant la station (par défaut 'station_id').
    date_col : str
        Nom de la colonne de date (par défaut 'date').
    value_col : str
        Nom de la colonne contenant la valeur mesurée (par défaut 'no2_ug_m3').

    Retour
    ------
    summary_df : DataFrame
        Une ligne par station avec :
        station_id, station_name (si disponible), station_env, station_influence,
        total_jours_manquants, date_min, date_max.
    gaps_df : DataFrame
        Une ligne par intervalle de jours consécutifs manquants, avec :
        station_id, station_name, gap_start, gap_end, gap_length_days.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    summary_rows = []
    gaps_rows = []

    for sid, sub in df.groupby(id_col):
        sub = sub.sort_values(date_col)

        # Informations descriptives (optionnelles si les colonnes existent)
        station_name = sub.get("station_name", pd.Series([None])).iloc[0]
        station_env = sub.get("station_env", pd.Series([None])).iloc[0]
        station_influence = sub.get("station_influence", pd.Series([None])).iloc[0]

        # Grille journalière complète sur l'intervalle observé pour la station
        start = sub[date_col].min()
        end = sub[date_col].max()
        full_index = pd.date_range(start, end, freq="D")

        sub_full = (
            sub.set_index(date_col)[[value_col]]
            .reindex(full_index)
            .rename_axis(date_col)
        )

        missing_mask = sub_full[value_col].isna()
        total_missing = int(missing_mask.sum())

        summary_rows.append(
            {
                "station_id": sid,
                "station_name": station_name,
                "station_env": station_env,
                "station_influence": station_influence,
                "total_jours_manquants": total_missing,
                "date_min": start.date(),
                "date_max": end.date(),
            }
        )

        if total_missing == 0:
            continue

        # Construction des intervalles de jours consécutifs manquants
        missing_dates = full_index[missing_mask]

        gap_start = None
        previous_date = None

        for current_date in missing_dates:
            if gap_start is None:
                gap_start = current_date
                previous_date = current_date
            elif (current_date - previous_date).days == 1:
                # Toujours dans le même intervalle de jours consécutifs
                previous_date = current_date
            else:
                # Fin d'un intervalle, on enregistre
                gaps_rows.append(
                    {
                        "station_id": sid,
                        "station_name": station_name,
                        "gap_start": gap_start.date(),
                        "gap_end": previous_date.date(),
                        "gap_length_days": (previous_date - gap_start).days + 1,
                    }
                )
                gap_start = current_date
                previous_date = current_date

        # Dernier intervalle pour la station
        if gap_start is not None:
            gaps_rows.append(
                {
                    "station_id": sid,
                    "station_name": station_name,
                    "gap_start": gap_start.date(),
                    "gap_end": previous_date.date(),
                    "gap_length_days": (previous_date - gap_start).days + 1,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "total_jours_manquants", ascending=False
    )
    gaps_df = pd.DataFrame(gaps_rows).sort_values(["station_id", "gap_start"])

    return summary_df, gaps_df

def interpolate_daily_per_station(
    df: pd.DataFrame,
    id_col: str = "station_id",
    date_col: str = "date",
    value_col: str = "no2_ug_m3",
    max_missing_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complète les valeurs manquantes par interpolation temporelle, station par station.

    Pour chaque station_id :
    - on reconstruit une grille journalière complète entre la première et la dernière
      date observée ;
    - on calcule la part de jours manquants ;
    - si cette part dépasse max_missing_ratio, la station est exclue ;
    - sinon, on interpole les valeurs manquantes dans value_col avec method="time",
      en complétant dans les deux directions.

    Paramètres
    ----------
    df : DataFrame
        Jeu de données journalier avec au moins id_col, date_col, value_col.
    id_col : str
        Nom de la colonne identifiant la station (par défaut 'station_id').
    date_col : str
        Nom de la colonne de dates (par défaut 'date').
    value_col : str
        Nom de la colonne à interpoler (par défaut 'no2_ug_m3').
    max_missing_ratio : float
        Part maximale de jours manquants tolérée pour garder une station.
        Au-delà, la station est exclue.

    Retour
    ------
    df_filled : DataFrame
        Jeu de données avec grille journalière complète par station et valeurs
        interpolées pour value_col.
    dropped_ids : list of str
        Liste des station_id exclus pour cause de trop forte proportion
        de valeurs manquantes.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    filled_list = []
    dropped_ids: List[str] = []

    for sid, sub in df.groupby(id_col):
        sub = sub.sort_values(date_col)

        start = sub[date_col].min()
        end = sub[date_col].max()
        full_index = pd.date_range(start, end, freq="D")

        sub_full = (
            sub.set_index(date_col)
            .reindex(full_index)
        )

        n_days = len(full_index)
        n_missing = int(sub_full[value_col].isna().sum())
        missing_ratio = n_missing / n_days if n_days > 0 else 0.0

        if missing_ratio > max_missing_ratio:
            dropped_ids.append(str(sid))
            continue

        sub_full[value_col] = sub_full[value_col].interpolate(
            method="time",
            limit_direction="both",
        )

        # On réinjecte les méta-informations stationaires
        sub_full[id_col] = sid
        for col in ["station_name", "station_env", "station_influence", "zone"]:
            if col in sub.columns:
                sub_full[col] = sub[col][col].iloc[0] if isinstance(sub[col], pd.DataFrame) else sub[col].iloc[0]

        sub_full = sub_full.reset_index().rename(columns={"index": date_col})
        filled_list.append(sub_full)

    if not filled_list:
        return pd.DataFrame(columns=df.columns), dropped_ids

    df_filled = pd.concat(filled_list, ignore_index=True)
    df_filled = df_filled.sort_values([id_col, date_col])

    return df_filled, dropped_ids
