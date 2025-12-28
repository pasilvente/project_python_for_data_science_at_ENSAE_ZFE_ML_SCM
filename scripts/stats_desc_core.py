"""
stats_desc_core.py
==================

Briques "métier" (sans figures) pour les statistiques descriptives NO₂.

Objectif :
- Centraliser les transformations / agrégations (tables) utilisées par les figures.
- Le notebook rapport_zfe orchestre et raconte, ce module calcule.

Conventions projet (attendues en entrée) :
- date : datetime-like ou convertible
- station_id : identifiant station (str / int convertible)
- no2_ug_m3 : concentration NO₂ journalière (float)

Colonnes optionnelles :
- station_name, lat, lon
- station_env, station_influence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = ("date", "station_id", "no2_ug_m3")


def _ensure_required_cols(df: pd.DataFrame, required: Sequence[str] = REQUIRED_COLS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes: {missing}. Attendu au minimum: {list(required)}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )


def normalize_no2_daily(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """
    Normalise un DataFrame journalier NO₂ au schéma projet + ajoute la colonne 'group'.
    """
    _ensure_required_cols(df)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["station_id"] = out["station_id"].astype(str)
    out["no2_ug_m3"] = pd.to_numeric(out["no2_ug_m3"], errors="coerce")
    out["group"] = group
    out = out.dropna(subset=["date", "station_id", "no2_ug_m3"]).sort_values(["station_id", "date"])
    return out.reset_index(drop=True)


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Hiver"
    if month in (3, 4, 5):
        return "Printemps"
    if month in (6, 7, 8):
        return "Été"
    return "Automne"


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["season"] = out["month"].map(season_from_month)
    return out


@dataclass(frozen=True)
class CommonNo2Daily:
    common_daily: pd.DataFrame
    groups: Tuple[str, ...]


def build_common_daily(
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    group_grenoble: str = "Grenoble",
    group_paris: str = "Paris",
    group_donors: str = "Donneuses",
) -> CommonNo2Daily:
    d0 = normalize_no2_daily(donors_daily, group_donors)
    d1 = normalize_no2_daily(grenoble_daily, group_grenoble)
    d2 = normalize_no2_daily(paris_daily, group_paris)

    common = pd.concat([d0, d1, d2], axis=0, ignore_index=True, sort=False)
    common = common.dropna(subset=["date", "station_id", "no2_ug_m3"]).sort_values(["group", "station_id", "date"])
    common = add_calendar_columns(common)
    return CommonNo2Daily(common_daily=common.reset_index(drop=True), groups=(group_donors, group_grenoble, group_paris))


def build_stations_meta(common_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Construit une table stations_meta (1 ligne par station) à partir du dataset commun.
    """
    _ensure_required_cols(common_daily)
    cols = [c for c in ["station_id", "station_name", "lat", "lon", "station_env", "station_influence"] if c in common_daily.columns]
    if "station_id" not in cols:
        cols = ["station_id"] + cols
    meta = (
        common_daily[cols]
        .drop_duplicates("station_id")
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    return meta


def _mean_top_share(x: np.ndarray, share: float) -> float:
    if x.size == 0:
        return np.nan
    x_sorted = np.sort(x)
    k = max(1, int(np.ceil(share * x_sorted.size)))
    return float(np.mean(x_sorted[-k:]))


def compute_station_year_metrics(
    common_daily: pd.DataFrame,
    thresholds_daily: Sequence[float] = (30.0, 40.0),
    percentiles: Sequence[int] = (90, 95),
    top_share: float = 0.10,
    annual_limit: float = 40.0,
) -> pd.DataFrame:
    """
    Calcule des indicateurs par station et par année.
    """
    _ensure_required_cols(common_daily)
    df = common_daily.copy()
    if "year" not in df.columns:
        df = add_calendar_columns(df)

    def agg_station_year(g: pd.DataFrame) -> pd.Series:
        x = g["no2_ug_m3"].to_numpy(dtype=float)
        mean_annual = float(np.nanmean(x)) if x.size else np.nan
        out = {
            "mean_annual": mean_annual,
            "n_days": int(np.sum(~np.isnan(x))),
            "above_annual_limit": float(mean_annual > annual_limit) if np.isfinite(mean_annual) else np.nan,
        }
        for thr in thresholds_daily:
            out[f"pct_days_gt_{int(thr)}"] = float(np.nanmean(x > thr) * 100.0) if x.size else np.nan
        for q in percentiles:
            out[f"p{q}"] = float(np.nanpercentile(x, q)) if x.size else np.nan
        out["top10_mean"] = _mean_top_share(x[~np.isnan(x)], top_share)
        return pd.Series(out)

    metrics = (
        df.groupby(["group", "station_id", "year"], as_index=False)
        .apply(agg_station_year)
        .reset_index(drop=True)
    )
    return metrics


def summarize_group_year_from_station_metrics(station_year_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les indicateurs station-année au niveau groupe-année.
    """
    df = station_year_metrics.copy()
    required = {"group", "year", "station_id", "mean_annual"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"station_year_metrics doit contenir {sorted(required)}. Manquantes: {missing}")

    def agg_group_year(g: pd.DataFrame) -> pd.Series:
        out = {
            "n_stations": int(g["station_id"].nunique()),
            "mean_mean_annual": float(np.nanmean(g["mean_annual"])),
            "share_stations_above_40": float(np.nanmean(g["mean_annual"] > 40.0) * 100.0),
        }
        for c in [c for c in g.columns if c.startswith("pct_days_gt_")]:
            out[f"mean_{c}"] = float(np.nanmean(g[c]))
        for q in (90, 95):
            col = f"p{q}"
            if col in g.columns:
                out[f"mean_{col}"] = float(np.nanmean(g[col]))
        if "top10_mean" in g.columns:
            out["mean_top10"] = float(np.nanmean(g["top10_mean"]))
        return pd.Series(out)

    summary = (
        df.groupby(["group", "year"], as_index=False)
        .apply(agg_group_year)
        .reset_index(drop=True)
        .sort_values(["group", "year"])
        .reset_index(drop=True)
    )
    return summary


def compute_oms_station_year_metrics(
    common_daily: pd.DataFrame,
    annual_oms: float = 10.0,
    daily_oms: float = 25.0,
    daily_proxy_ue: float = 40.0,
) -> pd.DataFrame:
    """Indicateurs OMS par station-année."""
    _ensure_required_cols(common_daily)
    df = common_daily.copy()
    if "year" not in df.columns:
        df = add_calendar_columns(df)

    def agg(g: pd.DataFrame) -> pd.Series:
        x = g["no2_ug_m3"].to_numpy(dtype=float)
        mean_annual = float(np.nanmean(x)) if x.size else np.nan
        return pd.Series(
            {
                "mean_annual": mean_annual,
                "above_oms_10": float(mean_annual > annual_oms) if np.isfinite(mean_annual) else np.nan,
                "pct_days_gt_25": float(np.nanmean(x > daily_oms) * 100.0) if x.size else np.nan,
                "pct_days_gt_40": float(np.nanmean(x > daily_proxy_ue) * 100.0) if x.size else np.nan,
                "n_days": int(np.sum(~np.isnan(x))),
            }
        )

    out = (
        df.groupby(["group", "station_id", "year"], as_index=False)
        .apply(agg)
        .reset_index(drop=True)
    )
    return out


def summarize_oms_group_year(oms_station_year: pd.DataFrame) -> pd.DataFrame:
    """Agrégation OMS au niveau groupe-année."""
    required = {"group", "year", "station_id", "above_oms_10", "pct_days_gt_25", "pct_days_gt_40"}
    missing = [c for c in required if c not in oms_station_year.columns]
    if missing:
        raise ValueError(f"oms_station_year doit contenir {sorted(required)}. Manquantes: {missing}")

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_stations": int(g["station_id"].nunique()),
                "share_stations_above_oms_10": float(np.nanmean(g["above_oms_10"]) * 100.0),
                "mean_pct_days_gt_25": float(np.nanmean(g["pct_days_gt_25"])),
                "mean_pct_days_gt_40": float(np.nanmean(g["pct_days_gt_40"])),
            }
        )

    return (
        oms_station_year.groupby(["group", "year"], as_index=False)
        .apply(agg)
        .reset_index(drop=True)
        .sort_values(["group", "year"])
        .reset_index(drop=True)
    )
