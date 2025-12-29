"""
Boîte à outils pour l'analyse descriptive des séries de NO2.

Contient :
- Fonctions pour une station ciblée (Grenoble Boulevards, Paris Champs-Élysées) :
    - plot_daily_with_smoothing
    - compute_pre_post_summary
    - plot_monthly_seasonality_pre_post
    - plot_weekly_pattern_pre_post

- Fonctions pour analyser les stations donneuses :
    - plot_stations_map
    - summarise_donors_pre_post
    - plot_preZFE_boxplot_treated_vs_donors
    - plot_donor_deltas_pre_post
    - compute_preZFE_correlations
    - plot_preZFE_correlations
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point



# OUTILS POUR UNE STATION CIBLE (Grenoble / Paris)

def plot_daily_with_smoothing(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window: int = 30,
    treatment_start: Optional[pd.Timestamp] = None,
    covid_start: Optional[pd.Timestamp] = None,
    covid_end: Optional[pd.Timestamp] = None,
    label_series: str = "Station traitée - NO₂ quotidien",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Trace la série quotidienne + une moyenne mobile centrée.

    Paramètres
    ----------
    df : DataFrame
        Données journalières pour une station (une ligne par jour).
    date_col : str
        Nom de la colonne de date.
    value_col : str
        Nom de la colonne de NO2.
    window : int
        Fenêtre de moyenne mobile (en jours).
    treatment_start : Timestamp ou None
        Date de début de la ZFE (ligne verticale pointillée).
    covid_start, covid_end : Timestamp ou None
        Période Covid à surligner en gris.
    label_series : str
        Légende pour la série brute.
    ax : Axes ou None
        Axes matplotlib existant. Si None, un nouvel axe est créé.

    Retour
    ------
    ax : Axes
        Axes sur lesquels la figure est dessinée.
    """
    df = df.copy().sort_values(date_col)
    dates = pd.to_datetime(df[date_col])
    values = df[value_col].astype(float)

    smooth = values.rolling(window=window, center=True, min_periods=1).mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(dates, values, color="C0", alpha=0.2, label=label_series)
    ax.plot(dates, smooth, color="C1", linewidth=2, label=f"Moyenne mobile {window} jours")

    if treatment_start is not None:
        ax.axvline(treatment_start, color="red", linestyle="--", linewidth=1.5, label="Début ZFE")

    if (covid_start is not None) and (covid_end is not None):
        ax.axvspan(covid_start, covid_end, color="grey", alpha=0.2, label="Période Covid")

    ax.set_xlabel("Date")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.legend(loc="upper right")

    return ax


def compute_pre_post_summary(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    treatment_start: pd.Timestamp,
    covid_start: Optional[pd.Timestamp] = None,
    covid_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Résumé statistique pré / post ZFE pour une station.

    Retourne un DataFrame avec index :
        - 'pré_ZFE'
        - 'post_ZFE'
        - 'post_ZFE_sans_Covid'

    et colonnes :
        mean, median, std, p10, p90, n_days
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    pre = df[df[date_col] < treatment_start]
    post = df[df[date_col] >= treatment_start]

    def _summary(sub: pd.DataFrame) -> pd.Series:
        vals = sub[value_col].dropna().astype(float)
        return pd.Series(
            {
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "p10": vals.quantile(0.10),
                "p90": vals.quantile(0.90),
                "n_days": len(vals),
            }
        )

    rows = {
        "pré_ZFE": _summary(pre),
        "post_ZFE": _summary(post),
    }

    if covid_start is not None and covid_end is not None:
        covid_mask = (post[date_col] >= covid_start) & (post[date_col] <= covid_end)
        post_no_covid = post.loc[~covid_mask]
        rows["post_ZFE_sans_Covid"] = _summary(post_no_covid)
    else:
        # si pas de Covid spécifié, on recopie la colonne post_ZFE
        rows["post_ZFE_sans_Covid"] = rows["post_ZFE"]

    summary = pd.DataFrame(rows).T
    return summary


def plot_monthly_seasonality_pre_post(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    treatment_start: pd.Timestamp,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Compare la saisonnalité moyenne mensuelle pré / post ZFE.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.month

    pre = df[df[date_col] < treatment_start]
    post = df[df[date_col] >= treatment_start]

    monthly_pre = pre.groupby("month")[value_col].mean()
    monthly_post = post.groupby("month")[value_col].mean()

    months = np.arange(1, 13)
    monthly_pre = monthly_pre.reindex(months)
    monthly_post = monthly_post.reindex(months)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(months, monthly_post, marker="o", label="Post-ZFE")
    ax.plot(months, monthly_pre, marker="o", label="Pré-ZFE")

    ax.set_xticks(months)
    ax.set_xlabel("Mois de l'année")
    ax.set_ylabel("NO₂ moyen (µg/m³)")
    ax.legend(title="Période")
    return ax


def plot_weekly_pattern_pre_post(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    treatment_start: pd.Timestamp,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Compare NO2 moyen en semaine vs week-end pré / post ZFE.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["dow"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["dow"] >= 5

    pre = df[df[date_col] < treatment_start]
    post = df[df[date_col] >= treatment_start]

    def _weekly_means(sub: pd.DataFrame) -> pd.Series:
        weekday = sub.loc[~sub["is_weekend"], value_col].mean()
        weekend = sub.loc[sub["is_weekend"], value_col].mean()
        return pd.Series({"weekday": weekday, "weekend": weekend})

    pre_m = _weekly_means(pre)
    post_m = _weekly_means(post)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(2)
    width = 0.35

    ax.bar(x - width / 2, [post_m["weekday"], post_m["weekend"]],
           width=width, label="Post-ZFE")
    ax.bar(x + width / 2, [pre_m["weekday"], pre_m["weekend"]],
           width=width, label="Pré-ZFE")

    ax.set_xticks(x)
    ax.set_xticklabels(["Semaine", "Week-end"])
    ax.set_ylabel("NO₂ moyen (µg/m³)")
    ax.legend(title="Période")

    return ax



# OUTILS POUR LES STATIONS DONNEUSES


def plot_stations_map(
    stations: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    station_id_col: str = "station_id",
    treated_ids: Optional[Sequence[str]] = None,
    ax: Optional["plt.Axes"] = None,
    zoom: int = 6,
) -> "plt.Axes":
    """
    Carte avec fond OpenStreetMap : stations NO₂ donneuses vs traitées.

    Paramètres
    ----------
    stations : DataFrame
        Doit contenir au moins les colonnes :
        - station_id_col (par défaut 'station_id')
        - lat_col        (par défaut 'lat')
        - lon_col        (par défaut 'lon')
    lat_col, lon_col : str
        Noms des colonnes latitude / longitude (en WGS84).
    station_id_col : str
        Nom de la colonne identifiant les stations.
    treated_ids : séquence d'identifiants de stations traitées
        (ex. ['FR15046', 'FR01034']). Comparaison en str.
    ax : Axes
        Axe matplotlib optionnel. Si None, un nouvel axe est créé.
    zoom : int
        Niveau de zoom pour le fond de carte (contextily).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Copie défensive + nettoyage des NaN de coordonnées
    df = stations.copy()
    df = df.dropna(subset=[lat_col, lon_col])

    # Tout en string pour éviter les surprises
    df[station_id_col] = df[station_id_col].astype(str)
    if treated_ids is None:
        treated_ids_str: set[str] = set()
    else:
        treated_ids_str = {str(s) for s in treated_ids}

    # Passage en GeoDataFrame puis reprojection en Web Mercator
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # Flag stations traitées
    gdf["is_treated"] = gdf[station_id_col].isin(treated_ids_str)
    donors = gdf[~gdf["is_treated"]]
    treated = gdf[gdf["is_treated"]]

    # Points donneurs
    if not donors.empty:
        donors.plot(
            ax=ax,
            marker="o",
            color="tab:blue",
            edgecolor="k",
            linewidth=0.3,
            alpha=0.8,
            label="Stations donneuses",
        )

    # Points traités (Grenoble + Paris par ex.)
    if not treated.empty:
        treated.plot(
            ax=ax,
            marker="*",
            color="red",
            edgecolor="k",
            linewidth=0.6,
            markersize=160,
            label="Stations traitées",
        )

    # Fond de carte
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)

    ax.set_axis_off()
    ax.set_title("Stations NO₂ (donneuses + traitées)")
    ax.legend(loc="upper right")

    return ax



def summarise_donors_pre_post(
    df: pd.DataFrame,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    value_col: str = "no2_ug_m3",
    min_days_pre: int = 30,
    min_days_post: int = 30,
) -> pd.DataFrame:
    """
    Résume, pour chaque station, le NO2 moyen pré- et post-ZFE.

    Retourne un DataFrame indexé par station_id avec colonnes :
        - mean_pre, n_pre
        - mean_post, n_post
        - delta_post_pre = mean_post - mean_pre

    Les stations avec trop peu de jours en pré ou post sont filtrées.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    pre = df[df[date_col] < treatment_start]
    post = df[df[date_col] >= treatment_start]

    agg_pre = (
        pre.groupby(station_id_col)[value_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_pre", "count": "n_pre"})
    )
    agg_post = (
        post.groupby(station_id_col)[value_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_post", "count": "n_post"})
    )

    summary = agg_pre.join(agg_post, how="outer")

    summary["delta_post_pre"] = summary["mean_post"] - summary["mean_pre"]

    mask = (summary["n_pre"] >= min_days_pre) & (summary["n_post"] >= min_days_post)
    summary = summary.loc[mask].sort_index()

    return summary


def plot_preZFE_boxplot_treated_vs_donors(
    summary_df: pd.DataFrame,
    treated_id: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Boxplot des niveaux moyens pré-ZFE des donneurs + moyenne de la station traitée.

    Paramètres
    ----------
    summary_df : DataFrame
        Sortie de summarise_donors_pre_post (index = station_id).
    treated_id : str
        Identifiant de la station traitée (présente dans l'index).
    """
    if "mean_pre" not in summary_df.columns:
        raise ValueError("summary_df doit contenir une colonne 'mean_pre'.")

    mean_pre = summary_df["mean_pre"].dropna()

    if treated_id not in mean_pre.index:
        raise ValueError(f"Station traitée '{treated_id}' absente de summary_df.")

    treated_mean = float(mean_pre.loc[treated_id])
    donor_means = mean_pre.drop(index=treated_id)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))

    bp = ax.boxplot(
        [donor_means.values],
        positions=[1],
        widths=0.5,
        vert=True,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.scatter(
        1,
        treated_mean,
        color="red",
        zorder=3,
        label="Station traitée (moyenne pré-ZFE)",
    )

    ax.set_xticks([1])
    ax.set_xticklabels(["Donneurs"])
    ax.set_ylabel("NO₂ moyen pré-ZFE (µg/m³)")
    ax.set_title("Niveau moyen pré-ZFE : station traitée vs donneurs")
    ax.legend(loc="upper right")

    return ax

def plot_preZFE_boxplot_treated_vs_donors_daily(
    df: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    value_col: str = "no2_ug_m3",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Boxplot des NO2 quotidiens pré-ZFE :
      - moyenne quotidienne des donneurs,
      - station traitée.

    Idée : comparer la distribution journalière de la station traitée
    à celle du "panel moyen" des donneurs.

    Paramètres
    ----------
    df : DataFrame
        Données journalières pour station traitée + donneurs.
    treated_id : str
        Identifiant de la station traitée.
    treatment_start : Timestamp
        Date de début de la ZFE (on ne garde que les jours < treatment_start).
    date_col, station_id_col, value_col : str
        Noms des colonnes.
    ax : Axes ou None
        Axe matplotlib existant. Si None, un nouvel axe est créé.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Pré-ZFE uniquement
    pre = df[df[date_col] < treatment_start]

    # Station traitée
    treated_pre = pre.loc[pre[station_id_col] == treated_id]
    if treated_pre.empty:
        raise ValueError(f"Aucune donnée pré-ZFE pour la station traitée '{treated_id}'.")

    treated_series = (
        treated_pre
        .set_index(date_col)[value_col]
        .astype(float)
        .sort_index()
    )

    # Donneurs : moyenne quotidienne sur toutes les stations ≠ treated
    donors_pre = pre.loc[pre[station_id_col] != treated_id]
    if donors_pre.empty:
        raise ValueError("Aucune donnée de donneur pré-ZFE dans df.")

    donors_daily = (
        donors_pre
        .groupby(date_col)[value_col]
        .mean()
        .astype(float)
        .sort_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    bp = ax.boxplot(
        [donors_daily.values, treated_series.values],
        positions=[1, 2],
        widths=0.6,
        vert=True,
        patch_artist=True,
    )

    # Couleurs sympa
    colors = ["lightblue", "salmon"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Donneurs (moyenne quotidienne)", "Station traitée"])
    ax.set_ylabel("NO₂ quotidien pré-ZFE (µg/m³)")
    ax.set_title("NO₂ quotidien pré-ZFE : station traitée vs donneurs")

    return ax



def plot_donor_deltas_pre_post(
    summary_df: pd.DataFrame,
    treated_id: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Barplot des variations moyennes de NO2 (post - pré) pour chaque station.

    La station traitée est mise en évidence via une ligne horizontale.
    """
    if "delta_post_pre" not in summary_df.columns:
        raise ValueError("summary_df doit contenir une colonne 'delta_post_pre'.")

    deltas = summary_df["delta_post_pre"].dropna()

    if treated_id not in deltas.index:
        raise ValueError(f"Station traitée '{treated_id}' absente de summary_df.")

    treated_delta = float(deltas.loc[treated_id])
    donor_deltas = deltas.drop(index=treated_id).sort_values()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(donor_deltas))
    ax.bar(x, donor_deltas.values, color="tab:blue", alpha=0.7, label="Donneurs")

    ax.axhline(
        treated_delta,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Station traitée",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(donor_deltas.index, rotation=90)
    ax.set_ylabel("Δ NO₂ (post - pré) (µg/m³)")
    ax.set_title("Variation moyenne du NO₂ (post - pré) par station")
    ax.legend(loc="upper right")

    plt.tight_layout()
    return ax


def compute_preZFE_correlations(
    df: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    value_col: str = "no2_ug_m3",
) -> pd.Series:
    """
    Calcule la corrélation pré-ZFE entre la station traitée et chaque station.

    Retourne une Series indexée par station_id (donneurs) avec les corrélations.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    pre = df[df[date_col] < treatment_start]

    wide = (
        pre.pivot(index=date_col, columns=station_id_col, values=value_col)
        .sort_index()
    )

    if treated_id not in wide.columns:
        raise ValueError(f"Station traitée '{treated_id}' absente des données pivotées.")

    treated_series = wide[treated_id]
    corrs = wide.corrwith(treated_series).drop(labels=[treated_id])
    corrs = corrs.dropna().sort_values(ascending=False)

    return corrs


def plot_preZFE_correlations(
    corr_series: pd.Series,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Barplot des corrélations pré-ZFE entre la station traitée et chaque donneur.

    Paramètres
    ----------
    corr_series : Series
        Series indexée par station_id, typiquement la sortie de
        compute_preZFE_correlations (triée décroissante).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(corr_series))
    ax.bar(x, corr_series.values, color="tab:green", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(corr_series.index, rotation=90)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Corrélation pré-ZFE")
    ax.set_title("Corrélations pré-ZFE entre la station traitée et les donneurs")

    plt.tight_layout()
    return ax
