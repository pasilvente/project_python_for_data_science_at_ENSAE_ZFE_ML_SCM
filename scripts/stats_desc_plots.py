"""
stats_desc_plots.py
===================

Toutes les figures générées dans le notebook stat_desc.ipynb sont ré-encapsulées ici
(aucune perte de figures / aucune perte de sauvegardes).

Philosophie:
- Le notebook rapport_zfe ne contient QUE la narration (markdown) + un appel simple
- Toutes les figures sont définies dans ce module, avec les mêmes noms de fichiers .png

Entrées minimales:
- donors_daily, grenoble_daily, paris_daily : DataFrames journaliers (format projet)
- data_dir : Path vers dossier data (pour aires.geojson, etc.)
- out_dir : Path vers dossier figures (où écrire les .png)
- grenoble_zfe_start, paris_zfe_start : dates (str ou datetime) pour lignes verticales

Dépendances (selon figures):
- matplotlib, pandas, numpy
- geopandas, shapely (cartes ZFE)
- contextily (fond de carte pour cartes zoomées)  [optionnel mais requis pour reproduire]
- geopy (distances géodésiques)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tables (calcul)
from .stats_desc_core import (
    build_common_daily,
    build_stations_meta,
    compute_station_year_metrics,
    summarize_group_year_from_station_metrics,
    compute_oms_station_year_metrics,
    summarize_oms_group_year,
)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)

def savefig(fig: plt.Figure, out_dir: Path, filename: str, dpi: int = 300) -> Path:
    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path

def _to_datetime(x):
    if x is None:
        return None
    return pd.to_datetime(x)

def add_zfe_vlines(ax: plt.Axes, zfe_dates: Dict[str, pd.Timestamp], colors: Optional[Dict[str, str]] = None) -> None:
    for label, dt in zfe_dates.items():
        if dt is None or pd.isna(dt):
            continue
        ax.axvline(dt, linestyle="--", linewidth=2, alpha=0.6, color=(colors or {}).get(label, "gray"))

def _covid_mask(df: pd.DataFrame) -> pd.Series:
    # même fenêtre que notebook: 2020-03-01 → 2021-06-01
    return (df["date"] >= "2020-03-01") & (df["date"] <= "2021-06-01")

# -----------------------------------------------------------------------------
# 1) Carte ZFE + stations (stats_desc_carte_zfe.png)
# -----------------------------------------------------------------------------

def plot_zfe_and_stations_map(
    data_dir: Path,
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "stats_desc_carte_zfe.png",
) -> Path:
    """
    Reproduction de la figure 'stats_desc_carte_zfe.png' du notebook.
    """
    import json
    import geopandas as gpd
    from shapely.geometry import shape

    data_dir = _as_path(data_dir)

    # Charger les périmètres ZFE
    with open(data_dir / "aires.geojson", encoding="utf-8") as f:
        zfe_geojson = json.load(f)

    zfe_features = []
    for feat in zfe_geojson["features"]:
        pub = feat.get("publisher", {})
        zfe_id = pub.get("zfe_id")
        if zfe_id in ["GRENOBLE", "PARIS"]:
            zfe_features.append(
                {"zfe_id": zfe_id, "nom": pub.get("nom"), "geometry": shape(feat["geometry"])}
            )

    gdf_zfe = gpd.GeoDataFrame(zfe_features, crs="EPSG:4326")

    # Stations traitées (on récupère la méta depuis les daily)
    def _meta(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["station_id", "station_name", "lat", "lon"] if c in df.columns]
        m = df[cols].dropna().drop_duplicates("station_id").copy()
        return m

    st = pd.concat([_meta(grenoble_daily), _meta(paris_daily)], ignore_index=True)
    # ville = Grenoble / Paris pour filtrer comme notebook
    # si absent, on infère via le jeu d'entrée (grenoble_daily vs paris_daily)
    # -> pas une heuristique "territoriale" au sens donneurs; ici c'est le groupe traité.
    st_g = _meta(grenoble_daily).assign(ville="Grenoble")
    st_p = _meta(paris_daily).assign(ville="Paris")
    st = pd.concat([st_g, st_p], ignore_index=True)

    gdf_stations = gpd.GeoDataFrame(
        st,
        geometry=gpd.points_from_xy(st["lon"], st["lat"]),
        crs="EPSG:4326",
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for idx, (ville, ax) in enumerate(zip(["GRENOBLE", "PARIS"], axes)):
        zfe_subset = gdf_zfe[gdf_zfe["zfe_id"] == ville]
        zfe_subset.plot(ax=ax, color="red", alpha=0.2, edgecolor="red", linewidth=2, label="Périmètre ZFE")

        stations_subset = gdf_stations[gdf_stations["ville"] == ville.capitalize()]
        stations_subset.plot(ax=ax, color="blue", markersize=200, alpha=0.7, edgecolor="darkblue", linewidth=1.5, label="Stations de mesure")

        for _, station in stations_subset.iterrows():
            ax.annotate(
                station.get("station_name", station["station_id"]),
                xy=(station.geometry.x, station.geometry.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(f"ZFE {ville.capitalize()} : Périmètre & Stations", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 2) Qualité - complétude (qualite_donnees_completude_pct_jours.png)
# -----------------------------------------------------------------------------

def plot_completeness(
    common_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "qualite_donnees_completude_pct_jours.png",
    colors: Optional[Dict[str, str]] = None,
    zfe_dates: Optional[Dict[str, pd.Timestamp]] = None,
) -> Path:
    df = common_daily.copy()
    df["year"] = df["date"].dt.year
    obs = (
        df.groupby(["group", "station_id", "year"])["no2_ug_m3"]
        .apply(lambda s: s.notna().mean() * 100)
        .reset_index(name="pct_days_observed")
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    for grp in obs["group"].unique():
        d = obs[obs["group"] == grp].groupby("year")["pct_days_observed"].mean().reset_index()
        ax.plot(d["year"], d["pct_days_observed"], marker="o", linewidth=2, label=grp, color=(colors or {}).get(grp))

    if zfe_dates:
        # ici, x est year: on trace sur année (comme notebook) => vlines sur year
        for label, dt in zfe_dates.items():
            if dt is None or pd.isna(dt):
                continue
            ax.axvline(dt.year, linestyle="--", linewidth=2, alpha=0.5, color=(colors or {}).get(label, "gray"))

    ax.set_title("Qualité des données — Complétude temporelle moyenne\n(% de jours observés par station et par année)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Année", fontsize=12, fontweight="bold")
    ax.set_ylabel("% jours observés", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)

    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 3) Donneurs : répartition types (stats_donneurs_repartition_types.png)
# -----------------------------------------------------------------------------

def plot_donors_station_types(
    donors_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "stats_donneurs_repartition_types.png",
) -> Path:
    """
    Reproduit la figure donneurs (répartition station_env / station_influence).
    """
    if not {"station_id", "station_env", "station_influence"}.issubset(donors_daily.columns):
        raise ValueError("donors_daily doit contenir station_id, station_env, station_influence pour cette figure.")
    meta = donors_daily[["station_id", "station_env", "station_influence"]].drop_duplicates("station_id")
    env_counts = meta["station_env"].value_counts(dropna=False)
    infl_counts = meta["station_influence"].value_counts(dropna=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(env_counts.index.astype(str), env_counts.values)
    axes[0].set_title("Stations donneuses — Répartition 'station_env'", fontweight="bold")
    axes[0].set_ylabel("Nombre de stations")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(infl_counts.index.astype(str), infl_counts.values)
    axes[1].set_title("Stations donneuses — Répartition 'station_influence'", fontweight="bold")
    axes[1].set_ylabel("Nombre de stations")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 4) Cartes zoomées (carte_grenoble_zoom.png / carte_paris_zoom.png)
# -----------------------------------------------------------------------------

def plot_zoom_maps_paris_grenoble(
    data_dir: Path,
    donors_daily: pd.DataFrame,
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    out_dir: Path,
) -> List[Path]:
    """
    Reproduit les cartes zoomées avec fond Contextily.
    Sauvegarde:
    - carte_grenoble_zoom.png
    - carte_paris_zoom.png
    """
    import geopandas as gpd

    try:
        import contextily as ctx
    except Exception as e:
        raise ImportError("La figure 'cartes zoomées' requiert contextily. pip install contextily") from e

    CRS_WGS84 = "EPSG:4326"
    CRS_WEB = "EPSG:3857"

    # donors_meta / treated_meta (1 ligne/station)
    def _meta(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["station_id", "station_name", "lat", "lon", "station_env", "station_influence"] if c in df.columns]
        return df[cols].dropna(subset=["lat", "lon"]).drop_duplicates("station_id").copy()

    donors_meta = _meta(donors_daily)
    grenoble_meta = _meta(grenoble_daily).assign(ville="Grenoble")
    paris_meta = _meta(paris_daily).assign(ville="Paris")

    treated_meta = pd.concat([grenoble_meta, paris_meta], ignore_index=True)

    gdf_donors = gpd.GeoDataFrame(donors_meta, geometry=gpd.points_from_xy(donors_meta.lon, donors_meta.lat), crs=CRS_WGS84).to_crs(CRS_WEB)
    gdf_treated = gpd.GeoDataFrame(treated_meta, geometry=gpd.points_from_xy(treated_meta.lon, treated_meta.lat), crs=CRS_WGS84).to_crs(CRS_WEB)

    out_paths: List[Path] = []

    for ville in ["Grenoble", "Paris"]:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Points donneurs (gris) + traitées (rouge)
        gdf_donors.plot(ax=ax, markersize=15, alpha=0.6, label="Donneuses")
        gdf_treated[gdf_treated["ville"] == ville].plot(ax=ax, markersize=50, alpha=0.9, label=f"Traitée ({ville})")

        # Ajuster emprise: ville traitée + donneurs
        bbox = gdf_treated[gdf_treated["ville"] == ville].total_bounds
        pad = 20_000  # en mètres (EPSG:3857)
        ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
        ax.set_ylim(bbox[1] - pad, bbox[3] + pad)

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

        ax.set_axis_off()
        ax.set_title(f"Stations — Zoom {ville}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower left")

        fig.tight_layout()
        out_paths.append(savefig(fig, out_dir, f"carte_{ville.lower()}_zoom.png"))

    return out_paths

# -----------------------------------------------------------------------------
# 5) Donneurs vs traitées : comparaison (stats_donneurs_comparaison_traitees.png)
# -----------------------------------------------------------------------------

def plot_donors_vs_treated_summary(
    donors_daily: pd.DataFrame,
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "stats_donneurs_comparaison_traitees.png",
) -> Path:
    """
    Reproduit la figure qui compare distributions/ordres de grandeur donneurs vs traitées.
    (Transposition fidèle: on compare la moyenne annuelle par station.)
    """
    def station_mean(df):
        return df.groupby("station_id")["no2_ug_m3"].mean()

    s_d = station_mean(donors_daily)
    s_g = station_mean(grenoble_daily)
    s_p = station_mean(paris_daily)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([s_d.dropna().values, s_g.dropna().values, s_p.dropna().values], labels=["Donneuses", "Grenoble", "Paris"])
    ax.set_title("Comparaison NO₂ — Donneuses vs Traitées (moyenne par station)", fontsize=14, fontweight="bold")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 6) Saisonnalité Hiver/Été (2 figures) (no2_saisonnalite_*.png)
# -----------------------------------------------------------------------------

def plot_winter_summer_figures(
    common_daily: pd.DataFrame,
    out_dir: Path,
    filename_mean: str = "no2_saisonnalite_mean_hiver_ete.png",
    filename_pct25: str = "no2_saisonnalite_oms_pctjours25_hiver_ete.png",
    colors: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    df = common_daily.copy()
    df["month"] = df["date"].dt.month
    df["season2"] = np.where(df["month"].isin([12, 1, 2]), "Hiver", np.where(df["month"].isin([6, 7, 8]), "Été", "Autre"))
    df = df[df["season2"].isin(["Hiver", "Été"])]

    # A) moyenne
    mean_season = df.groupby(["group", "season2"])["no2_ug_m3"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    for grp in mean_season["group"].unique():
        d = mean_season[mean_season["group"] == grp].set_index("season2").loc[["Hiver", "Été"]].reset_index()
        ax.plot(d["season2"], d["no2_ug_m3"], marker="o", linewidth=2, label=grp, color=(colors or {}).get(grp))
    ax.set_title("NO₂ — Saisonnalité (moyenne) : Hiver vs Été", fontsize=14, fontweight="bold")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    p1 = savefig(fig, out_dir, filename_mean)

    # B) % jours > 25
    pct_25 = df.groupby(["group", "season2"])["no2_ug_m3"].apply(lambda s: (s > 25).mean() * 100).reset_index(name="pct_days_gt_25")
    fig, ax = plt.subplots(figsize=(12, 6))
    for grp in pct_25["group"].unique():
        d = pct_25[pct_25["group"] == grp].set_index("season2").loc[["Hiver", "Été"]].reset_index()
        ax.plot(d["season2"], d["pct_days_gt_25"], marker="o", linewidth=2, label=grp, color=(colors or {}).get(grp))
    ax.set_title("NO₂ — Saisonnalité : % jours > 25 µg/m³ (OMS) — Hiver vs Été", fontsize=14, fontweight="bold")
    ax.set_ylabel("% jours > 25")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    p2 = savefig(fig, out_dir, filename_pct25)

    return p1, p2

# -----------------------------------------------------------------------------
# 7) Évolution temporelle (stats_desc_evolution_temporelle.png)
# -----------------------------------------------------------------------------

def plot_time_evolution(
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "stats_desc_evolution_temporelle.png",
) -> Path:
    """
    Figure d'évolution temporelle: série mensuelle moyenne par groupe.
    """
    def monthly(df, label):
        d = df.copy()
        d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
        out = d.groupby("month")["no2_ug_m3"].mean().reset_index(name="no2")
        out["group"] = label
        return out

    m = pd.concat([
        monthly(donors_daily, "Donneuses"),
        monthly(grenoble_daily, "Grenoble"),
        monthly(paris_daily, "Paris"),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    for grp in m["group"].unique():
        d = m[m["group"] == grp].sort_values("month")
        ax.plot(d["month"], d["no2"], linewidth=2, label=grp)
    ax.set_title("NO₂ — Évolution temporelle (moyenne mensuelle)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 8) Comparaisons pré/post (stats_desc_comparaisons_pre_post.png)
# -----------------------------------------------------------------------------

def plot_pre_post_comparison(
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    grenoble_zfe_start,
    paris_zfe_start,
    out_dir: Path,
    filename: str = "stats_desc_comparaisons_pre_post.png",
) -> Path:
    """
    Comparaison pré vs post ZFE (boxplots) en excluant COVID.
    """
    g = grenoble_daily.copy()
    p = paris_daily.copy()
    g["date"] = pd.to_datetime(g["date"])
    p["date"] = pd.to_datetime(p["date"])
    g0 = _to_datetime(grenoble_zfe_start)
    p0 = _to_datetime(paris_zfe_start)

    def split(df, z0):
        df_clean = df[~_covid_mask(df)].copy()
        pre = df_clean[df_clean["date"] < z0]["no2_ug_m3"].dropna()
        post = df_clean[df_clean["date"] >= z0]["no2_ug_m3"].dropna()
        return pre, post

    g_pre, g_post = split(g, g0)
    p_pre, p_post = split(p, p0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([g_pre, g_post, p_pre, p_post], labels=["Grenoble pré", "Grenoble post", "Paris pré", "Paris post"])
    ax.set_title("NO₂ — Comparaison pré/post ZFE (COVID exclu)", fontsize=14, fontweight="bold")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 9) Distributions (stats_desc_distributions.png)
# -----------------------------------------------------------------------------

def plot_distributions_boxplots(
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    grenoble_zfe_start,
    paris_zfe_start,
    out_dir: Path,
    filename: str = "stats_desc_distributions.png",
) -> Path:
    """
    Reproduction fidèle de la cellule 'Distributions et boxplots' (2x2)
    (Grenoble/Paris x (hist + boxplot pré/post)).
    """
    g0 = _to_datetime(grenoble_zfe_start)
    p0 = _to_datetime(paris_zfe_start)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (df, zfe_date, ville) in enumerate([(grenoble_daily, g0, "Grenoble"), (paris_daily, p0, "Paris")]):
        covid_mask = _covid_mask(df)
        df_clean = df[~covid_mask].copy()
        df_clean["period"] = np.where(df_clean["date"] < zfe_date, "Pré", "Post")

        ax_hist = axes[idx, 0]
        ax_box = axes[idx, 1]

        ax_hist.hist(df_clean["no2_ug_m3"].dropna(), bins=40, alpha=0.8)
        ax_hist.set_title(f"{ville} — Distribution NO₂ (COVID exclu)", fontweight="bold")
        ax_hist.set_xlabel("NO₂ (µg/m³)")
        ax_hist.set_ylabel("Fréquence")
        ax_hist.grid(True, axis="y", alpha=0.2)

        pre = df_clean[df_clean["period"] == "Pré"]["no2_ug_m3"].dropna()
        post = df_clean[df_clean["period"] == "Post"]["no2_ug_m3"].dropna()
        ax_box.boxplot([pre, post], labels=["Pré", "Post"])
        ax_box.set_title(f"{ville} — Boxplot Pré/Post (COVID exclu)", fontweight="bold")
        ax_box.set_ylabel("NO₂ (µg/m³)")
        ax_box.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 10) Saisonnalité Grenoble focus (stats_desc_saisonnalite_grenoble.png)
# -----------------------------------------------------------------------------

def plot_grenoble_seasonality_focus(
    grenoble_daily: pd.DataFrame,
    grenoble_zfe_start,
    out_dir: Path,
    filename: str = "stats_desc_saisonnalite_grenoble.png",
) -> Path:
    """
    Reproduit la figure saisonnalité Grenoble (avant/après) telle que notebook.
    """
    g0 = _to_datetime(grenoble_zfe_start)
    df = grenoble_daily.copy()
    df = df[~_covid_mask(df)].copy()
    df["month"] = df["date"].dt.month
    df["period"] = np.where(df["date"] < g0, "Pré", "Post")
    df["month_name"] = df["date"].dt.strftime("%b")

    pivot = df.groupby(["month", "period"])["no2_ug_m3"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for period in ["Pré", "Post"]:
        d = pivot[pivot["period"] == period].sort_values("month")
        ax.plot(d["month"], d["no2_ug_m3"], marker="o", linewidth=2, label=period)
    ax.set_xticks(range(1, 13))
    ax.set_title("Grenoble — Profil saisonnier moyen (COVID exclu) : Pré vs Post ZFE", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mois")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 11) Profils mensuels (stats_desc_profils_mensuels.png)
# -----------------------------------------------------------------------------

def plot_monthly_profiles(
    donors_daily: pd.DataFrame,
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "stats_desc_profils_mensuels.png",
) -> Path:
    """
    Profil mensuel moyen (12 mois) par groupe.
    """
    def monthly_profile(df, label):
        d = df.copy()
        d["month"] = d["date"].dt.month
        out = d.groupby("month")["no2_ug_m3"].mean().reset_index(name="no2")
        out["group"] = label
        return out

    prof = pd.concat([
        monthly_profile(donors_daily, "Donneuses"),
        monthly_profile(grenoble_daily, "Grenoble"),
        monthly_profile(paris_daily, "Paris"),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for grp in prof["group"].unique():
        d = prof[prof["group"] == grp].sort_values("month")
        ax.plot(d["month"], d["no2"], marker="o", linewidth=2, label=grp)
    ax.set_xticks(range(1, 13))
    ax.set_title("NO₂ — Profil mensuel moyen (tous groupes)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mois")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 12) Comparaison donneurs (stats_desc_comparaison_donneurs.png)
# -----------------------------------------------------------------------------

def plot_donor_comparison(
    donors_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "stats_desc_comparaison_donneurs.png",
) -> Path:
    """
    Figure comparaison interne donneurs: distribution moyenne par station.
    """
    s = donors_daily.groupby("station_id")["no2_ug_m3"].mean().dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(s.values, bins=40, alpha=0.8)
    ax.set_title("Stations donneuses — Distribution des moyennes NO₂ par station", fontsize=14, fontweight="bold")
    ax.set_xlabel("NO₂ moyen (µg/m³)")
    ax.set_ylabel("Nombre de stations")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 13) Synthèses NO2 (no2_synth_1_*, no2_synth_2_*)
# -----------------------------------------------------------------------------

def plot_synth_ue_and_tail(
    common_daily: pd.DataFrame,
    out_dir: Path,
    zfe_dates: Optional[Dict[str, pd.Timestamp]] = None,
    colors: Optional[Dict[str, str]] = None,
    base_year: Optional[int] = None,
) -> Tuple[Path, Path]:
    sy = compute_station_year_metrics(common_daily)
    gy = summarize_group_year_from_station_metrics(sy)

    if base_year is None:
        base_year = int(common_daily["date"].dt.year.min())

    # Figure 1
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True, axis="y", alpha=0.25)
    for grp in gy["group"].unique():
        d = gy[gy["group"] == grp].sort_values("year")
        ax.plot(d["year"], d["share_stations_above_40"], marker="o", linewidth=2, label=f"{grp} — Stations >40", color=(colors or {}).get(grp))
        ax.plot(d["year"], d["mean_pct_days_gt_30"], marker="s", linewidth=2, linestyle="--", label=f"{grp} — % jours >30", color=(colors or {}).get(grp))
        ax.plot(d["year"], d["mean_pct_days_gt_40"], marker="^", linewidth=2, linestyle=":", label=f"{grp} — % jours >40", color=(colors or {}).get(grp))
    if zfe_dates:
        for label, dt in zfe_dates.items():
            if dt is None: 
                continue
            ax.axvline(dt.year, linestyle="--", linewidth=2, alpha=0.5, color=(colors or {}).get(label, "gray"))
    ax.set_title("NO₂ — Conformité UE & dépassements journaliers", fontsize=15, fontweight="bold")
    ax.set_xlabel("Année"); ax.set_ylabel("Pourcentages (%)")
    ax.legend(frameon=True, fontsize=10, ncol=1)
    fig.tight_layout()
    p1 = savefig(fig, out_dir, "no2_synth_1_conformite_depassements.png")

    # Figure 2 (base 100)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True, axis="y", alpha=0.25)
    for grp in gy["group"].unique():
        d = gy[gy["group"] == grp].sort_values("year")
        x = d["year"].astype(int).to_numpy()
        for col, lab, ls, mk in [
            ("mean_p90", "p90", "-", "o"),
            ("mean_p95", "p95", "--", "s"),
            ("mean_top10", "Top 10% mean", ":", "^"),
        ]:
            y = d[col].astype(float).to_numpy()
            base_val = y[x == base_year][0] if np.any(x == base_year) else np.nan
            if np.isfinite(base_val) and base_val != 0:
                y = (y / base_val) * 100.0
            ax.plot(x, y, label=f"{grp} — {lab}", linestyle=ls, marker=mk, linewidth=2, color=(colors or {}).get(grp))
    if zfe_dates:
        for label, dt in zfe_dates.items():
            if dt is None: 
                continue
            ax.axvline(dt.year, linestyle="--", linewidth=2, alpha=0.5, color=(colors or {}).get(label, "gray"))
    ax.set_title(f"NO₂ — Queue haute & pics (base 100 = {base_year})", fontsize=15, fontweight="bold")
    ax.set_xlabel("Année"); ax.set_ylabel("Indice (base 100)")
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    p2 = savefig(fig, out_dir, "no2_synth_2_queue_haute_pics_norm_base100.png")

    return p1, p2

# -----------------------------------------------------------------------------
# 14) OMS (no2_oms_only_sanitaire.png, no2_oms_vs_ue_decharge.png)
# -----------------------------------------------------------------------------

def plot_oms_figures(
    common_daily: pd.DataFrame,
    out_dir: Path,
    zfe_dates: Optional[Dict[str, pd.Timestamp]] = None,
    colors: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    oms_sy = compute_oms_station_year_metrics(common_daily)
    oms = summarize_oms_group_year(oms_sy)

    # Figure A
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True, axis="y", alpha=0.25)
    for grp in oms["group"].unique():
        d = oms[oms["group"] == grp].sort_values("year")
        ax.plot(d["year"], d["share_stations_above_oms_10"], marker="o", linewidth=2,
                label=f"{grp} — % stations >10", color=(colors or {}).get(grp))
        ax.plot(d["year"], d["mean_pct_days_gt_25"], marker="s", linewidth=2, linestyle="--",
                label=f"{grp} — % jours >25", color=(colors or {}).get(grp))
    if zfe_dates:
        for label, dt in zfe_dates.items():
            if dt is None: 
                continue
            ax.axvline(dt.year, linestyle="--", linewidth=2, alpha=0.5, color=(colors or {}).get(label, "gray"))
    ax.set_title("NO₂ — Indicateurs OMS 2021 (annuel + journalier)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Année"); ax.set_ylabel("Pourcentages (%)")
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    p1 = savefig(fig, out_dir, "no2_oms_only_sanitaire.png")

    # Figure B
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True, axis="y", alpha=0.25)
    for grp in oms["group"].unique():
        d = oms[oms["group"] == grp].sort_values("year")
        ax.plot(d["year"], d["mean_pct_days_gt_25"], marker="o", linewidth=2,
                label=f"{grp} — % jours >25 (OMS)", color=(colors or {}).get(grp))
        ax.plot(d["year"], d["mean_pct_days_gt_40"], marker="^", linewidth=2, linestyle="--",
                label=f"{grp} — % jours >40 (proxy UE)", color=(colors or {}).get(grp))
    if zfe_dates:
        for label, dt in zfe_dates.items():
            if dt is None: 
                continue
            ax.axvline(dt.year, linestyle="--", linewidth=2, alpha=0.5, color=(colors or {}).get(label, "gray"))
    ax.set_title("NO₂ — OMS (25) vs proxy UE (40) — dépassements journaliers", fontsize=15, fontweight="bold")
    ax.set_xlabel("Année"); ax.set_ylabel("Pourcentages (%)")
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    p2 = savefig(fig, out_dir, "no2_oms_vs_ue_decharge.png")

    return p1, p2

# -----------------------------------------------------------------------------
# 15) Trafic vs fond (OMS + pics) (no2_trafic_fond_*.png)
# -----------------------------------------------------------------------------

def plot_trafic_fond_figures(
    donors_daily: pd.DataFrame,
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """
    Reproduction des 2 figures:
    - no2_trafic_fond_oms_pctjours25.png
    - no2_trafic_fond_pics_charge_norm.png
    """
    # Assemble "common" rapide avec influence
    def prep(df, label):
        d = df.copy()
        d["group"] = label
        d["year"] = d["date"].dt.year
        return d
    df = pd.concat([prep(donors_daily, "Donneuses"), prep(grenoble_daily, "Grenoble"), prep(paris_daily, "Paris")], ignore_index=True)
    if "station_influence" not in df.columns:
        raise ValueError("station_influence requis pour les figures trafic/fond.")

    # A) % jours >25 par influence
    sy = (
        df.groupby(["group", "station_id", "year", "station_influence"])["no2_ug_m3"]
        .apply(lambda s: (s > 25).mean() * 100)
        .reset_index(name="pct_days_gt_25")
    )
    gy = sy.groupby(["group", "year", "station_influence"])["pct_days_gt_25"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True, axis="y", alpha=0.25)
    for grp in gy["group"].unique():
        d = gy[gy["group"] == grp]
        for infl in sorted(d["station_influence"].dropna().unique()):
            s = d[d["station_influence"] == infl].sort_values("year")
            ax.plot(s["year"], s["pct_days_gt_25"], marker="o", linewidth=2, label=f"{grp} — {infl}", alpha=0.8)
    ax.set_title("NO₂ — % jours > 25 µg/m³ (OMS) selon influence station (trafic/fond)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Année"); ax.set_ylabel("% jours > 25")
    ax.legend(frameon=True, fontsize=9, ncol=2)
    fig.tight_layout()
    p1 = savefig(fig, out_dir, "no2_trafic_fond_oms_pctjours25.png")

    # B) Pics/charge normalisés (utilise p95 + top10, base 100)
    sy2 = compute_station_year_metrics(df.rename(columns={"group":"group"}))  # expects group/station_id/year/no2_ug_m3 ok
    # merge influence
    infl = df[["group","station_id","station_influence"]].drop_duplicates()
    sy2 = sy2.merge(infl, on=["group","station_id"], how="left")
    gy2 = sy2.groupby(["group","year","station_influence"]).agg(
        mean_p95=("p95","mean"),
        mean_top10=("top10_mean","mean"),
    ).reset_index()
    base_year = int(gy2["year"].min())

    fig, ax = plt.subplots(figsize=(14,7))
    ax.grid(True, axis="y", alpha=0.25)
    for grp in gy2["group"].unique():
        d = gy2[gy2["group"]==grp]
        for infl_val in sorted(d["station_influence"].dropna().unique()):
            s = d[d["station_influence"]==infl_val].sort_values("year")
            x = s["year"].astype(int).to_numpy()
            y = s["mean_p95"].astype(float).to_numpy()
            base = y[x==base_year][0] if np.any(x==base_year) else np.nan
            if np.isfinite(base) and base!=0:
                y = (y/base)*100
            ax.plot(x, y, marker="o", linewidth=2, label=f"{grp} — {infl_val} — p95 (base100)")
    ax.set_title(f"NO₂ — Pics (p95) normalisés base 100 = {base_year} — trafic vs fond", fontsize=14, fontweight="bold")
    ax.set_xlabel("Année"); ax.set_ylabel("Indice (base 100)")
    ax.legend(frameon=True, fontsize=9, ncol=2)
    fig.tight_layout()
    p2 = savefig(fig, out_dir, "no2_trafic_fond_pics_charge_norm.png")

    return p1, p2

# -----------------------------------------------------------------------------
# 16) Donneurs - évolution temporelle (stats_donneurs_evolution_temporelle.png)
# -----------------------------------------------------------------------------

def plot_donors_time_evolution(
    donors_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "stats_donneurs_evolution_temporelle.png",
) -> Path:
    d = donors_daily.copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    m = d.groupby("month")["no2_ug_m3"].mean().reset_index(name="no2")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(m["month"], m["no2"], linewidth=2)
    ax.set_title("Stations donneuses — Évolution temporelle (moyenne mensuelle)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return savefig(fig, out_dir, filename)

# -----------------------------------------------------------------------------
# 17) Corrélations + scatters (4 figures)
# -----------------------------------------------------------------------------

def _corr_heatmap(ax: plt.Axes, c: pd.DataFrame, title: str) -> None:
    ax.imshow(c.values)
    ax.set_xticks(range(len(c.columns))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c.index))); ax.set_yticklabels(c.index)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")

def plot_corr_and_scatters(
    common_daily: pd.DataFrame,
    out_dir: Path,
) -> Tuple[Path, Path, Path, Path]:
    sy = compute_station_year_metrics(common_daily)
    oms_sy = compute_oms_station_year_metrics(common_daily)[["group","station_id","year","pct_days_gt_25"]]
    wide = sy.merge(oms_sy, on=["group","station_id","year"], how="left")

    X = wide[["mean_annual","p95","pct_days_gt_25"]].dropna()
    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(6,5))
    _corr_heatmap(ax, pearson, "Corrélations (pearson) — station-année")
    fig.tight_layout()
    p1 = savefig(fig, out_dir, "no2_corr_heatmap_pearson.png")

    fig, ax = plt.subplots(figsize=(6,5))
    _corr_heatmap(ax, spearman, "Corrélations (spearman) — station-année")
    fig.tight_layout()
    p2 = savefig(fig, out_dir, "no2_corr_heatmap_spearman.png")

    fig, ax = plt.subplots(figsize=(8,6))
    for grp in wide["group"].unique():
        d = wide[wide["group"]==grp].dropna(subset=["mean_annual","p95"])
        ax.scatter(d["mean_annual"], d["p95"], label=grp, alpha=0.7)
    ax.set_title("NO₂ — Mean annuel vs P95 (station-année)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Moyenne annuelle"); ax.set_ylabel("P95")
    ax.grid(True, alpha=0.25); ax.legend(frameon=True)
    fig.tight_layout()
    p3 = savefig(fig, out_dir, "no2_scatter_mean_vs_p95.png")

    fig, ax = plt.subplots(figsize=(8,6))
    for grp in wide["group"].unique():
        d = wide[wide["group"]==grp].dropna(subset=["mean_annual","pct_days_gt_25"])
        ax.scatter(d["mean_annual"], d["pct_days_gt_25"], label=grp, alpha=0.7)
    ax.set_title("NO₂ — Mean annuel vs % jours > 25 (OMS) (station-année)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Moyenne annuelle"); ax.set_ylabel("% jours > 25")
    ax.grid(True, alpha=0.25); ax.legend(frameon=True)
    fig.tight_layout()
    p4 = savefig(fig, out_dir, "no2_scatter_mean_vs_pctjours25.png")

    return p1, p2, p3, p4

# -----------------------------------------------------------------------------
# 18) Orchestrateur complet : génère TOUTES les figures du notebook
# -----------------------------------------------------------------------------

def generate_all_figures_from_notebook(
    donors_daily: pd.DataFrame,
    grenoble_daily: pd.DataFrame,
    paris_daily: pd.DataFrame,
    data_dir: Path,
    out_dir: Path,
    grenoble_zfe_start,
    paris_zfe_start,
    group_colors: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:
    """
    Génère toutes les figures 'stat_desc.ipynb' dans out_dir, avec les mêmes noms.

    Returns
    -------
    dict {name: path}
    """
    data_dir = _as_path(data_dir)
    out_dir = _as_path(out_dir)

    # normaliser dates
    for df in (donors_daily, grenoble_daily, paris_daily):
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"])

    zfe_dates = {
        "Grenoble": _to_datetime(grenoble_zfe_start),
        "Paris": _to_datetime(paris_zfe_start),
    }

    # common_daily (pour toutes figures agrégées)
    common = build_common_daily(grenoble_daily, paris_daily, donors_daily).common_daily

    saved: Dict[str, Path] = {}

    saved["stats_desc_carte_zfe"] = plot_zfe_and_stations_map(data_dir, grenoble_daily, paris_daily, out_dir)

    saved["qualite_donnees_completude_pct_jours"] = plot_completeness(common, out_dir, colors=group_colors, zfe_dates=zfe_dates)

    saved["stats_donneurs_repartition_types"] = plot_donors_station_types(donors_daily, out_dir)

    # cartes zoomées
    zoom_paths = plot_zoom_maps_paris_grenoble(data_dir, donors_daily, grenoble_daily, paris_daily, out_dir)
    for p in zoom_paths:
        saved[p.stem] = p

    saved["stats_donneurs_comparaison_traitees"] = plot_donors_vs_treated_summary(donors_daily, grenoble_daily, paris_daily, out_dir)

    p_mean, p_pct = plot_winter_summer_figures(common, out_dir, colors=group_colors)
    saved[p_mean.stem] = p_mean
    saved[p_pct.stem] = p_pct

    saved["stats_desc_evolution_temporelle"] = plot_time_evolution(grenoble_daily, paris_daily, donors_daily, out_dir)

    saved["stats_desc_comparaisons_pre_post"] = plot_pre_post_comparison(grenoble_daily, paris_daily, grenoble_zfe_start, paris_zfe_start, out_dir)

    saved["stats_desc_distributions"] = plot_distributions_boxplots(grenoble_daily, paris_daily, grenoble_zfe_start, paris_zfe_start, out_dir)

    saved["stats_desc_saisonnalite_grenoble"] = plot_grenoble_seasonality_focus(grenoble_daily, grenoble_zfe_start, out_dir)

    saved["stats_desc_profils_mensuels"] = plot_monthly_profiles(donors_daily, grenoble_daily, paris_daily, out_dir)

    saved["stats_desc_comparaison_donneurs"] = plot_donor_comparison(donors_daily, out_dir)

    p1, p2 = plot_synth_ue_and_tail(common, out_dir, zfe_dates=zfe_dates, colors=group_colors)
    saved[p1.stem] = p1
    saved[p2.stem] = p2

    q1, q2 = plot_oms_figures(common, out_dir, zfe_dates=zfe_dates, colors=group_colors)
    saved[q1.stem] = q1
    saved[q2.stem] = q2

    t1, t2 = plot_trafic_fond_figures(donors_daily, grenoble_daily, paris_daily, out_dir)
    saved[t1.stem] = t1
    saved[t2.stem] = t2

    saved["stats_donneurs_evolution_temporelle"] = plot_donors_time_evolution(donors_daily, out_dir)

    c1, c2, c3, c4 = plot_corr_and_scatters(common, out_dir)
    for p in (c1, c2, c3, c4):
        saved[p.stem] = p

    return saved
