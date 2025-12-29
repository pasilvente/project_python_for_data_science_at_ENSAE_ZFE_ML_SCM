from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import warnings

try:
    import geopandas as gpd
    import contextily as ctx
    from shapely.geometry import Point
except Exception:
    gpd = None
    ctx = None
    Point = None

warnings.filterwarnings("ignore")


def _ensure_out_dir(out_dir: Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _with_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    out = df.copy()
    out["group"] = group
    return out


def make_common_daily(
    donors: pd.DataFrame,
    gre: pd.DataFrame,
    par: pd.DataFrame,
    label_donors="Donneuses",
    label_gre="Grenoble",
    label_par="Paris",
) -> pd.DataFrame:
    all_df = pd.concat(
        [_with_group(donors, label_donors), _with_group(gre, label_gre), _with_group(par, label_par)],
        ignore_index=True,
        sort=False,
    )
    all_df["year"] = all_df["date"].dt.year
    all_df["month"] = all_df["date"].dt.to_period("M").dt.to_timestamp()
    all_df["season"] = all_df["date"].dt.month.map(
        lambda m: "Hiver" if m in (12, 1, 2) else "Printemps" if m in (3, 4, 5) else "Été" if m in (6, 7, 8) else "Automne"
    )
    return all_df


def fig_distribution_groups(
    common_daily: pd.DataFrame,
    out_dir: Path,
    filename: str = "stats_desc_distributions.png",
    sample_max: int = 30000,
) -> Path:
    out_dir = _ensure_out_dir(out_dir)
    df = common_daily[["group", "no2_ug_m3"]].dropna()
    if len(df) > sample_max:
        df = df.sample(sample_max, random_state=0)
    groups = list(df["group"].unique())
    data = [df.loc[df["group"] == g, "no2_ug_m3"].to_numpy() for g in groups]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=groups, showfliers=False)
    ax.set_title("Distribution du NO₂ (µg/m³) — donneuses vs traitées", fontweight="bold")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_time_evolution(
    common_daily: pd.DataFrame,
    out_dir: Path,
    *,
    grenoble_zfe_start,
    paris_zfe_start,
    filename: str = "stats_desc_evolution_temporelle.png",
) -> Path:
    out_dir = _ensure_out_dir(out_dir)
    series = common_daily.groupby(["group", "month"])["no2_ug_m3"].mean().reset_index(name="no2")

    fig, ax = plt.subplots(figsize=(13, 6))
    for grp in series["group"].unique():
        d = series[series["group"] == grp].sort_values("month")
        ax.plot(d["month"], d["no2"], linewidth=2, label=str(grp))

    ax.axvline(pd.to_datetime(grenoble_zfe_start), linestyle=":", linewidth=2, alpha=0.8)
    ax.axvline(pd.to_datetime(paris_zfe_start), linestyle=":", linewidth=2, alpha=0.8)

    ax.set_title("Évolution temporelle du NO₂ (moyenne mensuelle) — avec débuts ZFE", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_seasonality_hiver_ete(
    common_daily: pd.DataFrame,
    out_dir: Path,
    *,
    filename: str = "no2_saisonnalite_mean_hiver_ete.png",
    connect: bool = False,
) -> Path:
    out_dir = _ensure_out_dir(out_dir)
    seasonal = common_daily.groupby(["group", "season"])["no2_ug_m3"].mean().reset_index(name="no2")
    order = ["Hiver", "Printemps", "Été", "Automne"]
    seasonal["season"] = pd.Categorical(seasonal["season"], categories=order, ordered=True)
    seasonal = seasonal.sort_values(["group", "season"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for grp in seasonal["group"].unique():
        d = seasonal[seasonal["group"] == grp]
        if connect:
            ax.plot(d["season"], d["no2"], marker="o", linewidth=2, label=str(grp))
        else:
            ax.scatter(d["season"], d["no2"], s=60, label=str(grp))

    ax.set_title("Saisonnalité du NO₂ — moyenne par saison", fontweight="bold")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_trafic_fond_monthly(
    common_daily: pd.DataFrame,
    out_dir: Path,
    *,
    filename: str = "no2_trafic_fond_pics_charge_norm.png",
) -> Optional[Path]:
    out_dir = _ensure_out_dir(out_dir)
    if "station_influence" not in common_daily.columns:
        return None

    df = common_daily.dropna(subset=["station_influence"]).copy()
    df["influence"] = df["station_influence"].astype(str)
    series = df.groupby(["group", "influence", "month"])["no2_ug_m3"].mean().reset_index(name="no2")

    fig, ax = plt.subplots(figsize=(13, 6))
    for (grp, infl) in series[["group", "influence"]].drop_duplicates().itertuples(index=False):
        d = series[(series["group"] == grp) & (series["influence"] == infl)].sort_values("month")
        ax.plot(d["month"], d["no2"], linewidth=2, label=f"{grp} — {infl}")

    ax.set_title("NO₂ moyen mensuel — trafic vs fond (si disponible)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_prepost_completeness(before: pd.DataFrame, after: pd.DataFrame, out_dir: Path,
                             filename: str = "prepost_1_completeness.png") -> Path:
    out_dir = _ensure_out_dir(out_dir)

    def compl(df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        tmp["obs"] = tmp["no2_ug_m3"].notna().astype(int)
        st_year = tmp.groupby(["group", "station_id", "year"])["obs"].mean().reset_index(name="pct_obs")
        grp_year = st_year.groupby(["group", "year"])["pct_obs"].mean().reset_index()
        grp_year["pct_obs"] *= 100
        return grp_year

    c = pd.concat([compl(before).assign(stage="Avant"), compl(after).assign(stage="Après")], ignore_index=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    for grp in c["group"].unique():
        for stage, ls in [("Avant", "--"), ("Après", "-")]:
            d = c[(c["group"] == grp) & (c["stage"] == stage)].sort_values("year")
            ax.plot(d["year"], d["pct_obs"], linestyle=ls, marker="o", linewidth=2, label=f"{grp} — {stage}")

    ax.set_title("Avant / Après — Complétude moyenne (% jours observés)", fontweight="bold")
    ax.set_xlabel("Année")
    ax.set_ylabel("% jours observés")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_prepost_monthly(before: pd.DataFrame, after: pd.DataFrame, out_dir: Path, *,
                        grenoble_zfe_start, paris_zfe_start,
                        filename: str = "prepost_2_evolution_monthly.png") -> Path:
    out_dir = _ensure_out_dir(out_dir)

    def monthly(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["group", "month"])["no2_ug_m3"].mean().reset_index(name="no2")

    m = pd.concat([monthly(before).assign(stage="Avant"), monthly(after).assign(stage="Après")], ignore_index=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    for grp in m["group"].unique():
        for stage, ls in [("Avant", "--"), ("Après", "-")]:
            d = m[(m["group"] == grp) & (m["stage"] == stage)].sort_values("month")
            ax.plot(d["month"], d["no2"], linestyle=ls, linewidth=2, label=f"{grp} — {stage}")

    ax.axvline(pd.to_datetime(grenoble_zfe_start), linestyle=":", linewidth=2, alpha=0.8)
    ax.axvline(pd.to_datetime(paris_zfe_start), linestyle=":", linewidth=2, alpha=0.8)

    ax.set_title("Avant / Après — NO₂ moyen mensuel (avec débuts ZFE)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_prepost_distributions(before: pd.DataFrame, after: pd.DataFrame, out_dir: Path,
                              filename: str = "prepost_3_distributions.png",
                              sample_max: int = 30000) -> Path:
    out_dir = _ensure_out_dir(out_dir)

    def sample(df: pd.DataFrame) -> pd.DataFrame:
        d = df[["group", "no2_ug_m3"]].dropna()
        if len(d) > sample_max:
            d = d.sample(sample_max, random_state=0)
        return d

    b = sample(before).assign(stage="Avant")
    a = sample(after).assign(stage="Après")
    ba = pd.concat([b, a], ignore_index=True)

    groups = list(ba["group"].unique())
    fig, ax = plt.subplots(figsize=(13, 6))
    data = []
    labels = []
    for g in groups:
        for stage in ["Avant", "Après"]:
            vals = ba[(ba["group"] == g) & (ba["stage"] == stage)]["no2_ug_m3"].to_numpy()
            data.append(vals)
            labels.append(f"{g}\n{stage}")

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Avant / Après — Distribution NO₂ (boxplots)", fontweight="bold")
    ax.set_ylabel("NO₂ (µg/m³)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


@dataclass(frozen=True)
class FigurePaths:
    paths: Dict[str, Path]


def _station_points_gdf(common_daily: pd.DataFrame):
    """Stations -> GeoDataFrame (EPSG:4326) si lat/lon dispo."""
    if gpd is None or Point is None:
        return None

    needed = {"station_id", "lat", "lon"}
    if not needed.issubset(common_daily.columns):
        return None

    st = (
        common_daily.dropna(subset=["lat", "lon"])
        .drop_duplicates(subset=["station_id"])
        .copy()
    )
    st["geometry"] = [Point(xy) for xy in zip(st["lon"], st["lat"])]
    return gpd.GeoDataFrame(st, geometry="geometry", crs="EPSG:4326")


def _load_zfe_aires_gdf(data_dir: Path):
    """
    Charge une couche ZFE (aires) si possible.
    On tente plusieurs formats. Si rien n'est trouvé, retourne None.
    """
    if gpd is None:
        return None

    data_dir = Path(data_dir)

    # priorité aux fichiers géo
    candidates = [
        data_dir / "aires.geojson",
        data_dir / "aires_clean.geojson",
        data_dir / "aires_flat.geojson",
        data_dir / "aires.gpkg",
    ]
    for p in candidates:
        if p.exists():
            try:
                g = gpd.read_file(p)
                # s'assurer d'un CRS
                if g.crs is None:
                    g = g.set_crs("EPSG:4326")
                return g
            except Exception:
                pass

    # fallback : si tu as une colonne WKT dans aires_clean.csv
    csv_candidates = [data_dir / "aires_clean.csv", data_dir / "aires_flat.csv"]
    for p in csv_candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                # cherche une colonne geometry / geom / wkt
                wkt_col = None
                for col in ["geometry", "geom", "wkt", "wkt_geometry"]:
                    if col in df.columns:
                        wkt_col = col
                        break
                if wkt_col is None:
                    continue
                g = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df[wkt_col]), crs="EPSG:4326")
                return g
            except Exception:
                pass

    return None


def fig_map_zfe_and_stations(
    common_daily: pd.DataFrame,
    out_dir: Path,
    *,
    data_dir: Path,
    filename: str = "stats_desc_carte_zfe.png",
    tile_provider=None,
) -> Optional[Path]:
    """
    Carte nationale/régionale: ZFE (si dispo) + stations (donneuses + traitées) sur fond OSM.
    """
    out_dir = _ensure_out_dir(out_dir)
    if gpd is None or ctx is None:
        return None

    stations = _station_points_gdf(common_daily)
    if stations is None:
        return None

    aires = _load_zfe_aires_gdf(data_dir)

    # Proj Web Mercator pour fond de carte
    stations_3857 = stations.to_crs(epsg=3857)
    aires_3857 = aires.to_crs(epsg=3857) if aires is not None else None

    fig, ax = plt.subplots(figsize=(12, 10))

    if aires_3857 is not None and len(aires_3857) > 0:
        aires_3857.plot(ax=ax, alpha=0.10, edgecolor="black", linewidth=0.6)

    stations_3857.plot(ax=ax, markersize=18, alpha=0.85)

    # Emprise = ZFE + stations (buffer)
    xmin, ymin, xmax, ymax = stations_3857.total_bounds
    if aires_3857 is not None and len(aires_3857) > 0:
        axmin, aymin, axmax, aymax = aires_3857.total_bounds
        xmin, ymin, xmax, ymax = min(xmin, axmin), min(ymin, aymin), max(xmax, axmax), max(ymax, aymax)

    pad_x = (xmax - xmin) * 0.08
    pad_y = (ymax - ymin) * 0.08
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_title("ZFE (si disponible) et stations NO₂ (donneuses + traitées)", fontweight="bold")
    ax.axis("off")

    # Fond de carte
    try:
        provider = tile_provider or ctx.providers.OpenStreetMap.Mapnik
        ctx.add_basemap(ax, source=provider)
    except Exception:
        # Si les tuiles échouent (réseau), on garde la carte sans fond mais on sauvegarde quand même
        pass

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_city_zoom_with_donors_extent(
    common_daily: pd.DataFrame,
    out_dir: Path,
    *,
    city_label: str,
    filename: str,
    tile_provider=None,
    min_buffer_km: float = 10.0,
) -> Optional[Path]:
    """
    Zoom sur une ville (Grenoble/Paris) MAIS emprise calculée pour inclure aussi les donneuses
    proches, pour éviter une carte trop zoomée où on perd les points.
    """
    out_dir = _ensure_out_dir(out_dir)
    if gpd is None or ctx is None:
        return None

    stations = _station_points_gdf(common_daily)
    if stations is None:
        return None

    # sous-ensemble "ville" = stations dont group == city_label
    if "group" not in stations.columns:
        return None

    st_3857 = stations.to_crs(epsg=3857)
    city_pts = st_3857[st_3857["group"] == city_label].copy()

    if len(city_pts) == 0:
        return None

    # centre = bbox ville
    xmin, ymin, xmax, ymax = city_pts.total_bounds
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    # distance des stations à ce centre -> prendre les donneuses proches
    st_3857["dist2"] = (st_3857.geometry.x - cx) ** 2 + (st_3857.geometry.y - cy) ** 2
    # garde les 80 plus proches pour avoir contexte donneurs
    near = st_3857.nsmallest(80, "dist2").copy()

    # emprise = bbox ville + near, avec buffer minimal
    xmin2, ymin2, xmax2, ymax2 = near.total_bounds
    xmin, ymin, xmax, ymax = min(xmin, xmin2), min(ymin, ymin2), max(xmax, xmax2), max(ymax, ymax2)

    min_buf = float(min_buffer_km) * 1000.0
    pad_x = max((xmax - xmin) * 0.12, min_buf)
    pad_y = max((ymax - ymin) * 0.12, min_buf)

    fig, ax = plt.subplots(figsize=(12, 10))
    near.plot(ax=ax, markersize=22, alpha=0.85)

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_title(f"Stations NO₂ — zoom {city_label} (emprise incluant donneuses proches)", fontweight="bold")
    ax.axis("off")

    try:
        provider = tile_provider or ctx.providers.OpenStreetMap.Mapnik
        ctx.add_basemap(ax, source=provider)
    except Exception:
        pass

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_selected_figures_for_report(
    *,
    before_inputs,
    after_inputs,
    out_dir_raw: Path,
    out_dir_prepost: Path,
    grenoble_zfe_start,
    paris_zfe_start,
    connect_seasonality: bool = False,data_dir: Optional[Path] = None,
) -> FigurePaths:
    out: Dict[str, Path] = {}

    b = make_common_daily(
        before_inputs.donors_daily, before_inputs.grenoble_daily, before_inputs.paris_daily,
        before_inputs.label_donors, before_inputs.label_grenoble, before_inputs.label_paris
    )
    a = make_common_daily(
        after_inputs.donors_daily, after_inputs.grenoble_daily, after_inputs.paris_daily,
        after_inputs.label_donors, after_inputs.label_grenoble, after_inputs.label_paris
    )

    out["dist_groups_before"] = fig_distribution_groups(b, out_dir_raw)
    out["evolution_before"] = fig_time_evolution(
        b, out_dir_raw,
        grenoble_zfe_start=grenoble_zfe_start,
        paris_zfe_start=paris_zfe_start,
    )
    out["seasonality_before"] = fig_seasonality_hiver_ete(b, out_dir_raw, connect=connect_seasonality)

    tf = fig_trafic_fond_monthly(b, out_dir_raw)
    if tf is not None:
        out["trafic_fond_before"] = tf

    out["prepost_completeness"] = fig_prepost_completeness(b, a, out_dir_prepost)
    out["prepost_monthly"] = fig_prepost_monthly(
        b, a, out_dir_prepost,
        grenoble_zfe_start=grenoble_zfe_start,
        paris_zfe_start=paris_zfe_start,
    )
    out["prepost_distributions"] = fig_prepost_distributions(b, a, out_dir_prepost)

        # --- Cartes avec fond (optionnelles) ---
    if data_dir is not None:
        m1 = fig_map_zfe_and_stations(b, out_dir_raw, data_dir=Path(data_dir), filename="stats_desc_carte_zfe.png")
        if m1 is not None:
            out["map_zfe_stations"] = m1

        mg = fig_city_zoom_with_donors_extent(
            b, out_dir_raw, city_label=before_inputs.label_grenoble, filename="carte_grenoble_zoom.png"
        )
        if mg is not None:
            out["map_grenoble_zoom"] = mg

        mp = fig_city_zoom_with_donors_extent(
            b, out_dir_raw, city_label=before_inputs.label_paris, filename="carte_paris_zoom.png"
        )
        if mp is not None:
            out["map_paris_zoom"] = mp


    return FigurePaths(paths=out)
