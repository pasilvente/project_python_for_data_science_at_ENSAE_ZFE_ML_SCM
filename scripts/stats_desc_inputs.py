"""
stats_desc_inputs.py
====================

Définit des conventions d'entrée *explicites* pour les statistiques descriptives NO2,
et fournit des helpers pour construire des datasets "avant" / "après" de manière contrôlée.

Objectif: éviter toute ambiguïté sur "quelles données alimentent quelles figures".

Conventions minimales (colonnes):
- date: datetime-like
- station_id: str/int
- no2_ug_m3: float

Colonnes utiles si disponibles:
- station_name, lat, lon
- station_env, station_influence   (trafic / fond)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


REQUIRED = ("date", "station_id", "no2_ug_m3")


def _ensure_cols(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] colonnes manquantes: {missing}. Colonnes dispo: {list(df.columns)}")


def normalize_daily(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalise types + supprime lignes sans (date, station_id, no2)."""
    _ensure_cols(df, name)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["station_id"] = out["station_id"].astype(str)
    out["no2_ug_m3"] = pd.to_numeric(out["no2_ug_m3"], errors="coerce")
    out = out.dropna(subset=["date", "station_id", "no2_ug_m3"]).sort_values(["station_id", "date"])
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class StatsDescInputs:
    """Conteneur explicite des 3 entrées nécessaires."""

    donors_daily: pd.DataFrame
    grenoble_daily: pd.DataFrame
    paris_daily: pd.DataFrame
    label_donors: str = "Donneuses"
    label_grenoble: str = "Grenoble"
    label_paris: str = "Paris"

    def normalized(self) -> "StatsDescInputs":
        return StatsDescInputs(
            donors_daily=normalize_daily(self.donors_daily, "donors_daily"),
            grenoble_daily=normalize_daily(self.grenoble_daily, "grenoble_daily"),
            paris_daily=normalize_daily(self.paris_daily, "paris_daily"),
            label_donors=self.label_donors,
            label_grenoble=self.label_grenoble,
            label_paris=self.label_paris,
        )


def load_before_after_from_data_dir(
    data_dir: Path,
    *,
    before_kind: str = "clean",
    after_kind: str = "imputed",
) -> Tuple[StatsDescInputs, StatsDescInputs]:
    """Charge un couple (AVANT, APRÈS) à partir des fichiers du dossier data."""
    data_dir = Path(data_dir)

    def donors_path(city: str, kind: str) -> Path:
        return data_dir / f"no2_donors_{city}_daily_{kind}.csv"

    before = StatsDescInputs(
        donors_daily=pd.concat(
            [
                pd.read_csv(donors_path("grenoble", before_kind)),
                pd.read_csv(donors_path("paris", before_kind)),
            ],
            ignore_index=True,
        ),
        grenoble_daily=pd.read_csv(data_dir / "pollution_grenoble_no2_daily_clean.csv"),
        paris_daily=pd.read_csv(data_dir / "pollution_paris_no2_daily_clean.csv"),
    ).normalized()

    after = StatsDescInputs(
        donors_daily=pd.concat(
            [
                pd.read_csv(donors_path("grenoble", after_kind)),
                pd.read_csv(donors_path("paris", after_kind)),
            ],
            ignore_index=True,
        ),
        grenoble_daily=before.grenoble_daily.copy(),
        paris_daily=before.paris_daily.copy(),
    ).normalized()

    return before, after
