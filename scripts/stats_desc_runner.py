"""
stats_desc_runner.py
====================

Point d'entrée simple à appeler depuis rapport_zfe.ipynb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .stats_desc_inputs import StatsDescInputs, load_before_after_from_data_dir
from .stats_desc_figures import generate_selected_figures_for_report, FigurePaths


def run_stats_desc_report(
    *,
    data_dir: Path,
    out_dir: Path,
    grenoble_zfe_start,
    paris_zfe_start,
    before_inputs: Optional[StatsDescInputs] = None,
    after_inputs: Optional[StatsDescInputs] = None,
    connect_seasonality: bool = False,
) -> FigurePaths:
    """Génère un set réduit de figures pertinentes (raw + pre/post)."""

    out_dir = Path(out_dir)
    out_raw = out_dir / "raw"
    out_prepost = out_dir / "prepost"

    if before_inputs is None or after_inputs is None:
        b, a = load_before_after_from_data_dir(Path(data_dir))
        before_inputs = before_inputs or b
        after_inputs = after_inputs or a

    return generate_selected_figures_for_report(
        before_inputs=before_inputs,
        after_inputs=after_inputs,
        out_dir_raw=out_raw,
        out_dir_prepost=out_prepost,
        grenoble_zfe_start=pd.to_datetime(grenoble_zfe_start),
        paris_zfe_start=pd.to_datetime(paris_zfe_start),
        connect_seasonality=connect_seasonality,
    )
