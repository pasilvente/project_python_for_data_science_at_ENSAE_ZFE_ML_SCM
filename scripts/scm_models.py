# scripts/scm_models.py

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV


def fit_penalized_scm_monthly(
    treated_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    outcome_col: str = "no2_ug_m3",
    model_types: Iterable[str] = ("ridge", "lasso", "elasticnet"),
    alphas: Iterable[float] | None = None,
    l1_ratio: float = 0.5,
) -> Tuple[pd.DatetimeIndex, pd.Series, Dict[str, pd.Series], Dict[str, pd.Series]]:
    """
    Ajuste un contrôle synthétique pénalisé (Ridge, Lasso, ElasticNet) à partir
    de données journalières imputées, agrégées au pas mensuel.

    Paramètres
    ----------
    treated_daily : DataFrame
        Données journalières imputées pour la station traitée, contenant
        au minimum date_col, station_id_col et outcome_col.
    donors_daily : DataFrame
        Données journalières imputées pour les stations donneuses.
    treated_id : str
        Identifiant de la station traitée (valeur de station_id_col).
    treatment_start : pd.Timestamp
        Date de début du traitement (ZFE). Les mois dont la date est
        antérieure à treatment_start sont considérés comme pré-traitement.
    date_col : str
        Nom de la colonne de dates (par défaut 'date').
    station_id_col : str
        Nom de la colonne identifiant les stations (par défaut 'station_id').
    outcome_col : str
        Nom de la variable de résultat (par défaut 'no2_ug_m3').
    model_types : iterable de str
        Liste des modèles à ajuster parmi {'ridge', 'lasso', 'elasticnet'}.
    alphas : iterable de float ou None
        Grille d'alphas pour la validation croisée. Si None, une grille
        logarithmique standard est utilisée.
    l1_ratio : float
        Paramètre de mélange L1/L2 pour ElasticNet (entre 0 et 1).

    Retour
    ------
    dates : DatetimeIndex
        Index temporel mensuel (début de mois).
    treated_series : Series
        Série mensuelle observée pour la station traitée.
    synthetic_dict : dict
        Dictionnaire {nom_modele: série synthétique prédite}.
    weights_dict : dict
        Dictionnaire {nom_modele: série de coefficients par station donneuse}.
    """
    treated = treated_daily.copy()
    donors = donors_daily.copy()

    treated[date_col] = pd.to_datetime(treated[date_col])
    donors[date_col] = pd.to_datetime(donors[date_col])

    # Filtrage de la station traitée
    treated_sub = treated[treated[station_id_col] == treated_id].copy()
    if treated_sub.empty:
        raise ValueError(
            f"Aucune observation trouvée pour {station_id_col}={treated_id} "
            "dans treated_daily."
        )

    # Série mensuelle pour la station traitée
    treated_monthly = (
        treated_sub
        .set_index(date_col)[outcome_col]
        .resample("MS")  # début de mois
        .mean()
        .sort_index()
    )

    # Panel mensuel pour les donneurs
    donors_monthly = (
        donors
        .set_index(date_col)
        .groupby(station_id_col)[outcome_col]
        .resample("MS")
        .mean()
        .reset_index()
    )

    donors_wide = (
        donors_monthly
        .pivot(index=date_col, columns=station_id_col, values=outcome_col)
        .sort_index()
    )

    # Jointure traitée + donneurs, suppression des mois sans traitée
    panel = donors_wide.join(treated_monthly.rename("treated"), how="inner")
    panel = panel.dropna(subset=["treated"])

    # Suppression éventuelle de donneurs entièrement vides
    panel = panel.dropna(axis=1, how="all")

    donor_cols = [c for c in panel.columns if c != "treated"]
    if not donor_cols:
        raise ValueError(
            "Aucune colonne donneuse disponible après préparation du panel."
        )

    dates = panel.index
    X = panel[donor_cols].values
    y = panel["treated"].values

    # Période pré-traitement
    treatment_start = pd.to_datetime(treatment_start)
    pre_mask = dates < treatment_start
    if pre_mask.sum() < 5:
        raise ValueError(
            "Trop peu de points en période pré-traitement pour ajuster le SCM."
        )

    X_pre = X[pre_mask]
    y_pre = y[pre_mask]

    if alphas is None:
        alphas = np.logspace(-3, 2, 20)

    synthetic_dict: Dict[str, pd.Series] = {}
    weights_dict: Dict[str, pd.Series] = {}

    # Ridge
    if "ridge" in model_types:
        ridge = RidgeCV(alphas=alphas, fit_intercept=True, cv=5)
        ridge.fit(X_pre, y_pre)
        y_hat = ridge.predict(X)
        synthetic_dict["ridge"] = pd.Series(y_hat, index=dates)
        weights_dict["ridge"] = pd.Series(ridge.coef_, index=donor_cols)

    # Lasso
    if "lasso" in model_types:
            lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000)
            lasso.fit(X_pre, y_pre)
            y_hat = lasso.predict(X)
            synthetic_dict["lasso"] = pd.Series(y_hat, index=dates)
            weights_dict["lasso"] = pd.Series(lasso.coef_, index=donor_cols)

    # ElasticNet
    if "elasticnet" in model_types:
        en = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, cv=5, max_iter=10000)
        en.fit(X_pre, y_pre)
        y_hat = en.predict(X)
        synthetic_dict["elasticnet"] = pd.Series(y_hat, index=dates)
        weights_dict["elasticnet"] = pd.Series(en.coef_, index=donor_cols)

    treated_series = pd.Series(y, index=dates, name="treated")

    return dates, treated_series, synthetic_dict, weights_dict


def compute_att_summary(
    dates: pd.DatetimeIndex,
    y_treated: pd.Series,
    synthetic_dict: Dict[str, pd.Series],
    treatment_start: pd.Timestamp,
    covid_start: pd.Timestamp | None = None,
    covid_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Calcule des effets moyens du traitement (ATT) à partir des trajectoires
    observée et synthétiques.

    Paramètres
    ----------
    dates : DatetimeIndex
        Index temporel mensuel.
    y_treated : Series
        Série mensuelle observée pour la station traitée.
    synthetic_dict : dict
        Dictionnaire {nom_modele: série synthétique prédite}.
    treatment_start : pd.Timestamp
        Date de début de la ZFE au pas mensuel.
    covid_start, covid_end : pd.Timestamp ou None
        Bornes de la période Covid à exclure du calcul des ATT post-traitement
        "hors Covid". Si None, la colonne ATT_moy_post_sans_Covid sera à NaN.

    Retour
    ------
    summary : DataFrame
        Tableau résumant, pour chaque méthode, l'ATT moyen pré-traitement,
        post-traitement et post-traitement hors période Covid.
    """
    idx = pd.to_datetime(dates)
    treatment_start = pd.to_datetime(treatment_start)

    mask_pre = idx < treatment_start
    mask_post = idx >= treatment_start

    if covid_start is not None and covid_end is not None:
        covid_start = pd.to_datetime(covid_start)
        covid_end = pd.to_datetime(covid_end)
        mask_covid = (idx >= covid_start) & (idx <= covid_end)
    else:
        mask_covid = pd.Series(False, index=idx)

    rows = []
    for name, y_syn in synthetic_dict.items():
        y_syn_aligned = y_syn.reindex(idx)
        att = y_treated - y_syn_aligned

        att_pre = att[mask_pre].mean()
        att_post = att[mask_post].mean()

        if covid_start is not None and covid_end is not None:
            mask_post_no_covid = mask_post & (~mask_covid)
            att_post_no_covid = att[mask_post_no_covid].mean()
        else:
            att_post_no_covid = np.nan

        rows.append(
            {
                "méthode": name.capitalize(),
                "ATT_moy_pre": att_pre,
                "ATT_moy_post": att_post,
                "ATT_moy_post_sans_Covid": att_post_no_covid,
            }
        )

    summary = pd.DataFrame(rows)

    return summary
