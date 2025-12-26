"""
Fonctions liées au contrôle synthétique pénalisé (Ridge, Lasso, ElasticNet)
pour le projet ZFE / NO2.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error


def fit_penalized_scm(
    X_pre: pd.DataFrame,
    y_pre: pd.Series,
    X_all: pd.DataFrame,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, Dict[str, float]]]:
    """
    Ajuste un contrôle synthétique pénalisé (Ridge, Lasso, ElasticNet).

    Paramètres
    ----------
    X_pre : DataFrame
        Matrice des donneurs sur la période pré-traitement.
    y_pre : Series
        Série de la station traitée sur la période pré-traitement.
    X_all : DataFrame
        Matrice des donneurs sur l'ensemble de la période.

    Retour
    ------
    synth_series : dict
        Dictionnaire {nom_modele: série prédite sur toute la période}.
    weights : dict
        Dictionnaire {nom_modele: coefficients par donneur}.
    params : dict
        Dictionnaire {nom_modele: paramètres sélectionnés (alpha, l1_ratio)}.
    """
    synth_series: Dict[str, pd.Series] = {}
    weights: Dict[str, pd.Series] = {}
    params: Dict[str, Dict[str, float]] = {}

    # Ridge
    alphas_ridge = np.logspace(-3, 3, 50)
    ridge = RidgeCV(alphas=alphas_ridge, cv=5, fit_intercept=False)
    ridge.fit(X_pre, y_pre)
    y_ridge = pd.Series(ridge.predict(X_all), index=X_all.index, name="ridge")
    w_ridge = pd.Series(ridge.coef_, index=X_all.columns, name="ridge")

    synth_series["ridge"] = y_ridge
    weights["ridge"] = w_ridge
    params["ridge"] = {"alpha": float(ridge.alpha_)}

    # Lasso
    alphas_lasso = np.logspace(-3, 1, 50)
    lasso = LassoCV(alphas=alphas_lasso, cv=5, fit_intercept=False, max_iter=10000)
    lasso.fit(X_pre, y_pre)
    y_lasso = pd.Series(lasso.predict(X_all), index=X_all.index, name="lasso")
    w_lasso = pd.Series(lasso.coef_, index=X_all.columns, name="lasso")

    synth_series["lasso"] = y_lasso
    weights["lasso"] = w_lasso
    params["lasso"] = {"alpha": float(lasso.alpha_)}

    # ElasticNet
    alphas_en = np.logspace(-3, 1, 40)
    l1_ratios = [0.1, 0.5, 0.9]
    en = ElasticNetCV(
        alphas=alphas_en,
        l1_ratio=l1_ratios,
        cv=5,
        fit_intercept=False,
        max_iter=10000,
    )
    en.fit(X_pre, y_pre)
    y_en = pd.Series(en.predict(X_all), index=X_all.index, name="elasticnet")
    w_en = pd.Series(en.coef_, index=X_all.columns, name="elasticnet")

    synth_series["elasticnet"] = y_en
    weights["elasticnet"] = w_en
    params["elasticnet"] = {
        "alpha": float(en.alpha_),
        "l1_ratio": float(en.l1_ratio_),
    }

    return synth_series, weights, params


def compute_att_summary(
    treated: pd.Series,
    synth_dict: Dict[str, pd.Series],
    zfe_start: pd.Timestamp,
    covid_start: pd.Timestamp,
    covid_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Calcule les séries ATT_t = y_obs - y_synth et résume les ATT moyens
    avant et après traitement, avec et sans période Covid.

    Paramètres
    ----------
    treated : Series
        Série observée de la station traitée.
    synth_dict : dict
        Dictionnaire {nom_modele: série synthétique}.
    zfe_start : Timestamp
        Date de début de la ZFE (au format cohérent avec l'index).
    covid_start, covid_end : Timestamp
        Bornes approximatives de la période Covid à exclure
        pour l'ATT "post sans Covid".

    Retour
    ------
    att_summary : DataFrame
        ATT moyens pré, post, post hors Covid pour chaque modèle.
    att_series_dict : dict
        Dictionnaire {nom_modele: série ATT_t}.
    """
    att_rows = []
    att_series_dict: Dict[str, pd.Series] = {}

    for name, y_hat in synth_dict.items():
        common_idx = treated.index.intersection(y_hat.index)
        att = treated.loc[common_idx] - y_hat.loc[common_idx]
        att.name = f"ATT_{name}"
        att_series_dict[name] = att

        idx = att.index
        pre_mask = idx < zfe_start
        post_mask = idx >= zfe_start
        covid_mask = (idx >= covid_start) & (idx <= covid_end)
        post_nocovid_mask = post_mask & ~covid_mask

        att_rows.append(
            {
                "méthode": name,
                "ATT_moy_pre": att[pre_mask].mean(),
                "ATT_moy_post": att[post_mask].mean(),
                "ATT_moy_post_sans_Covid": att[post_nocovid_mask].mean(),
            }
        )

    att_summary = pd.DataFrame(att_rows)

    return att_summary, att_series_dict
