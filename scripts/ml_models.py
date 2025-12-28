"""
Modèles de machine learning pour construire un contrefactuel de NO2.

Les fonctions ci-dessous prennent en entrée :
 - les données journalières imputées de la station traitée,
 - les données journalières imputées d'un panel d'autres stations françaises
   (les mêmes que celles utilisées comme « donneurs » dans le SCM),
 - la date de début de la ZFE,

et ajustent des modèles de régression supervisée uniquement sur la
période pré-traitement. On prédit ensuite la trajectoire contrefactuelle
sur toute la période (pré + post) à partir :
 - des niveaux de NO2 observés sur ces autres stations,
 - et de quelques variables calendaires (saison, jour de la semaine).

On obtient ainsi, pour chaque modèle, une série synthétique Y_t^ML
que l'on peut comparer à la série observée de la station traitée ou
résumer via un ATT, en réutilisant compute_att_summary du module
scripts.scm_models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from .scm_models import _build_daily_panel


def _default_ml_models(random_state: int = 0) -> Dict[str, Any]:
    """
    Définit un petit dictionnaire de modèles par défaut :

    - 'rf'   : RandomForestRegressor (scikit-learn)
    - 'lgbm' : LGBMRegressor (LightGBM)

    Les deux modèles sont adaptés à des relations non linéaires et
    à de fortes interactions entre stations (on peut imaginer que la
    dynamique du NO2 à Paris ou Grenoble soit mieux expliquée par une
    combinaison non linéaire des niveaux observés dans d'autres villes).

    Choix des hyperparamètres (compromis biais / variance) :

    *RandomForestRegressor*
    -----------------------
    - n_estimators=500 :
        Assez d'arbres pour stabiliser la prédiction par bagging,
        sans exploser les temps de calcul (avec ~8-9 ans de données
        journalières, on a quelques milliers d'observations).
    - max_depth=None :
        On laisse les arbres grandir, mais on contrôle la complexité
        via min_samples_leaf ; ça permet de laisser émerger des
        non-linéarités marquées sans aller jusqu'au sur-apprentissage
        sur quelques points isolés.
    - min_samples_leaf=10 :
        Empêche les feuilles de ne contenir que 1-2 observations.
        On impose un « lissage » minimal : chaque règle de décision
        est estimée sur au moins 10 jours, ce qui est raisonnable
        compte tenu du niveau de bruit jour-à-jour sur le NO2.
    - n_jobs=-1 :
        On parallélise les arbres pour rester compatible avec un
        environnement d'évaluation raisonnable (VSCode/SSP Cloud).

    *LGBMRegressor*
    ---------------
    - n_estimators=500, learning_rate=0.05 :
        Schéma classique de boosting « lent mais stable » :
        on ajoute beaucoup de petits arbres faiblement contributeurs,
        ce qui permet au modèle de capturer des effets complexes
        tout en limitant le sur-ajustement.
    - num_leaves=31 :
        Taille des arbres relativement modérée : on autorise des
        interactions non triviales mais on évite d'avoir des arbres
        gigantesques qui sur-apprennent la période pré-ZFE.
    - subsample=0.8, colsample_bytree=0.8 :
        On introduit du sous-échantillonnage en lignes et en colonnes
        pour réduire la variance et améliorer la robustesse des
        prédictions hors échantillon (même logique que le bagging).
    - random_state=random_state :
        Graine fixée pour la reproductibilité du projet.

    Ces valeurs ne prétendent pas être « optimales » au sens d'une
    recherche d'hyperparamètres exhaustive. L'idée est plutôt :
      - d'avoir des modèles suffisamment flexibles pour capturer les
        effets non linéaires et les interactions entre stations,
      - tout en restant raisonnables en termes de temps de calcul
        et de risque de sur-apprentissage sur la période pré-traitement.
    """
    models: Dict[str, Any] = {}

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=random_state,
    )
    models["rf"] = rf

    lgbm = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )
    models["lgbm"] = lgbm

    return models



def _build_ml_design_matrix_daily(
    treated_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    outcome_col: str = "no2_ug_m3",
    add_calendar_features: bool = True,
) -> Tuple[pd.DatetimeIndex, pd.Series, pd.DataFrame]:
    """
    Construit la matrice de régression (X, y) pour le ML en pas journalier.

    On part de _build_daily_panel (déjà utilisé pour le SCM) qui renvoie
    un panel "propre" :
        - une série y_treated (NO2 de la station traitée),
        - une matrice donor_matrix (NO2 des donneurs, une colonne par station).

    On enrichit éventuellement cette matrice avec des variables calendaires
    (mois, jour de la semaine encodés en sinus/cosinus) pour aider les modèles
    à capturer la saisonnalité.
    """
    dates, y_treated, donor_matrix = _build_daily_panel(
        treated_daily=treated_daily,
        donors_daily=donors_daily,
        treated_id=treated_id,
        treatment_start=treatment_start,
        date_col=date_col,
        station_id_col=station_id_col,
        outcome_col=outcome_col,
    )

    X = donor_matrix.copy()

    if add_calendar_features:
        cal = pd.DataFrame(index=dates)
        cal["month"] = dates.month
        cal["dow"] = dates.dayofweek

        # Encodages cycliques pour la saisonnalité
        cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12)
        cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12)
        cal["dow_sin"] = np.sin(2 * np.pi * cal["dow"] / 7)
        cal["dow_cos"] = np.cos(2 * np.pi * cal["dow"] / 7)

        X = pd.concat([X, cal], axis=1)

    return dates, y_treated, X


def fit_ml_counterfactual_daily(
    treated_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    outcome_col: str = "no2_ug_m3",
    models: Optional[Dict[str, Any]] = None,
    add_calendar_features: bool = True,
    random_state: int = 0,
) -> Tuple[pd.DatetimeIndex, pd.Series, Dict[str, pd.Series], Dict[str, Any]]:
    """
    Ajuste des modèles de ML pour construire un contrefactuel NO2 journalier.

    Étapes :
      1. Construction d'un panel journalier propre (traitée + donneurs)
         via _build_ml_design_matrix_daily.
      2. Définition des masques pré / post-traitement à partir de
         treatment_start.
      3. Ajustement des modèles uniquement sur le pré-traitement.
      4. Prédiction sur toute la période pour obtenir Y_t^ML.

    Paramètres
    ----------
    treated_daily, donors_daily, treated_id, treatment_start :
        Voir _build_ml_design_matrix_daily.
    models : dict[str, estimator] ou None
        Dictionnaire {nom: estimateur sklearn-like}.
        Si None, on utilise un RandomForest et un LGBM.
    add_calendar_features : bool
        Si True, ajoute des variables calendaires (mois / jour de semaine).
    random_state : int
        Graine de reproductibilité pour les modèles par défaut.

    Retour
    ------
    dates : DatetimeIndex
        Index des dates journalières du panel.
    y_treated : Series
        Série observée pour la station traitée.
    synthetic_dict : dict[str, Series]
        Pour chaque modèle, la série synthétique prédite (pré + post).
    fitted_models : dict[str, estimator]
        Modèles ajustés, pour inspection ultérieure si besoin.
    """
    # 1) Design matrix X + série traitée
    dates, y_treated, X = _build_ml_design_matrix_daily(
        treated_daily=treated_daily,
        donors_daily=donors_daily,
        treated_id=treated_id,
        treatment_start=treatment_start,
        date_col=date_col,
        station_id_col=station_id_col,
        outcome_col=outcome_col,
        add_calendar_features=add_calendar_features,
    )

    if models is None:
        models = _default_ml_models(random_state=random_state)

    # 2) Découpage pré / post
    pre_mask = dates < treatment_start
    if pre_mask.sum() < 50:
        raise ValueError(
            "Trop peu d'observations en pré-traitement pour ajuster "
            "des modèles de ML daily (moins de 50 jours)."
        )

    X_pre = X.loc[pre_mask]
    y_pre = y_treated.loc[pre_mask]

    synthetic_dict: Dict[str, pd.Series] = {}
    fitted_models: Dict[str, Any] = {}

    # 3) Ajustement et prédiction pour chaque modèle
    for name, est in models.items():
        m = clone(est)
        if hasattr(m, "random_state"):
            setattr(m, "random_state", random_state)

        m.fit(X_pre, y_pre)
        y_hat = pd.Series(m.predict(X), index=dates, name=name)

        synthetic_dict[name] = y_hat
        fitted_models[name] = m

    return dates, y_treated, synthetic_dict, fitted_models
