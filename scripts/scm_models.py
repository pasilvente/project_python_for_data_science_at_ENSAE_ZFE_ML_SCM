"""
Outils pour contrôle synthétique mensuel (NO2).

Hypothèse importante : les fichiers journaliers passés à ces fonctions
ont déjà été nettoyés et, le cas échéant, imputés (interpolation journalière).
Ici, on se contente :
 - d'agréger au mensuel,
 - d'aligner la station traitée et les donneurs sur une même grille,
 - d'ajuster les modèles de contrôle synthétique.

On travaille au format "maison" :
    date, station_id, station_name, station_env, station_influence,
    no2_ug_m3, lat, lon
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
)


def _build_monthly_panel(
    treated_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    outcome_col: str = "no2_ug_m3",
) -> Tuple[pd.DatetimeIndex, pd.Series, pd.DataFrame]:
    """
    Construit le panel mensuel traité + donneurs à partir de données journalières.

    Paramètres
    ----------
    treated_daily : DataFrame
        Données journalières contenant au moins la station traitée.
    donors_daily : DataFrame
        Données journalières des stations donneuses (une ou plusieurs stations).
    treated_id : str
        Identifiant de la station traitée (station_id_col).
    treatment_start : Timestamp
        Date de début de l'intervention (ZFE). Sert uniquement pour vérifier
        qu'on a bien une période pré-traitement suffisante.
    date_col : str
        Nom de la colonne de date.
    station_id_col : str
        Nom de la colonne d'identifiant de station.
    outcome_col : str
        Nom de la variable de résultat (NO2).

    Retour
    ------
    dates : DatetimeIndex
        Index des dates mensuelles (début de mois) retenues.
    y_treated : Series
        Série mensuelle de la station traitée (index = dates).
    donor_matrix : DataFrame
        Matrice mensuelle des donneurs (index = dates, colonnes = station_id).
        Toutes les valeurs sont non manquantes (les mois incomplets sont exclus).
    """
    df_treated = treated_daily.copy()
    df_donors = donors_daily.copy()

    df_treated[date_col] = pd.to_datetime(df_treated[date_col])
    df_donors[date_col] = pd.to_datetime(df_donors[date_col])

    # Station traitée
    df_treated = df_treated[df_treated[station_id_col] == treated_id].copy()
    if df_treated.empty:
        raise ValueError(
            f"Impossible de trouver la station traitée '{treated_id}' "
            f"dans le DataFrame passed en treated_daily."
        )

    treated_monthly = (
        df_treated
        .set_index(date_col)[outcome_col]
        .resample("MS")
        .mean()
    )

    # Donneurs mensuels (moyenne mensuelle par station)
    donors_monthly = (
        df_donors
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

    # Panel commun traité + donneurs
    panel = donors_wide.copy()
    panel["treated"] = treated_monthly
    panel = panel.sort_index()

    # On retire tous les mois où : la station traitée est manquante ou au moins un donneur est manquant
    panel = panel.dropna(how="any")

    if panel.empty:
        raise ValueError(
            "Le panel mensuel est vide après suppression des mois avec valeurs manquantes. "
            "Vérifier les dates et la qualité des données journalières (imputation, etc.)."
        ) # Ce n'est évidemment pas le cas ici car on a traité les valeurs manquantes !

    dates = panel.index
    donor_matrix = panel.drop(columns=["treated"])
    y_treated = panel["treated"]

    # Vérification minimale : existence d'une vraie période pré-traitement
    if not (dates < treatment_start).any():
        raise ValueError(
            "Aucune observation pré-traitement dans le panel mensuel. "
            "Vérifier la date de début de ZFE ou la période des données."
        )

    return dates, y_treated, donor_matrix


def fit_penalized_scm_monthly(
    treated_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    outcome_col: str = "no2_ug_m3",
    model_types: Sequence[str] = ("ols", "ridge", "lasso", "elasticnet"),
    alphas_ridge: Optional[Iterable[float]] = None,
    alphas_lasso: Optional[Iterable[float]] = None,
    alphas_enet: Optional[Iterable[float]] = None,
    l1_ratios_enet: Optional[Iterable[float]] = None,
) -> Tuple[pd.DatetimeIndex, pd.Series, Dict[str, pd.Series], Dict[str, pd.Series]]:
    """
    Ajuste un contrôle synthétique mensuel à partir de séries journalières.

    Les étapes sont les suivantes :
      1. Construction d'un panel mensuel traité + donneurs sans valeurs manquantes
         (une ligne par mois, une colonne par station donneuse).
      2. Ajustement sur la période pré-traitement uniquement.
      3. Prédiction sur toute la période pour obtenir les séries synthétiques.

    Quatre variantes sont disponibles :
      - 'ols'        : régression linéaire sans intercept (SCM "classique" sans pénalisation),
      - 'ridge'      : RidgeCV (alpha choisi par CV),
      - 'lasso'      : LassoCV,
      - 'elasticnet' : ElasticNetCV.

    Paramètres
    ----------
    treated_daily, donors_daily, treated_id, treatment_start :
        Voir _build_monthly_panel.
    model_types : séquence de str
        Sous-ensemble de {"ols", "ridge", "lasso", "elasticnet"}.
    alphas_ridge, alphas_lasso, alphas_enet :
        Grilles d'alphas pour les modèles pénalisés. Si None, des grilles
        log-spacées standards sont utilisées.
    l1_ratios_enet :
        Grille de l1_ratio pour ElasticNetCV. Si None, valeurs usuelles.

    Retour
    ------
    dates : DatetimeIndex
        Dates mensuelles (début de mois) du panel.
    y_treated : Series
        Série mensuelle observée de la station traitée.
    synthetic_dict : dict[str, Series]
        Pour chaque modèle, la série synthétique pré/post ZFE.
    weights_dict : dict[str, Series]
        Pour chaque modèle, les poids associés à chaque station donneuse.
    """
    # Panel mensuel propre
    dates, y_treated, donor_matrix = _build_monthly_panel(
        treated_daily=treated_daily,
        donors_daily=donors_daily,
        treated_id=treated_id,
        treatment_start=treatment_start,
        date_col=date_col,
        station_id_col=station_id_col,
        outcome_col=outcome_col,
    )

    donor_ids = donor_matrix.columns
    X_all = donor_matrix.values

    pre_mask = dates < treatment_start
    X_pre = donor_matrix.loc[pre_mask].values
    y_pre = y_treated.loc[pre_mask].values

    synthetic_dict: Dict[str, pd.Series] = {}
    weights_dict: Dict[str, pd.Series] = {}

    # Grilles par défaut
    if alphas_ridge is None:
        alphas_ridge = np.logspace(-3, 3, 50)
    if alphas_lasso is None:
        alphas_lasso = np.logspace(-3, 1, 50)
    if alphas_enet is None:
        alphas_enet = np.logspace(-3, 1, 40)
    if l1_ratios_enet is None:
        l1_ratios_enet = [0.2, 0.5, 0.8]

    # SCM "classique" i.e. régression OLS sans intercept
    if "ols" in model_types:
        ols = LinearRegression(fit_intercept=False)
        ols.fit(X_pre, y_pre)

        y_hat = ols.predict(X_all)
        synthetic_dict["ols"] = pd.Series(y_hat, index=dates, name="ols")
        weights_dict["ols"] = pd.Series(ols.coef_, index=donor_ids, name="ols")

    # Ridge pénalisé
    if "ridge" in model_types:
        ridge = RidgeCV(alphas=list(alphas_ridge), cv=5, fit_intercept=False)
        ridge.fit(X_pre, y_pre)

        y_hat = ridge.predict(X_all)
        synthetic_dict["ridge"] = pd.Series(y_hat, index=dates, name="ridge")
        weights_dict["ridge"] = pd.Series(ridge.coef_, index=donor_ids, name="ridge")

    # Lasso pénalisé
    if "lasso" in model_types:
        lasso = LassoCV(
            alphas=list(alphas_lasso),
            cv=5,
            fit_intercept=False,
            max_iter=10_000,
            random_state=0,
        )
        lasso.fit(X_pre, y_pre)

        y_hat = lasso.predict(X_all)
        synthetic_dict["lasso"] = pd.Series(y_hat, index=dates, name="lasso")
        weights_dict["lasso"] = pd.Series(lasso.coef_, index=donor_ids, name="lasso")

    # ElasticNet pénalisé
    if "elasticnet" in model_types:
        enet = ElasticNetCV(
            alphas=list(alphas_enet),
            l1_ratio=list(l1_ratios_enet),
            cv=5,
            fit_intercept=False,
            max_iter=10_000,
            random_state=0,
        )
        enet.fit(X_pre, y_pre)

        y_hat = enet.predict(X_all)
        synthetic_dict["elasticnet"] = pd.Series(y_hat, index=dates, name="elasticnet")
        weights_dict["elasticnet"] = pd.Series(enet.coef_, index=donor_ids, name="elasticnet")

    return dates, y_treated, synthetic_dict, weights_dict



def compute_att_summary(
    dates: pd.DatetimeIndex,
    y_treated: pd.Series,
    synthetic_dict: Dict[str, pd.Series],
    treatment_start: pd.Timestamp,
    covid_start: Optional[pd.Timestamp] = None,
    covid_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Calcule des ATT moyens pré / post / post hors Covid pour chaque méthode.

    ATT_t = Y_treated,t - Y_synth,t

    Paramètres
    ----------
    dates : DatetimeIndex
        Index temporel commun.
    y_treated : Series
        Série observée de la station traitée (index = dates).
    synthetic_dict : dict[str, Series]
        Séries synthétiques renvoyées par fit_penalized_scm_monthly.
    treatment_start : Timestamp
        Date de début de l'intervention.
    covid_start, covid_end : Timestamp ou None
        Fenêtre à exclure du calcul "post_sans_Covid". Si None, la colonne
        ATT_moy_post_sans_Covid sera égale à ATT_moy_post.

    Retour
    ------
    DataFrame avec colonnes :
        - méthode
        - ATT_moy_pre
        - ATT_moy_post
        - ATT_moy_post_sans_Covid
    """
    if not isinstance(y_treated, pd.Series):
        y_treated = pd.Series(y_treated, index=dates, name="treated")
    else:
        y_treated = y_treated.reindex(dates)

    pre_mask = dates < treatment_start
    post_mask = dates >= treatment_start

    if covid_start is not None and covid_end is not None:
        covid_mask = (dates >= covid_start) & (dates <= covid_end)
        post_sans_covid_mask = post_mask & ~covid_mask
    else:
        covid_mask = pd.Series(False, index=dates)
        post_sans_covid_mask = post_mask

    rows = []
    for key, y_syn in synthetic_dict.items():
        y_syn = y_syn.reindex(dates)
        att = y_treated - y_syn

        rows.append(
            {
                "méthode": key.capitalize(),
                "ATT_moy_pre": att[pre_mask].mean(),
                "ATT_moy_post": att[post_mask].mean(),
                "ATT_moy_post_sans_Covid": att[post_sans_covid_mask].mean(),
            }
        )

    return pd.DataFrame(rows)


def _build_daily_panel(
    treated_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    outcome_col: str = "no2_ug_m3",
) -> tuple[pd.DatetimeIndex, pd.Series, pd.DataFrame]:
    """
    Construit le panel journalier (traitée + donneurs) pour le SCM.

    Paramètres
    ----------
    treated_daily : DataFrame
        Données journalières pour la zone traitée (plusieurs stations possibles).
    donors_daily : DataFrame
        Données journalières pour les stations donneuses.
    treated_id : str
        Identifiant de la station traitée (valeur de `station_id`).
    treatment_start : Timestamp
        Date de début du traitement (ZFE).
    date_col : str
        Nom de la colonne de dates (par défaut "date").
    station_id_col : str
        Nom de la colonne identifiant les stations (par défaut "station_id").
    outcome_col : str
        Nom de la colonne de concentration (par défaut "no2_ug_m3").

    Retour
    ------
    dates : DatetimeIndex
        Index des dates utilisées dans le panel.
    y_treated : Series
        Série journalière observée pour la station traitée.
    donor_matrix : DataFrame
        Matrice journalère des donneurs (colonnes = station_id, index = dates).
    """

    # Copie
    treated = treated_daily.copy()
    donors = donors_daily.copy()

    # Mise au bon type pour la date
    treated[date_col] = pd.to_datetime(treated[date_col])
    donors[date_col] = pd.to_datetime(donors[date_col])

    # Station traitée, on garde uniquement treated_id
    treated = treated.loc[treated[station_id_col] == treated_id].copy()
    if treated.empty:
        raise ValueError(f"Aucune observation trouvée pour la station traitée {treated_id}.")

    # En cas de doublons sur une date, on moyenne
    treated_series = (
        treated
        .groupby(date_col)[outcome_col]
        .mean()
        .sort_index()
    )

    # Donneurs, moyenne par station_id et date
    donors_agg = (
        donors
        .groupby([date_col, station_id_col], as_index=False)[outcome_col]
        .mean()
    )

    donors_wide = (
        donors_agg
        .pivot(index=date_col, columns=station_id_col, values=outcome_col)
        .sort_index()
    )

    # On ne garde que l'intersection des dates où la station traitée est observée
    # et où au moins un donneur a une valeur, pas utile ici car cas des valeurs manquantes traitées
    panel = donors_wide.copy()
    panel["treated"] = treated_series
    panel = panel.sort_index()

    # On retire les dates où la série traitée est manquante
    # Pas utile ici car on a traité le cas des valeurs manquantes
    panel = panel.dropna(subset=["treated"])

    # On retire les lignes où au moins un donneur est manquant
    # Pas utile ici car on a traité le cas des valeurs manquantes
    panel = panel.dropna(how="any")

    if panel.empty:
        raise ValueError("Panel journalier vide après nettoyage. Vérifier les recouvrements de dates.")

    # On restreint éventuellement à [min_date, max_date] explicites
    # Ici on se contente de renvoyer la plage couverte par le panel
    dates = panel.index
    y_treated = panel["treated"]
    donor_matrix = panel.drop(columns=["treated"])

    return dates, y_treated, donor_matrix

def fit_penalized_scm_daily(
    treated_daily: pd.DataFrame,
    donors_daily: pd.DataFrame,
    treated_id: str,
    treatment_start: pd.Timestamp,
    date_col: str = "date",
    station_id_col: str = "station_id",
    outcome_col: str = "no2_ug_m3",
    model_types: tuple[str, ...] = ("ols", "ridge", "lasso", "elasticnet"),
    alphas_ridge: np.ndarray | None = None,
    alphas_lasso: np.ndarray | None = None,
    alphas_en: np.ndarray | None = None,
    l1_ratio_grid: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> tuple[pd.DatetimeIndex, pd.Series, dict[str, pd.Series], dict[str, pd.Series]]:
    """
    Ajuste un contrôle synthétique journalier avec pénalisation (ou non).

    Paramètres
    ----------
    treated_daily : DataFrame
        Données journalières pour la zone traitée.
    donors_daily : DataFrame
        Données journalières pour les stations donneuses.
    treated_id : str
        Identifiant de la station traitée.
    treatment_start : Timestamp
        Date de début de la ZFE.
    date_col, station_id_col, outcome_col : str
        Noms des colonnes de date, station et variable de sortie.
    model_types : tuple
        Modèles à ajuster parmi ("ols", "ridge", "lasso", "elasticnet").
    alphas_ridge, alphas_lasso, alphas_en : np.ndarray | None
        Grilles d'alphas pour la validation croisée.
    l1_ratio_grid : tuple[float, ...]
        Grille de l1_ratio pour ElasticNet.

    Retour
    ------
    dates : DatetimeIndex
        Index des dates du panel.
    y_treated : Series
        Série observée pour la station traitée.
    synthetic_dict : dict[str, Series]
        Séries synthétiques par méthode.
    weights_dict : dict[str, Series]
        Coefficients (poids) des donneurs par méthode.
    """

    # Construction du panel journalier (traitée + donneurs)
    dates, y_treated, donor_matrix = _build_daily_panel(
        treated_daily=treated_daily,
        donors_daily=donors_daily,
        treated_id=treated_id,
        treatment_start=treatment_start,
        date_col=date_col,
        station_id_col=station_id_col,
        outcome_col=outcome_col,
    )

    # Découpage pré / post traitement
    pre_mask = dates < treatment_start
    if pre_mask.sum() < 10:
        raise ValueError("Trop peu de points d'observation en pré-traitement pour ajuster le SCM daily.")

    X_pre = donor_matrix.loc[pre_mask]
    y_pre = y_treated.loc[pre_mask]
    X_all = donor_matrix

    donor_ids = donor_matrix.columns.tolist()

    if alphas_ridge is None:
        alphas_ridge = np.logspace(-3, 3, 40)
    if alphas_lasso is None:
        alphas_lasso = np.logspace(-3, 1, 40)
    if alphas_en is None:
        alphas_en = np.logspace(-3, 1, 40)

    synthetic_dict: dict[str, pd.Series] = {}
    weights_dict: dict[str, pd.Series] = {}

    # OLS classique (équivalent SCM sans pénalisation)
    if "ols" in model_types:
        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_pre, y_pre)
        y_hat_ols = pd.Series(ols.predict(X_all), index=dates, name="ols")
        w_ols = pd.Series(ols.coef_, index=donor_ids, name="ols")

        synthetic_dict["ols"] = y_hat_ols
        weights_dict["ols"] = w_ols

    # Ridge
    if "ridge" in model_types:
        ridge = RidgeCV(alphas=alphas_ridge, fit_intercept=True, cv=5)
        ridge.fit(X_pre, y_pre)
        y_hat_ridge = pd.Series(ridge.predict(X_all), index=dates, name="ridge")
        w_ridge = pd.Series(ridge.coef_, index=donor_ids, name="ridge")

        synthetic_dict["ridge"] = y_hat_ridge
        weights_dict["ridge"] = w_ridge

    # Lasso
    if "lasso" in model_types:
        lasso = LassoCV(alphas=alphas_lasso, fit_intercept=True, cv=5, max_iter=10000)
        lasso.fit(X_pre, y_pre)
        y_hat_lasso = pd.Series(lasso.predict(X_all), index=dates, name="lasso")
        w_lasso = pd.Series(lasso.coef_, index=donor_ids, name="lasso")

        synthetic_dict["lasso"] = y_hat_lasso
        weights_dict["lasso"] = w_lasso

    # ElasticNet
    if "elasticnet" in model_types:
        en = ElasticNetCV(
            alphas=alphas_en,
            l1_ratio=list(l1_ratio_grid),
            fit_intercept=True,
            cv=5,
            max_iter=10000,
        )
        en.fit(X_pre, y_pre)
        y_hat_en = pd.Series(en.predict(X_all), index=dates, name="elasticnet")
        w_en = pd.Series(en.coef_, index=donor_ids, name="elasticnet")

        synthetic_dict["elasticnet"] = y_hat_en
        weights_dict["elasticnet"] = w_en

    return dates, y_treated, synthetic_dict, weights_dict

# Fonctions utilitaires pour les tableaux de poids

def make_weights_tables(weights_dict, donors_daily):
    """
    Construit, pour chaque méthode SCM, un tableau des donneurs
    avec métadonnées et poids associés.
    """
    meta = (
        donors_daily[
            ["station_id", "station_name", "station_env", "station_influence"]
        ]
        .drop_duplicates()
        .set_index("station_id")
    )

    tables = {}
    for method, w in weights_dict.items():
        df = (
            w.to_frame("poids")
            .reset_index()
            .rename(columns={"index": "station_id"})
            .merge(meta, on="station_id", how="left")
            [["station_id", "station_name", "station_env", "station_influence", "poids"]]
            .sort_values("poids", ascending=False)
        )
        tables[method] = df

    return tables
