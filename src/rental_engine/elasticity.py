"""
Own-price response of occupancy, estimated from a unit-month panel.

Model:   logit(occupancy_it) = unit_i + time_t + beta * log(ADR_it) + error

`beta` is the change in the log-odds of occupancy for a 1-unit change in log price
(about +1% price -> beta/100 change in log-odds). At occupancy p, the implied price
elasticity of the occupancy rate is  beta * (1 - p).

Three specifications are reported to show WHY the identification strategy matters:

  naive      pooled OLS of logit(occ) on log(ADR). Biased: high-season months have both
             high prices and high occupancy, and market-wide demand shocks push both up.
  controls   unit effects + calendar-month effects + linear trend. Removes seasonality
             and trend but not the market shock, so some bias remains.
  twfe       unit effects + year-month effects ("two-way fixed effects"). Removes every
             market-wide shock, but units with different seasonal patterns (e.g. a city with
             weaker seasonality) still confound the price/occupancy relationship.
  twfe_unit_season (headline)  unit x calendar-month effects + year-month effects. Each unit's
             own seasonal pattern and all market-wide shocks are absorbed; only the unit's
             deviations from its own seasonal price pattern identify beta.

Uncertainty for the headline estimate comes from a cluster bootstrap over units.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _logit_smooth(occ: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Smoothed log-odds so that 0% / 100% occupancy months stay finite."""
    p = (occ * days + 0.5) / (days + 1.0)
    return np.log(p / (1.0 - p))


def _to_matrices(panel: pd.DataFrame):
    df = panel.sort_values(["unit_id", "year_month"])
    units = sorted(df["unit_id"].unique())
    months = sorted(df["year_month"].unique())
    if len(df) != len(units) * len(months):
        raise ValueError("estimate_elasticity requires a balanced panel (same months for every unit).")
    y = _logit_smooth(df["occupancy_rate"].to_numpy(float), df["days_in_month"].to_numpy(float))
    x = np.log(df["adr_mxn"].to_numpy(float))
    Y = y.reshape(len(units), len(months))
    X = x.reshape(len(units), len(months))
    month_num = np.array([int(m[5:7]) for m in months])
    return Y, X, month_num


def _within(M: np.ndarray) -> np.ndarray:
    """Two-way demeaning for a balanced [units x time] matrix."""
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def _slope(xt: np.ndarray, yt: np.ndarray) -> float:
    return float(np.sum(xt * yt) / np.sum(xt * xt))


def _controls_slope(Y: np.ndarray, X: np.ndarray, month_num: np.ndarray) -> float:
    n_units, n_t = Y.shape
    cols = [X.ravel()]
    unit_idx = np.repeat(np.arange(n_units), n_t)
    for i in range(1, n_units):
        cols.append((unit_idx == i).astype(float))
    m_idx = np.tile(month_num, n_units)
    for m in range(2, 13):
        cols.append((m_idx == m).astype(float))
    cols.append(np.tile(np.arange(n_t, dtype=float), n_units))
    design = np.column_stack([np.ones(n_units * n_t)] + cols)
    coef, *_ = np.linalg.lstsq(design, Y.ravel(), rcond=None)
    return float(coef[1])


def _unit_season_twfe_slope(Y: np.ndarray, X: np.ndarray, month_num: np.ndarray) -> float:
    """Unit x calendar-month fixed effects plus year-month fixed effects (dummy regression)."""
    n_units, n_t = Y.shape
    n = n_units * n_t
    unit_idx = np.repeat(np.arange(n_units), n_t)
    moy_idx = np.tile(month_num - 1, n_units)
    cell = unit_idx * 12 + moy_idx
    cell_dummies = np.zeros((n, n_units * 12))
    cell_dummies[np.arange(n), cell] = 1.0
    t_idx = np.tile(np.arange(n_t), n_units)
    time_dummies = np.zeros((n, n_t - 1))
    mask = t_idx > 0
    time_dummies[np.arange(n)[mask], t_idx[mask] - 1] = 1.0
    design = np.column_stack([X.ravel(), cell_dummies, time_dummies])
    coef, *_ = np.linalg.lstsq(design, Y.ravel(), rcond=None)
    return float(coef[0])


def estimate_elasticity(
    panel: pd.DataFrame,
    n_boot: int = 300,
    seed: int = 0,
    true_value: Optional[float] = None,
) -> Dict[str, object]:
    Y, X, month_num = _to_matrices(panel)
    n_units = Y.shape[0]

    naive = _slope(X - X.mean(), Y - Y.mean())
    controls = _controls_slope(Y, X, month_num)

    Yt, Xt = _within(Y), _within(X)
    twfe = _slope(Xt, Yt)

    headline = _unit_season_twfe_slope(Y, X, month_num)

    # Cluster bootstrap over units for the headline estimate.
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        draws[b] = _unit_season_twfe_slope(Y[idx], X[idx], month_num)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    se = float(np.std(draws, ddof=1))

    mean_occ = float(panel["occupancy_rate"].mean())
    if true_value is None:
        true_value = panel.attrs.get("true_params", {}).get("elasticity")

    return {
        "naive_pooled_ols": round(naive, 3),
        "unit_month_trend_controls": round(controls, 3),
        "twfe_unit_and_year_month": round(twfe, 3),
        "twfe_unit_season": round(headline, 3),
        "headline_estimate": round(headline, 3),
        "headline_bootstrap_se": round(se, 3),
        "headline_ci95_bootstrap": [round(float(lo), 3), round(float(hi), 3)],
        "implied_occupancy_price_elasticity_at_mean_occ": round(headline * (1.0 - mean_occ), 3),
        "mean_occupancy": round(mean_occ, 3),
        "n_units": int(n_units),
        "n_months": int(Y.shape[1]),
        "true_value": None if true_value is None else float(true_value),
    }
