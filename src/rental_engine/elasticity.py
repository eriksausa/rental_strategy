"""
Own-price response of occupancy, estimated from a balanced unit-month panel.

Model:   logit(occupancy_it) = fixed_effects + beta * log(ADR_it) + error

`beta` is the change in the log-odds of occupancy for a 1-unit change in log price
(about +1% price -> beta/100 change in log-odds). At occupancy p, the implied price
elasticity of the occupancy rate is `beta * (1 - p)`.

Four specifications are reported to show WHY the identification strategy matters:

  1. naive            Pooled OLS of logit(occ) on log(ADR). Biased upward: high-season
                      months have both high prices and high occupancy, and market-wide
                      demand shocks push both up together.
  2. controls         Unit effects + shared calendar-month effects + linear time trend.
                      Removes average seasonality and trend, but not market-wide shocks.
  3. twfe             Unit effects + year-month effects ("two-way fixed effects"). Removes
                      every market-wide shock, but cities with different seasonal amplitudes
                      (e.g., CDMX vs. beach markets) still confound price and occupancy.
  4. twfe_unit_season (headline) Unit x calendar-month effects + year-month effects. Absorbs
                      each unit's own seasonal pattern and all market-wide shocks; only a
                      unit's deviations from its own seasonal price pattern identify beta.

Uncertainty for the headline estimate comes from a cluster bootstrap resampling whole units.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .transforms import logit, smooth_occupancy


def _to_panel_matrices(panel: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a balanced unit-month panel into `[n_units x n_months]` matrices `(Y, X)`."""
    df = panel.sort_values(["unit_id", "year_month"])
    units = sorted(df["unit_id"].unique())
    months = sorted(df["year_month"].unique())
    if len(df) != len(units) * len(months):
        raise ValueError("estimate_elasticity requires a balanced panel (same months for every unit).")

    occ_smooth = smooth_occupancy(
        df["occupancy_rate"].to_numpy(float),
        df["days_in_month"].to_numpy(float),
    )
    y_log_odds = logit(occ_smooth)
    x_log_adr = np.log(df["adr_mxn"].to_numpy(float))

    Y = y_log_odds.reshape(len(units), len(months))
    X = x_log_adr.reshape(len(units), len(months))
    month_nums = np.array([int(ym[5:7]) for ym in months])
    return Y, X, month_nums


def _two_way_demean(matrix: np.ndarray) -> np.ndarray:
    """Frisch-Waugh-Lovell within-transformation for balanced `[units x time]` fixed effects."""
    return matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + matrix.mean()


def _ols_slope(x_centered: np.ndarray, y_centered: np.ndarray) -> float:
    """Return the simple OLS slope `sum(x * y) / sum(x^2)` for already-demeaned arrays."""
    return float(np.sum(x_centered * y_centered) / np.sum(x_centered * x_centered))


def _controls_slope(Y: np.ndarray, X: np.ndarray, month_nums: np.ndarray) -> float:
    """Estimate beta with unit dummies, shared calendar-month dummies, and a linear time trend."""
    n_units, n_months = Y.shape
    cols = [X.ravel()]

    unit_idx = np.repeat(np.arange(n_units), n_months)
    for i in range(1, n_units):
        cols.append((unit_idx == i).astype(float))

    month_idx = np.tile(month_nums, n_units)
    for m in range(2, 13):
        cols.append((month_idx == m).astype(float))

    cols.append(np.tile(np.arange(n_months, dtype=float), n_units))
    design = np.column_stack([np.ones(n_units * n_months)] + cols)
    coef, *_ = np.linalg.lstsq(design, Y.ravel(), rcond=None)
    return float(coef[1])


def _unit_season_twfe_slope(Y: np.ndarray, X: np.ndarray, month_nums: np.ndarray) -> float:
    """Estimate beta with unit x calendar-month fixed effects plus year-month fixed effects."""
    n_units, n_months = Y.shape
    n_obs = n_units * n_months

    # One indicator column per (unit, calendar_month) cell absorbs each unit's individual seasonal curve.
    unit_idx = np.repeat(np.arange(n_units), n_months)
    moy_idx = np.tile(month_nums - 1, n_units)
    cell_idx = unit_idx * 12 + moy_idx
    cell_dummies = np.zeros((n_obs, n_units * 12))
    cell_dummies[np.arange(n_obs), cell_idx] = 1.0

    # Year-month dummies (dropping t=0 to avoid collinearity) absorb all common market shocks.
    time_idx = np.tile(np.arange(n_months), n_units)
    time_dummies = np.zeros((n_obs, n_months - 1))
    nonzero_t = time_idx > 0
    time_dummies[np.arange(n_obs)[nonzero_t], time_idx[nonzero_t] - 1] = 1.0

    design = np.column_stack([X.ravel(), cell_dummies, time_dummies])
    coef, *_ = np.linalg.lstsq(design, Y.ravel(), rcond=None)
    return float(coef[0])


def estimate_elasticity(
    panel: pd.DataFrame,
    n_boot: int = 300,
    seed: int = 0,
    true_value: Optional[float] = None,
) -> Dict[str, object]:
    """Compare naive vs. fixed-effects price-response estimators with a unit cluster bootstrap.

    Args:
        panel: Balanced unit-month panel containing `unit_id`, `year_month`, `adr_mxn`,
            `occupancy_rate`, and `days_in_month`.
        n_boot: Number of unit-level cluster bootstrap resamples for the headline 95% CI.
        seed: Random seed for reproducible bootstrap confidence intervals.
        true_value: Optional ground-truth beta (inferred from `panel.attrs` if omitted).

    Returns:
        Dictionary with all four specification estimates, bootstrap SE / 95% CI, implied
        occupancy-rate elasticity at the panel mean occupancy, and sample dimensions.
    """
    Y, X, month_nums = _to_panel_matrices(panel)
    n_units = Y.shape[0]

    naive = _ols_slope(X - X.mean(), Y - Y.mean())
    controls = _controls_slope(Y, X, month_nums)
    twfe = _ols_slope(_two_way_demean(X), _two_way_demean(Y))
    headline = _unit_season_twfe_slope(Y, X, month_nums)

    # Resample entire unit rows (with replacement) to preserve within-unit serial correlation.
    rng = np.random.default_rng(seed)
    boot_draws = np.empty(n_boot)
    for b in range(n_boot):
        sampled_units = rng.integers(0, n_units, size=n_units)
        boot_draws[b] = _unit_season_twfe_slope(Y[sampled_units], X[sampled_units], month_nums)

    ci_low, ci_high = np.percentile(boot_draws, [2.5, 97.5])
    boot_se = float(np.std(boot_draws, ddof=1))

    mean_occ = float(panel["occupancy_rate"].mean())
    if true_value is None:
        true_value = panel.attrs.get("true_params", {}).get("elasticity")

    return {
        "naive_pooled_ols": round(naive, 3),
        "unit_month_trend_controls": round(controls, 3),
        "twfe_unit_and_year_month": round(twfe, 3),
        "twfe_unit_season": round(headline, 3),
        "headline_estimate": round(headline, 3),
        "headline_bootstrap_se": round(boot_se, 3),
        "headline_ci95_bootstrap": [round(float(ci_low), 3), round(float(ci_high), 3)],
        "implied_occupancy_price_elasticity_at_mean_occ": round(headline * (1.0 - mean_occ), 3),
        "mean_occupancy": round(mean_occ, 3),
        "n_units": int(n_units),
        "n_months": int(Y.shape[1]),
        "true_value": None if true_value is None else float(true_value),
    }
