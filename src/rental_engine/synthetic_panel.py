"""
Synthetic monthly rental panel with a KNOWN price response (demo + method validation).

Why this exists
---------------
The real operating data behind the original analysis is private and is NOT part of this
repository. This generator produces an illustrative panel whose data-generating process is
fully specified, so that:

  * the forecasting pipeline (`forecasting.py`) is evaluated on enough history to be
    meaningful (12 units x 48 months, rolling-origin backtest), and
  * the elasticity estimator (`elasticity.py`) can be validated: the true price response is
    set here, so we can check that the method recovers it.

Everything produced here is SYNTHETIC. Results computed on it describe the generator,
not the real market. Public anchors (AirDNA city occupancy, seasonal curves, base ADRs)
come from `assumptions.py`.

Data-generating process (unit i, month t)
-----------------------------------------
  market shock         u_t ~ AR(1) with persistence `shock_phi`, shared by all units
  reference log-price  r_it = log(base_adr_i) + city_amp * log(season_adr[m]) + trend_adr * t
  posted log-price     p_it = r_it + pass_through * u_t + e_price        e_price ~ N(0, price_noise_sd)
  demand (log-odds)    z_it = a_i + city_amp * season_occ[m] + trend_occ * t + u_t
                              + elasticity * (p_it - r_it) + e_occ        e_occ ~ N(0, occ_noise_sd)
  occupied nights      k_it ~ Binomial(days_in_month, sigmoid(z_it))
  observed             ADR = exp(p_it),  occupancy = k_it / days_in_month

`elasticity` is the change in the log-odds of occupancy per unit change in log price
relative to the unit's own reference price. The market shock raises both prices (hosts
pass part of it through) and demand, so a naive regression of occupancy on price is badly
biased; fixed effects remove the shock (see `elasticity.py`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .assumptions import (
    AIRDNA_PUBLIC_BENCHMARKS,
    BASE_SEASONAL_ADR_MULT,
    BASE_SEASONAL_OCC_CURVE,
    PORTFOLIO_UNITS,
)
from .transforms import inv_logit, logit


@dataclass(frozen=True)
class TrueParams:
    """Ground-truth structural parameters of the synthetic data-generating process."""

    elasticity: float = -1.6          # d logit(occ) / d log(ADR), own-price response
    pass_through: float = 0.5         # Share of the market demand shock passed into posted prices
    shock_phi: float = 0.6            # AR(1) month-to-month persistence of the market shock
    shock_sd: float = 0.12            # Innovation std of the market shock (log-odds scale)
    price_noise_sd: float = 0.15      # Exogenous host pricing deviations (log-MXN scale)
    occ_noise_sd: float = 0.10        # Idiosyncratic demand noise (log-odds scale)
    trend_adr_per_month: float = 0.004  # ~4.9% annual nominal ADR drift
    trend_occ_per_month: float = 0.0


# Seasonal amplitude per city (urban CDMX has much flatter seasonality than beach markets).
_CITY_SEASONAL_AMP: Dict[str, float] = {"Playa del Carmen": 1.00, "Tulum": 1.00, "CDMX": 0.40}
_CITY_BASE_ADR_1BR: Dict[str, float] = {"Playa del Carmen": 1650.0, "Tulum": 1550.0, "CDMX": 1850.0}

# Seven extra synthetic units give the panel 12 units total for cross-sectional fixed effects.
_EXTRA_UNITS: List[Tuple[str, str, int]] = [
    ("SYN_PDC_04", "Playa del Carmen", 1),
    ("SYN_PDC_05", "Playa del Carmen", 2),
    ("SYN_PDC_06", "Playa del Carmen", 1),
    ("SYN_TUL_02", "Tulum", 2),
    ("SYN_TUL_03", "Tulum", 1),
    ("SYN_CDMX_02", "CDMX", 2),
    ("SYN_CDMX_03", "CDMX", 1),
]


def default_units() -> List[Tuple[str, str, int, Optional[float]]]:
    """Return `(unit_id, city, bedrooms, base_adr_or_None)` for 5 portfolio units + 7 extras."""
    units: List[Tuple[str, str, int, Optional[float]]] = [
        (uid, str(info["city"]), int(info["bedrooms"]), float(info["base_adr_mxn"]))
        for uid, info in PORTFOLIO_UNITS.items()
    ]
    units += [(uid, city, beds, None) for uid, city, beds in _EXTRA_UNITS]
    return units


def _centered_seasonal_curves() -> Tuple[Dict[int, float], Dict[int, float]]:
    """Convert raw monthly curves to zero-mean log-odds (occ) and log-multiplier (ADR) profiles."""
    raw_occ = {m: float(logit(np.array(p))) for m, p in BASE_SEASONAL_OCC_CURVE.items()}
    mean_occ = float(np.mean(list(raw_occ.values())))
    season_occ = {m: val - mean_occ for m, val in raw_occ.items()}

    raw_adr = {m: float(np.log(mult)) for m, mult in BASE_SEASONAL_ADR_MULT.items()}
    mean_adr = float(np.mean(list(raw_adr.values())))
    season_adr = {m: val - mean_adr for m, val in raw_adr.items()}
    return season_occ, season_adr


def _simulate_market_shock(rng: np.random.Generator, n_months: int, params: TrueParams) -> np.ndarray:
    """Simulate a stationary AR(1) market-wide demand shock shared across all units."""
    shock = np.zeros(n_months)
    # Draw t=0 from the stationary marginal variance: sigma^2 / (1 - phi^2).
    stationary_sd = params.shock_sd / np.sqrt(1.0 - params.shock_phi ** 2)
    shock[0] = rng.normal(0.0, stationary_sd)
    for t in range(1, n_months):
        shock[t] = params.shock_phi * shock[t - 1] + rng.normal(0.0, params.shock_sd)
    return shock


def generate_panel(
    seed: int = 7,
    n_months: int = 48,
    start: str = "2022-01",
    units: Optional[List[Tuple[str, str, int, Optional[float]]]] = None,
    params: TrueParams = TrueParams(),
) -> pd.DataFrame:
    """Generate a deterministic, balanced synthetic unit-month panel.

    Returns:
        DataFrame with columns `unit_id`, `city`, `bedrooms`, `year_month`, `month_num`,
        `t`, `days_in_month`, `adr_mxn`, `occupancy_rate`, `nights_occupied`, `p_true`.
        `p_true` (latent occupancy probability) is stored for noise-floor diagnostics only
        and is never used as a model feature.
    """
    units = units or default_units()
    rng = np.random.default_rng(seed)
    periods = pd.period_range(start=start, periods=n_months, freq="M")
    month_nums = np.array([p.month for p in periods])
    days = np.array([p.days_in_month for p in periods])

    season_occ, season_adr = _centered_seasonal_curves()
    market_shock = _simulate_market_shock(rng, n_months, params)

    rows = []
    for unit_id, city, bedrooms, base_adr in units:
        seasonal_amp = _CITY_SEASONAL_AMP[city]
        if base_adr is None:
            # 2BR units carry a ~45% ADR premium over 1BR, with small unit-specific dispersion.
            bedroom_mult = 1.0 if bedrooms == 1 else 1.45
            base_adr = _CITY_BASE_ADR_1BR[city] * bedroom_mult * float(np.exp(rng.normal(0.0, 0.08)))
        log_base_adr = float(np.log(base_adr))

        # Anchor unit-level baseline occupancy to the cited AirDNA city benchmark.
        city_occ_log_odds = float(logit(np.array(AIRDNA_PUBLIC_BENCHMARKS[city]["annual_mean_occupancy"])))
        unit_occ_intercept = city_occ_log_odds + rng.normal(0.0, 0.25)

        for t in range(n_months):
            month = int(month_nums[t])
            ref_log_price = log_base_adr + seasonal_amp * season_adr[month] + params.trend_adr_per_month * t
            # Posted price reflects both the shared market shock (confounder) and exogenous host pricing noise.
            posted_log_price = (
                ref_log_price
                + params.pass_through * market_shock[t]
                + rng.normal(0.0, params.price_noise_sd)
            )
            demand_log_odds = (
                unit_occ_intercept
                + seasonal_amp * season_occ[month]
                + params.trend_occ_per_month * t
                + market_shock[t]
                + params.elasticity * (posted_log_price - ref_log_price)
                + rng.normal(0.0, params.occ_noise_sd)
            )
            p_true = float(inv_logit(demand_log_odds))
            # Realized occupancy is a discrete count of booked nights out of the month's calendar days.
            nights = int(rng.binomial(int(days[t]), p_true))
            rows.append(
                {
                    "unit_id": unit_id,
                    "city": city,
                    "bedrooms": bedrooms,
                    "year_month": str(periods[t]),
                    "month_num": month,
                    "t": t,
                    "days_in_month": int(days[t]),
                    "adr_mxn": round(float(np.exp(posted_log_price)), 2),
                    "occupancy_rate": nights / float(days[t]),
                    "nights_occupied": nights,
                    "p_true": p_true,
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["true_params"] = asdict(params)
    df.attrs["synthetic"] = True
    return df
