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
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .assumptions import (
    AIRDNA_PUBLIC_BENCHMARKS,
    BASE_SEASONAL_ADR_MULT,
    BASE_SEASONAL_OCC_CURVE,
    PORTFOLIO_UNITS,
)


@dataclass(frozen=True)
class TrueParams:
    """Parameters of the synthetic generator (the 'ground truth')."""
    elasticity: float = -1.6          # d log-odds(occupancy) / d log(price), own-price response
    pass_through: float = 0.5         # share of the market shock hosts pass into prices
    shock_phi: float = 0.6            # AR(1) persistence of the market demand shock
    shock_sd: float = 0.12            # innovation sd of the market shock (log-odds units)
    price_noise_sd: float = 0.15      # host pricing deviations (exogenous price variation)
    occ_noise_sd: float = 0.10        # idiosyncratic demand noise (log-odds units)
    trend_adr_per_month: float = 0.004
    trend_occ_per_month: float = 0.0


# City-level settings (CDMX has weaker seasonality than the beach markets).
_CITY_AMP = {"Playa del Carmen": 1.00, "Tulum": 1.00, "CDMX": 0.40}
_CITY_BASE_ADR_1BR = {"Playa del Carmen": 1650.0, "Tulum": 1550.0, "CDMX": 1850.0}

# Extra illustrative units so the panel has enough cross-sectional variation.
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
    """(unit_id, city, bedrooms, base_adr_or_None) for the 5 portfolio units plus 7 synthetic extras."""
    units: List[Tuple[str, str, int, Optional[float]]] = [
        (uid, str(u["city"]), int(u["bedrooms"]), float(u["base_adr_mxn"])) for uid, u in PORTFOLIO_UNITS.items()
    ]
    units += [(uid, city, beds, None) for uid, city, beds in _EXTRA_UNITS]
    return units


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def generate_panel(
    seed: int = 7,
    n_months: int = 48,
    start: str = "2022-01",
    units: Optional[List[Tuple[str, str, int, Optional[float]]]] = None,
    params: TrueParams = TrueParams(),
) -> pd.DataFrame:
    """Balanced synthetic panel, one row per unit-month.

    Columns: unit_id, city, bedrooms, year_month, month_num, t, days_in_month, adr_mxn,
             occupancy_rate, nights_occupied, p_true.
    `p_true` (latent occupancy probability) is kept only for diagnostics; it is never a feature.
    """
    units = units or default_units()
    rng = np.random.default_rng(seed)
    periods = pd.period_range(start=start, periods=n_months, freq="M")
    month_nums = np.array([p.month for p in periods])
    days = np.array([p.days_in_month for p in periods])

    season_occ = {m: float(_logit(np.array(p))) for m, p in BASE_SEASONAL_OCC_CURVE.items()}
    mean_season = float(np.mean(list(season_occ.values())))
    season_occ = {m: v - mean_season for m, v in season_occ.items()}
    season_adr = {m: float(np.log(v)) for m, v in BASE_SEASONAL_ADR_MULT.items()}
    mean_adr = float(np.mean(list(season_adr.values())))
    season_adr = {m: v - mean_adr for m, v in season_adr.items()}

    u = np.zeros(n_months)
    u[0] = rng.normal(0.0, params.shock_sd / np.sqrt(1.0 - params.shock_phi ** 2))
    for t in range(1, n_months):
        u[t] = params.shock_phi * u[t - 1] + rng.normal(0.0, params.shock_sd)

    rows = []
    for unit_id, city, bedrooms, base_adr in units:
        amp = _CITY_AMP[city]
        if base_adr is None:
            base_adr = _CITY_BASE_ADR_1BR[city] * (1.0 if bedrooms == 1 else 1.45) * float(np.exp(rng.normal(0.0, 0.08)))
        log_base = float(np.log(base_adr))
        occ_level = float(_logit(np.array(AIRDNA_PUBLIC_BENCHMARKS[city]["annual_mean_occupancy"])))
        a_i = occ_level + rng.normal(0.0, 0.25)
        for t in range(n_months):
            m = int(month_nums[t])
            ref = log_base + amp * season_adr[m] + params.trend_adr_per_month * t
            logp = ref + params.pass_through * u[t] + rng.normal(0.0, params.price_noise_sd)
            z = (a_i + amp * season_occ[m] + params.trend_occ_per_month * t + u[t]
                 + params.elasticity * (logp - ref) + rng.normal(0.0, params.occ_noise_sd))
            p_true = 1.0 / (1.0 + np.exp(-z))
            nights = int(rng.binomial(int(days[t]), p_true))
            rows.append(
                {
                    "unit_id": unit_id, "city": city, "bedrooms": bedrooms,
                    "year_month": str(periods[t]), "month_num": m, "t": t,
                    "days_in_month": int(days[t]),
                    "adr_mxn": round(float(np.exp(logp)), 2),
                    "occupancy_rate": nights / float(days[t]),
                    "nights_occupied": nights,
                    "p_true": float(p_true),
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["true_params"] = asdict(params)
    df.attrs["synthetic"] = True
    return df
