"""
Rental Strategy Optimizer — Core Engine Package (`src/rental_engine`).

Modules:
  - `assumptions`: Explicit domain constants, CFE tariff limits, and cited AirDNA benchmarks.
  - `transforms`: Shared logit, inverse-logit, and Laplace occupancy smoothing helpers.
  - `cfe`: 12-month rolling-average CFE DAC electricity classifier and bill calculator.
  - `synthetic_panel`: Deterministic 12-unit x 48-month synthetic panel with known elasticity.
  - `forecasting`: Leak-free rolling-origin (walk-forward) backtest and OOF residual generator.
  - `elasticity`: Naive vs. fixed-effects price-response estimator with unit cluster bootstrap.
  - `forecast_service`: Loads saved artifacts and maps lease windows to point forecasts + residual blocks.
  - `engine`: Break-even occupancy, blended OTA fee drag, and Monte Carlo lease-vs-STR simulator.
  - `reporting`: Reproducible Markdown report builder for backtest and elasticity results.
  - `fb_parser`: Safe copy-paste Facebook Group rental listing parser.
"""

from .assumptions import (
    AIRDNA_PUBLIC_BENCHMARKS,
    Assumptions,
    CFE_TARIFF_DAC_LIMITS_KWH,
    DEFAULT_ASSUMPTIONS,
    PORTFOLIO_UNITS,
    days_in_month,
)
from .cfe import (
    compute_rolling_12m_avg_kwh,
    estimate_cfe_monthly_bill,
    evaluate_12m_dac_trajectory,
    is_dac_reclassified,
)
from .engine import (
    UserCostInputs,
    compute_blended_ota_drag,
    compute_break_even_occupancy,
    simulate_lease_vs_str,
)
from .fb_parser import parse_facebook_rental_post
from .transforms import inv_logit, logit, smooth_occupancy

__all__ = [
    "AIRDNA_PUBLIC_BENCHMARKS",
    "Assumptions",
    "CFE_TARIFF_DAC_LIMITS_KWH",
    "DEFAULT_ASSUMPTIONS",
    "PORTFOLIO_UNITS",
    "UserCostInputs",
    "compute_blended_ota_drag",
    "compute_break_even_occupancy",
    "compute_rolling_12m_avg_kwh",
    "days_in_month",
    "estimate_cfe_monthly_bill",
    "evaluate_12m_dac_trajectory",
    "inv_logit",
    "is_dac_reclassified",
    "logit",
    "parse_facebook_rental_post",
    "simulate_lease_vs_str",
    "smooth_occupancy",
]
