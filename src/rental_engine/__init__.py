"""
Rental Strategy Optimizer — Core Engine Package (`src/rental_engine`).

Provides:
  - `assumptions`: Explicit, cited domain and simulation constants + public AirDNA benchmarks.
  - `cfe`: 12-month rolling-average CFE DAC electricity classification and billing.
  - `data`: Synthetic SQLite builder & optional AirROI cached market adapter.
  - `models`: Chronological train/test split, Baseline / RandomForest / XGBoost training & residual backtesting.
  - `engine`: Break-even occupancy, blended OTA fee drag, and Monte Carlo lease-vs-STR simulator.
  - `fb_parser`: Copy-paste Facebook Group rental listing parser.
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
    "is_dac_reclassified",
    "parse_facebook_rental_post",
    "simulate_lease_vs_str",
]
