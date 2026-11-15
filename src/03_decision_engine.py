"""
Phase 3 Re-export Wrapper: Financial Decision Engine, CFE DAC Rolling Average & Monte Carlo Simulator.
Delegates all logic to `src.rental_engine`.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rental_engine.assumptions import (  # noqa: E402
    Assumptions,
    CFE_TARIFF_DAC_LIMITS_KWH,
    DEFAULT_ASSUMPTIONS,
    days_in_month,
)
from src.rental_engine.cfe import (  # noqa: E402
    compute_rolling_12m_avg_kwh,
    estimate_cfe_monthly_bill,
    evaluate_12m_dac_trajectory,
    is_dac_reclassified,
)
from src.rental_engine.engine import (  # noqa: E402
    UserCostInputs,
    compute_blended_ota_drag,
    compute_break_even_occupancy,
    simulate_lease_vs_str,
)

__all__ = [
    "Assumptions",
    "CFE_TARIFF_DAC_LIMITS_KWH",
    "DEFAULT_ASSUMPTIONS",
    "UserCostInputs",
    "compute_blended_ota_drag",
    "compute_break_even_occupancy",
    "compute_rolling_12m_avg_kwh",
    "days_in_month",
    "estimate_cfe_monthly_bill",
    "evaluate_12m_dac_trajectory",
    "is_dac_reclassified",
    "simulate_lease_vs_str",
]
