"""
Reproduce the comparison of uncertainty methods for one lease decision (synthetic panel).

    python scripts/demo_uncertainty.py

Shows how the short-term-rental net distribution changes under:
  1. independent normal draws per month (marginal sd only),
  2. block bootstrap of backtest residuals (full dependence),
  3. block bootstrap with months shuffled (keeps ADR/occupancy dependence, removes persistence),
  4. block bootstrap with ADR and occupancy decoupled (keeps persistence, removes their dependence).

Requires artifacts/ and reports/ from scripts/run_forecast_pipeline.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rental_engine import UserCostInputs, forecast_service, simulate_lease_vs_str  # noqa: E402


def main(unit: str = "UNIT_A_PLAYA", start: str = "2026-05", months: int = 6, n: int = 8000) -> None:
    art = forecast_service.load_artifacts(ROOT)
    monthly, blocks = forecast_service.forecast_window(art, unit, start, months)
    costs = UserCostInputs(
        unit_id=unit, contract_months=months, monthly_lease_offer_mxn=26000.0, hoa_monthly_mxn=3100.0,
        wifi_gas_water_monthly_mxn=1100.0, cleaning_cost_per_stay_mxn=700.0, avg_stay_nights=4.0,
        airbnb_fee_pct=0.18, facebook_direct_share=0.60, winter_cfe_monthly_mxn=1600.0, summer_cfe_monthly_mxn=6000.0,
    )
    rng = np.random.default_rng(1)
    n_blocks, length, _ = blocks.shape
    adr, occ = blocks[:, :, 0], blocks[:, :, 1]
    print(f"Unit {unit}, {months}-month window starting {start}; {n_blocks} residual blocks")
    print(f"corr(ADR residual, occupancy residual) within month: {np.corrcoef(adr.ravel(), occ.ravel())[0, 1]:+.2f}")
    print(
        "lag-1 corr across months: ADR "
        f"{np.mean([np.corrcoef(adr[:, i], adr[:, i + 1])[0, 1] for i in range(length - 1)]):+.2f}, occupancy "
        f"{np.mean([np.corrcoef(occ[:, i], occ[:, i + 1])[0, 1] for i in range(length - 1)]):+.2f}\n"
    )
    shuffled = np.stack([blocks[rng.integers(0, n_blocks, size=n_blocks), i, :] for i in range(length)], axis=1)
    decoupled = blocks.copy()
    decoupled[:, :, 1] = blocks[rng.permutation(n_blocks), :, 1]
    variants = {
        "independent normal (marginal sd only)": simulate_lease_vs_str(monthly, costs, n_simulations=n),
        "block bootstrap (full dependence)": simulate_lease_vs_str(monthly, costs, n_simulations=n, residual_blocks=blocks),
        "bootstrap, months shuffled": simulate_lease_vs_str(monthly, costs, n_simulations=n, residual_blocks=shuffled),
        "bootstrap, ADR/occ decoupled": simulate_lease_vs_str(monthly, costs, n_simulations=n, residual_blocks=decoupled),
    }
    print(f"{'method':40s} {'P10 (MXN)':>10s} {'P50':>9s} {'P90':>9s} {'P10-P90 width':>14s}  recommendation")
    for name, r in variants.items():
        p10, p50, p90 = r["str_p10_net_mxn"], r["str_p50_net_mxn"], r["str_p90_net_mxn"]
        print(f"{name:40s} {p10:10,.0f} {p50:9,.0f} {p90:9,.0f} {p90 - p10:14,.0f}  {r['recommendation']}")
    print(f"\nLease net over the window: {variants['block bootstrap (full dependence)']['lease_total_net_mxn']:,.0f} MXN")


if __name__ == "__main__":
    main()
