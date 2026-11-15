"""
CFE Domestic High-Consumption (Tarifa DAC) 12-Month Rolling Average Model.

Official CFE Rule (Comisión Federal de Electricidad — Tarifa DAC, accessed 2026-11-15):
  https://app.cfe.mx/Aplicaciones/CCFE/Tarifas/TarifasCRECasa/Casa.aspx
  - A domestic service meter is reclassified into the unsubsidized DAC tariff
    when the **rolling 12-month average** monthly electricity consumption
    (last 365 billed days / 6 bimonthly bills) exceeds the monthly high-consumption
    limit for the locality's domestic tariff:
      * Tariff 1  :   250 kWh/month
      * Tariff 1A :   300 kWh/month
      * Tariff 1B :   400 kWh/month  (default for this project)
      * Tariff 1C :   850 kWh/month
      * Tariff 1D : 1,000 kWh/month
      * Tariff 1E : 2,000 kWh/month
      * Tariff 1F : 2,500 kWh/month
  - Once reclassified into DAC, **every month** (including winter months) is
    billed at the unsubsidized DAC rate until the trailing 12-month rolling
    average drops back at or below the threshold.
  - Note for mid-term leases: Even when a 3M/6M lease tenant pays CFE directly
    via meter readings, their kWh usage remains on the meter's 12-month trailing
    record and can push subsequent short-term rental months into DAC.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np

from .assumptions import Assumptions, DEFAULT_ASSUMPTIONS


def compute_rolling_12m_avg_kwh(monthly_kwh_history: Sequence[float]) -> float:
    """
    Computes the trailing 12-month average monthly consumption in kWh.
    If fewer than 12 months are provided, averages over the available trailing
    months (up to 12). Returns 0.0 for an empty sequence.
    """
    if not monthly_kwh_history:
        return 0.0
    window = [max(float(x), 0.0) for x in list(monthly_kwh_history)[-12:]]
    return round(float(np.mean(window)), 2)


def is_dac_reclassified(
    monthly_kwh_history: Sequence[float],
    tariff: str | None = None,
    assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
) -> bool:
    """
    Returns True if the trailing 12-month average monthly consumption strictly
    exceeds the CFE DAC threshold for `tariff` (400 kWh/month for Tariff 1B).
    """
    limit_kwh = assumptions.dac_limit_for_tariff(tariff)
    rolling_avg = compute_rolling_12m_avg_kwh(monthly_kwh_history)
    return bool(rolling_avg > limit_kwh)


def estimate_cfe_monthly_bill(
    kwh_used: float,
    is_summer_month: bool = True,
    *,
    trailing_12m_kwh: Sequence[float] | None = None,
    dac_status: bool | None = None,
    include_fixed_charge: bool = False,
    tariff: str | None = None,
    assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
) -> float:
    """
    Calculates a monthly CFE electricity bill in MXN (including 16% IVA).

    How DAC status is determined (in priority order):
      1. If `dac_status` is explicitly passed (`True` or `False`), uses it directly.
      2. If `trailing_12m_kwh` is provided, applies the official CFE 12-month
         rolling average rule (`mean(trailing_12m_kwh[-12:]) > limit`). Once in
         DAC, the unsubsidized rate applies in both summer and winter until the
         12-month rolling average drops back to or below the limit.
      3. Fallback single-month approximation (when no history is passed): checks
         whether `is_summer_month` is True and `kwh_used` exceeds the tariff limit.
    """
    if kwh_used <= 0:
        return 0.0

    limit_kwh = assumptions.dac_limit_for_tariff(tariff)
    if dac_status is not None:
        in_dac = bool(dac_status)
    elif trailing_12m_kwh is not None and len(trailing_12m_kwh) > 0:
        in_dac = is_dac_reclassified(trailing_12m_kwh, tariff=tariff, assumptions=assumptions)
    else:
        in_dac = bool(is_summer_month and float(kwh_used) > limit_kwh)

    if in_dac:
        fixed = assumptions.cfe_dac_fixed_charge_mxn if include_fixed_charge else 0.0
        subtotal = fixed + float(kwh_used) * assumptions.cfe_dac_rate_mxn_per_kwh
        return round(subtotal * assumptions.cfe_iva_multiplier, 2)

    subtotal = float(kwh_used) * assumptions.cfe_subsidized_rate_mxn_per_kwh
    return round(subtotal * assumptions.cfe_iva_multiplier, 2)


def evaluate_12m_dac_trajectory(
    prior_11m_kwh: Sequence[float],
    window_monthly_kwh: Sequence[float],
    tariff: str | None = None,
    include_fixed_charge: bool = False,
    assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
) -> List[Dict[str, float | bool]]:
    """
    Steps month-by-month through `window_monthly_kwh`, updating the 12-month
    rolling kWh window at each step and computing whether the meter is in DAC
    and the resulting monthly bill in MXN.
    """
    history = [max(float(x), 0.0) for x in prior_11m_kwh]
    limit_kwh = assumptions.dac_limit_for_tariff(tariff)
    trajectory: List[Dict[str, float | bool]] = []

    for kwh in window_monthly_kwh:
        kwh_val = max(float(kwh), 0.0)
        history.append(kwh_val)
        rolling_avg = compute_rolling_12m_avg_kwh(history)
        in_dac = bool(rolling_avg > limit_kwh)
        bill_mxn = estimate_cfe_monthly_bill(
            kwh_val,
            dac_status=in_dac,
            include_fixed_charge=include_fixed_charge,
            tariff=tariff,
            assumptions=assumptions,
        )
        trajectory.append(
            {
                "kwh_used": round(kwh_val, 1),
                "rolling_12m_avg_kwh": rolling_avg,
                "dac_threshold_kwh": limit_kwh,
                "in_dac": in_dac,
                "estimated_bill_mxn": bill_mxn,
            }
        )
    return trajectory
