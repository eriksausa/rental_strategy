"""
Financial Decision Engine & Monte Carlo Cash-Flow Simulator (`src/rental_engine/engine.py`).

Unifies the core simulation logic used by tests and the Streamlit UI:
  - Uses exact calendar days per month (`calendar.monthrange`) rather than a fixed 30.0.
  - Pulls all modeling bounds, CFE parameters, and decision thresholds from `Assumptions`.
  - Supports both direct monthly CFE bill inputs and 12-month rolling kWh trajectory bills.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd

from .assumptions import (
    Assumptions,
    DEFAULT_ASSUMPTIONS,
    MONTH_NAMES_EN,
    MONTH_NAMES_ES,
    days_in_month as get_days_in_month,
)
from .cfe import evaluate_12m_dac_trajectory


@dataclass
class UserCostInputs:
    """Explicit user-entered property expenses and contract parameters."""
    unit_id: str
    contract_months: int                  # 3 or 6 months
    monthly_lease_offer_mxn: float        # Tenant's fixed monthly rent offer (MXN)
    hoa_monthly_mxn: float                # Monthly condo HOA / maintenance fee
    wifi_gas_water_monthly_mxn: float     # Monthly internet, water, and gas
    cleaning_cost_per_stay_mxn: float     # Unrecovered turnover/laundry cost per stay
    avg_stay_nights: float                # Average length of stay (nights per booking)
    airbnb_fee_pct: float                 # Platform + Q. Roo lodging tax % (e.g. 0.18)
    facebook_direct_share: float          # Share of short-term bookings from FB Groups (0.0 to 1.0)
    winter_cfe_monthly_mxn: float         # Expected Nov-Apr monthly electricity bill on STR
    summer_cfe_monthly_mxn: float         # Expected May-Oct monthly CFE DAC bill on STR


def compute_blended_ota_drag(airbnb_fee_pct: float, facebook_direct_share: float) -> float:
    """
    Facebook Group direct bookings carry 0% platform commission, reducing the
    effective platform drag proportionally to the direct booking share.
    """
    fb_share = float(np.clip(facebook_direct_share, 0.0, 1.0))
    ota_fee = float(np.clip(airbnb_fee_pct, 0.0, 0.50))
    return round((1.0 - fb_share) * ota_fee, 4)


def compute_break_even_occupancy(
    adr_mxn: float,
    monthly_lease_offer_mxn: float,
    costs: UserCostInputs,
    is_summer_month: bool,
    days_in_month: int = 30,
    explicit_cfe_bill_mxn: float | None = None,
) -> float:
    """
    Computes the monthly short-term occupancy rate (clamped to [0.0, 1.0]) required
    for short-term net income to equal the tenant's fixed lease offer (where the
    tenant pays CFE directly).
    """
    blended_fee = compute_blended_ota_drag(costs.airbnb_fee_pct, costs.facebook_direct_share)
    cleaning_per_occupied_night = costs.cleaning_cost_per_stay_mxn / max(costs.avg_stay_nights, 1.0)
    net_rev_per_occupied_night = adr_mxn * (1.0 - blended_fee) - cleaning_per_occupied_night
    if net_rev_per_occupied_night <= 0:
        return 1.0

    if explicit_cfe_bill_mxn is not None:
        cfe_bill = float(explicit_cfe_bill_mxn)
    else:
        cfe_bill = costs.summer_cfe_monthly_mxn if is_summer_month else costs.winter_cfe_monthly_mxn
    target_gross_contribution = monthly_lease_offer_mxn + cfe_bill
    required_nights = target_gross_contribution / net_rev_per_occupied_night
    raw_occ = required_nights / float(days_in_month)
    return float(np.clip(raw_occ, 0.0, 1.0))


def simulate_lease_vs_str(
    monthly_forecasts: List[Dict[str, float]],
    costs: UserCostInputs,
    n_simulations: int | None = None,
    random_seed: int = 42,
    lang: str = "en",
    reference_year: int | None = None,
    assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
    prior_11m_kwh: Sequence[float] | None = None,
    window_monthly_kwh: Sequence[float] | None = None,
    residual_blocks: np.ndarray | None = None,
) -> Dict[str, object]:
    """
    Runs a Monte Carlo cash-flow simulation over the 3M or 6M lease window.

    If `residual_blocks` (shape: n_blocks x n_months x 2) is given, each simulation draws one
    whole block of backtest forecast errors (log-ADR, log-odds occupancy) and applies it to the
    point forecasts. This keeps month-to-month persistence and the ADR/occupancy dependence seen
    in the backtest ("block bootstrap"). Without it, months are drawn as independent normals.

    Each entry in `monthly_forecasts` provides:
      - month_num (1..12)
      - pred_adr_mxn
      - pred_occ_rate (0..1)
      - adr_std_mxn (predictive uncertainty from chronological backtest residuals)
      - occ_std (predictive uncertainty from chronological backtest residuals)

    Uses exact calendar days per month (`calendar.monthrange(year, month)`).
    If `prior_11m_kwh` and `window_monthly_kwh` are supplied, computes each month's
    CFE bill via the 12-month rolling average DAC rule; otherwise uses the user's
    explicit `winter_cfe_monthly_mxn` / `summer_cfe_monthly_mxn` inputs.
    """
    n_runs = int(n_simulations if n_simulations is not None else assumptions.default_monte_carlo_runs)
    ref_year = int(reference_year if reference_year is not None else assumptions.default_reference_year)
    rng = np.random.default_rng(random_seed)
    n_months = len(monthly_forecasts)
    month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN

    # Lease Net Income: Tenant pays monthly rent + pays CFE meter directly.
    # Owner still pays HOA and Wi-Fi/Water/Gas.
    monthly_fixed_owner_costs = costs.hoa_monthly_mxn + costs.wifi_gas_water_monthly_mxn
    lease_monthly_net = costs.monthly_lease_offer_mxn - monthly_fixed_owner_costs
    lease_total_net_mxn = float(n_months * lease_monthly_net)

    blended_fee = compute_blended_ota_drag(costs.airbnb_fee_pct, costs.facebook_direct_share)
    cleaning_per_night = costs.cleaning_cost_per_stay_mxn / max(costs.avg_stay_nights, 1.0)

    cfe_trajectory = None
    if prior_11m_kwh is not None and window_monthly_kwh is not None and len(window_monthly_kwh) == n_months:
        cfe_trajectory = evaluate_12m_dac_trajectory(
            prior_11m_kwh=prior_11m_kwh,
            window_monthly_kwh=window_monthly_kwh,
            assumptions=assumptions,
        )

    use_blocks = residual_blocks is not None
    if use_blocks:
        residual_blocks = np.asarray(residual_blocks, dtype=float)
        if residual_blocks.ndim != 3 or residual_blocks.shape[1] != n_months or residual_blocks.shape[2] != 2:
            raise ValueError(
                f"residual_blocks must have shape (n_blocks, {n_months}, 2); got {residual_blocks.shape}."
            )
        block_idx = rng.integers(0, residual_blocks.shape[0], size=n_runs)

    sim_str_totals = np.zeros(n_runs, dtype=float)
    break_even_occs: List[float] = []
    monthly_breakdown: List[Dict[str, object]] = []

    current_year = ref_year
    prev_month = None

    for idx, m in enumerate(monthly_forecasts):
        month_num = int(m["month_num"])
        if prev_month is not None and month_num < prev_month:
            current_year += 1
        prev_month = month_num

        month_days = int(m.get("days_in_month", get_days_in_month(current_year, month_num)))
        month_name = str(m.get("month_name", month_map[month_num]))
        is_summer = month_num in assumptions.summer_months

        if cfe_trajectory is not None:
            cfe_bill = float(cfe_trajectory[idx]["estimated_bill_mxn"])
            in_dac_flag = bool(cfe_trajectory[idx]["in_dac"])
        else:
            cfe_bill = costs.summer_cfe_monthly_mxn if is_summer else costs.winter_cfe_monthly_mxn
            in_dac_flag = is_summer

        adr_std = max(float(m.get("adr_std_mxn", assumptions.fallback_adr_std_mxn)), 1.0)
        occ_std = max(float(m.get("occ_std", assumptions.fallback_occ_std)), 0.01)

        if use_blocks:
            res = residual_blocks[block_idx, idx, :]
            adr_samples = np.maximum(float(m["pred_adr_mxn"]) * np.exp(res[:, 0]), assumptions.min_sampled_adr_mxn)
            p0 = float(np.clip(m["pred_occ_rate"], 1e-4, 1.0 - 1e-4))
            occ_samples = np.clip(
                1.0 / (1.0 + np.exp(-(np.log(p0 / (1.0 - p0)) + res[:, 1]))),
                assumptions.min_sampled_occ_rate,
                assumptions.max_sampled_occ_rate,
            )
        else:
            adr_samples = np.maximum(
                rng.normal(loc=float(m["pred_adr_mxn"]), scale=adr_std, size=n_runs),
                assumptions.min_sampled_adr_mxn,
            )
            occ_samples = np.clip(
                rng.normal(loc=float(m["pred_occ_rate"]), scale=occ_std, size=n_runs),
                assumptions.min_sampled_occ_rate,
                assumptions.max_sampled_occ_rate,
            )

        occupied_nights = float(month_days) * occ_samples
        gross_rev = occupied_nights * adr_samples
        net_after_fees_and_cleaning = gross_rev * (1.0 - blended_fee) - (occupied_nights * cleaning_per_night)
        month_net_str = net_after_fees_and_cleaning - monthly_fixed_owner_costs - cfe_bill
        sim_str_totals += month_net_str

        be_occ = compute_break_even_occupancy(
            adr_mxn=float(m["pred_adr_mxn"]),
            monthly_lease_offer_mxn=costs.monthly_lease_offer_mxn,
            costs=costs,
            is_summer_month=is_summer,
            days_in_month=month_days,
            explicit_cfe_bill_mxn=cfe_bill,
        )
        break_even_occs.append(be_occ)

        if lang == "es":
            season_label = (
                "Verano (Riesgo CFE DAC)"
                if in_dac_flag
                else ("Invierno Alta" if month_num in assumptions.high_season_months else "Media")
            )
            monthly_breakdown.append(
                {
                    "Mes": f"{month_name} ({month_days}d)",
                    "Temporada": season_label,
                    "ADR Predicho (MXN)": round(float(m["pred_adr_mxn"]), 0),
                    "Ocupación Predicha (%)": round(float(m["pred_occ_rate"]) * 100.0, 1),
                    "Ocupación Equilibrio (%)": round(be_occ * 100.0, 1),
                    "Recibo CFE Renta Corta (MXN)": round(cfe_bill, 0),
                    "Neto Renta Corta Esperado (MXN)": round(float(np.mean(month_net_str)), 0),
                    "Neto Contrato Garantizado (MXN)": round(lease_monthly_net, 0),
                }
            )
        else:
            season_label = (
                "Summer / CFE DAC Risk"
                if in_dac_flag
                else ("High Winter" if month_num in assumptions.high_season_months else "Shoulder")
            )
            monthly_breakdown.append(
                {
                    "Month": f"{month_name} ({month_days}d)",
                    "Season": season_label,
                    "Pred ADR (MXN)": round(float(m["pred_adr_mxn"]), 0),
                    "Pred Occ (%)": round(float(m["pred_occ_rate"]) * 100.0, 1),
                    "Break-Even Occ (%)": round(be_occ * 100.0, 1),
                    "STR CFE Bill (MXN)": round(cfe_bill, 0),
                    "Expected STR Net (MXN)": round(float(np.mean(month_net_str)), 0),
                    "Guaranteed Lease Net (MXN)": round(lease_monthly_net, 0),
                }
            )

    p10_str = float(np.percentile(sim_str_totals, 10))
    p50_str = float(np.percentile(sim_str_totals, 50))
    p90_str = float(np.percentile(sim_str_totals, 90))
    mean_str = float(np.mean(sim_str_totals))

    cutoff_5pct = np.percentile(sim_str_totals, 5)
    cvar_5pct = float(np.mean(sim_str_totals[sim_str_totals <= cutoff_5pct]))
    prob_str_beats_lease = float(np.mean(sim_str_totals > lease_total_net_mxn))

    indifference_monthly_rent_mxn = (mean_str / max(n_months, 1)) + monthly_fixed_owner_costs
    mean_be_occ = float(np.mean(break_even_occs))

    if lease_total_net_mxn >= p50_str and prob_str_beats_lease < assumptions.accept_lease_max_str_win_prob:
        rec_code = "ACCEPT"
        recommendation = "ACEPTAR CONTRATO DE ARRENDAMIENTO" if lang == "es" else "ACCEPT CONTRACT"
        counter_offer_mxn = costs.monthly_lease_offer_mxn
        rationale = (
            f"El contrato garantizado de {n_months} meses genera **${lease_total_net_mxn:,.0f} MXN netos** "
            f"(el inquilino paga la luz CFE directamente), superando el escenario mediano de renta corta "
            f"(**${p50_str:,.0f} MXN**) sin riesgo de noches vacías ni cobros de tarifa CFE DAC."
            if lang == "es"
            else f"The guaranteed {n_months}-month lease nets **${lease_total_net_mxn:,.0f} MXN** "
            f"(tenant pays CFE electricity directly), beating the median short-term scenario "
            f"(**${p50_str:,.0f} MXN**) with zero vacancy or summer CFE DAC bill shock."
        )
    elif prob_str_beats_lease > assumptions.keep_open_min_str_win_prob:
        rec_code = "OPEN"
        recommendation = (
            "MANTENER CALENDARIO ABIERTO (AIRBNB / FACEBOOK DIRECTO)"
            if lang == "es"
            else "KEEP CALENDAR OPEN (STR / FB DIRECT)"
        )
        counter_offer_mxn = round(indifference_monthly_rent_mxn * assumptions.counter_offer_risk_premium, -2)
        rationale = (
            f"La renta corta supera esta oferta en el **{prob_str_beats_lease * 100:.1f}%** de las {n_runs:,} simulaciones Monte Carlo "
            f"(Mediana Renta Corta: **${p50_str:,.0f} MXN** vs. Contrato: **${lease_total_net_mxn:,.0f} MXN**). "
            f"Solo conviene firmar contrato si el inquilino sube su oferta al menos a **${counter_offer_mxn:,.0f} MXN/mes**."
            if lang == "es"
            else f"Short-term stays beat this lease offer in **{prob_str_beats_lease * 100:.1f}%** of {n_runs:,} Monte Carlo runs "
            f"(Median STR Net: **${p50_str:,.0f} MXN** vs. Lease Net: **${lease_total_net_mxn:,.0f} MXN**). "
            f"Only sign a lease if the tenant raises their offer to at least **${counter_offer_mxn:,.0f} MXN/month**."
        )
    else:
        rec_code = "NEGOTIATE"
        recommendation = "NEGOCIAR RENTA MENSUAL MÁS ALTA" if lang == "es" else "NEGOTIATE HIGHER LEASE RATE"
        counter_offer_mxn = round(
            max(indifference_monthly_rent_mxn, costs.monthly_lease_offer_mxn * assumptions.counter_offer_risk_premium),
            -2,
        )
        rationale = (
            f"Los flujos esperados están muy parejos (renta corta gana en el **{prob_str_beats_lease * 100:.1f}%** de las corridas), "
            f"pero la renta corta conlleva riesgo a la baja (5% CVaR: **${cvar_5pct:,.0f} MXN**). "
            f"Haz una contraoferta de **${counter_offer_mxn:,.0f} MXN/mes** (+ CFE pagado por el inquilino) para asegurar ganancia sin riesgo."
            if lang == "es"
            else f"Expected cash flows are close (STR wins in **{prob_str_beats_lease * 100:.1f}%** of runs), "
            f"but short-term carries downside risk (5% CVaR: **${cvar_5pct:,.0f} MXN**). "
            f"Counter-offer at **${counter_offer_mxn:,.0f} MXN/month** (+ CFE paid by tenant) to lock in guaranteed profit."
        )

    return {
        "uncertainty_method": "block_bootstrap" if use_blocks else "independent_normal",
        "sim_str_totals": sim_str_totals,
        "monthly_df": pd.DataFrame(monthly_breakdown),
        "cfe_trajectory": cfe_trajectory,
        "lease_total_net_mxn": round(lease_total_net_mxn, 2),
        "str_expected_net_mxn": round(mean_str, 2),
        "str_p10_net_mxn": round(p10_str, 2),
        "str_p50_net_mxn": round(p50_str, 2),
        "str_p90_net_mxn": round(p90_str, 2),
        "str_cvar_5pct_mxn": round(cvar_5pct, 2),
        "prob_str_beats_lease": round(prob_str_beats_lease, 4),
        "mean_break_even_occupancy": round(mean_be_occ, 4),
        "mean_break_even_occupancy_pct": round(mean_be_occ * 100.0, 1),
        "indifference_monthly_rent_mxn": round(indifference_monthly_rent_mxn, 2),
        "blended_platform_fee_pct": blended_fee,
        "rec_code": rec_code,
        "recommendation": recommendation,
        "rationale": rationale,
        "suggested_counter_offer_mxn": float(counter_offer_mxn),
    }
