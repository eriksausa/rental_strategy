"""
Financial Decision Engine & Monte Carlo Cash-Flow Simulator (`src/rental_engine/engine.py`).

Evaluates a guaranteed 3M/6M lease offer against short-term rentals (Airbnb + 0%-fee
Facebook Group direct bookings) over exact calendar month lengths (`calendar.monthrange`):
  - Pulls all sampling bounds, CFE parameters, and decision thresholds from `Assumptions`.
  - Supports both direct monthly CFE bill inputs and 12-month rolling kWh trajectory bills.
  - Supports both dependent block-bootstrap uncertainty (from walk-forward OOF residuals)
    and independent normal draws (marginal spreads) for apples-to-apples comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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
from .transforms import inv_logit, logit


@dataclass
class UserCostInputs:
    """Explicit user-entered property operating expenses and lease offer parameters."""

    unit_id: str
    contract_months: int                  # 3 or 6 months
    monthly_lease_offer_mxn: float        # Tenant's fixed monthly rent offer (MXN)
    hoa_monthly_mxn: float                # Monthly condo HOA / maintenance fee (owner pays in both cases)
    wifi_gas_water_monthly_mxn: float     # Monthly internet, water, and gas (owner pays in both cases)
    cleaning_cost_per_stay_mxn: float     # Unrecovered turnover/laundry cost per STR booking
    avg_stay_nights: float                # Average length of stay (nights per booking)
    airbnb_fee_pct: float                 # Airbnb platform + Q. Roo lodging tax drag (0.0..0.50)
    facebook_direct_share: float          # Share of STR bookings via 0%-fee Facebook Groups (0.0..1.0)
    winter_cfe_monthly_mxn: float         # Expected Nov-Apr monthly electricity bill on STR
    summer_cfe_monthly_mxn: float         # Expected May-Oct monthly CFE DAC bill on STR


def compute_blended_ota_drag(airbnb_fee_pct: float, facebook_direct_share: float) -> float:
    """Compute effective platform fee after accounting for 0%-commission Facebook direct bookings."""
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
    """Compute the monthly STR occupancy rate in `[0.0, 1.0]` needed to match a fixed lease offer.

    Because the owner pays HOA and Wi-Fi/Water/Gas under *both* strategies, those fixed costs
    cancel out. Short-term stays must cover both the foregone monthly lease rent AND the CFE
    electricity bill (which a mid-term tenant pays directly).
    """
    blended_fee = compute_blended_ota_drag(costs.airbnb_fee_pct, costs.facebook_direct_share)
    cleaning_per_occupied_night = costs.cleaning_cost_per_stay_mxn / max(costs.avg_stay_nights, 1.0)
    net_rev_per_occupied_night = adr_mxn * (1.0 - blended_fee) - cleaning_per_occupied_night
    if net_rev_per_occupied_night <= 0:
        return 1.0

    cfe_bill = (
        float(explicit_cfe_bill_mxn)
        if explicit_cfe_bill_mxn is not None
        else (costs.summer_cfe_monthly_mxn if is_summer_month else costs.winter_cfe_monthly_mxn)
    )
    target_gross_contribution = monthly_lease_offer_mxn + cfe_bill
    required_nights = target_gross_contribution / net_rev_per_occupied_night
    return float(np.clip(required_nights / float(days_in_month), 0.0, 1.0))


def _sample_monthly_adr_and_occ(
    month_forecast: Dict[str, float],
    month_idx: int,
    n_runs: int,
    rng: np.random.Generator,
    assumptions: Assumptions,
    residual_blocks: np.ndarray | None,
    block_idx: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw `n_runs` simulated `(ADR, occupancy)` pairs for a single month in the window."""
    pred_adr = float(month_forecast["pred_adr_mxn"])
    pred_occ = float(month_forecast["pred_occ_rate"])

    if residual_blocks is not None and block_idx is not None:
        # Block bootstrap: log-ADR residuals apply multiplicatively (exp(res_adr)), while
        # logit-occupancy residuals add in log-odds space before mapping back via sigmoid.
        res = residual_blocks[block_idx, month_idx, :]
        adr_samples = np.maximum(pred_adr * np.exp(res[:, 0]), assumptions.min_sampled_adr_mxn)
        occ_log_odds = float(logit(np.clip(pred_occ, 1e-4, 1.0 - 1e-4))) + res[:, 1]
        occ_samples = np.clip(
            inv_logit(occ_log_odds),
            assumptions.min_sampled_occ_rate,
            assumptions.max_sampled_occ_rate,
        )
        return adr_samples, occ_samples

    # Independent normal fallback: uses marginal spreads on level scales.
    adr_std = max(float(month_forecast.get("adr_std_mxn", assumptions.fallback_adr_std_mxn)), 1.0)
    occ_std = max(float(month_forecast.get("occ_std", assumptions.fallback_occ_std)), 0.01)
    adr_samples = np.maximum(
        rng.normal(loc=pred_adr, scale=adr_std, size=n_runs),
        assumptions.min_sampled_adr_mxn,
    )
    occ_samples = np.clip(
        rng.normal(loc=pred_occ, scale=occ_std, size=n_runs),
        assumptions.min_sampled_occ_rate,
        assumptions.max_sampled_occ_rate,
    )
    return adr_samples, occ_samples


def _format_monthly_row(
    month_num: int,
    month_name: str,
    month_days: int,
    in_dac_flag: bool,
    pred_adr: float,
    pred_occ: float,
    be_occ: float,
    cfe_bill: float,
    expected_str_net: float,
    lease_monthly_net: float,
    assumptions: Assumptions,
    lang: str,
) -> Dict[str, object]:
    """Build one localized row for the month-by-month comparison table."""
    is_high = month_num in assumptions.high_season_months
    if lang == "es":
        season_label = (
            "Verano (Riesgo CFE DAC)" if in_dac_flag else ("Invierno Alta" if is_high else "Media")
        )
        return {
            "Mes": f"{month_name} ({month_days}d)",
            "Temporada": season_label,
            "ADR Predicho (MXN)": round(pred_adr, 0),
            "Ocupación Predicha (%)": round(pred_occ * 100.0, 1),
            "Ocupación Equilibrio (%)": round(be_occ * 100.0, 1),
            "Recibo CFE Renta Corta (MXN)": round(cfe_bill, 0),
            "Neto Renta Corta Esperado (MXN)": round(expected_str_net, 0),
            "Neto Contrato Garantizado (MXN)": round(lease_monthly_net, 0),
        }

    season_label = (
        "Summer / CFE DAC Risk" if in_dac_flag else ("High Winter" if is_high else "Shoulder")
    )
    return {
        "Month": f"{month_name} ({month_days}d)",
        "Season": season_label,
        "Pred ADR (MXN)": round(pred_adr, 0),
        "Pred Occ (%)": round(pred_occ * 100.0, 1),
        "Break-Even Occ (%)": round(be_occ * 100.0, 1),
        "STR CFE Bill (MXN)": round(cfe_bill, 0),
        "Expected STR Net (MXN)": round(expected_str_net, 0),
        "Guaranteed Lease Net (MXN)": round(lease_monthly_net, 0),
    }


def _classify_decision(
    n_months: int,
    n_runs: int,
    lease_total_net_mxn: float,
    p50_str: float,
    cvar_5pct: float,
    prob_str_beats_lease: float,
    indifference_monthly_rent_mxn: float,
    monthly_lease_offer_mxn: float,
    assumptions: Assumptions,
    lang: str,
) -> Tuple[str, str, float, str]:
    """Map Monte Carlo summary statistics to `(rec_code, recommendation, counter_offer_mxn, rationale)`."""
    if lease_total_net_mxn >= p50_str and prob_str_beats_lease < assumptions.accept_lease_max_str_win_prob:
        rec_code = "ACCEPT"
        recommendation = "ACEPTAR CONTRATO DE ARRENDAMIENTO" if lang == "es" else "ACCEPT CONTRACT"
        counter_offer_mxn = monthly_lease_offer_mxn
        rationale = (
            f"El contrato garantizado de {n_months} meses genera **${lease_total_net_mxn:,.0f} MXN netos** "
            f"(el inquilino paga la luz CFE directamente), superando el escenario mediano de renta corta "
            f"(**${p50_str:,.0f} MXN**) sin riesgo de noches vacías ni cobros de tarifa CFE DAC."
            if lang == "es"
            else f"The guaranteed {n_months}-month lease nets **${lease_total_net_mxn:,.0f} MXN** "
            f"(tenant pays CFE electricity directly), beating the median short-term scenario "
            f"(**${p50_str:,.0f} MXN**) with zero vacancy or summer CFE DAC bill shock."
        )
        return rec_code, recommendation, float(counter_offer_mxn), rationale

    if prob_str_beats_lease > assumptions.keep_open_min_str_win_prob:
        rec_code = "OPEN"
        recommendation = (
            "MANTENER CALENDARIO ABIERTO (AIRBNB / FACEBOOK DIRECTO)"
            if lang == "es"
            else "KEEP CALENDAR OPEN (STR / FB DIRECT)"
        )
        # Round counter-offer to the nearest $100 MXN above indifference rent + risk markup.
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
        return rec_code, recommendation, float(counter_offer_mxn), rationale

    rec_code = "NEGOTIATE"
    recommendation = "NEGOCIAR RENTA MENSUAL MÁS ALTA" if lang == "es" else "NEGOTIATE HIGHER LEASE RATE"
    counter_offer_mxn = round(
        max(indifference_monthly_rent_mxn, monthly_lease_offer_mxn * assumptions.counter_offer_risk_premium),
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
    return rec_code, recommendation, float(counter_offer_mxn), rationale


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
    """Run a Monte Carlo cash-flow simulation comparing a fixed lease against short-term stays.

    Args:
        monthly_forecasts: Sequence of per-month forecast dicts containing `month_num` (1..12),
            `pred_adr_mxn`, `pred_occ_rate` (0..1), and optional `adr_std_mxn` / `occ_std`.
        costs: Property operating costs and lease offer inputs.
        n_simulations: Monte Carlo draws (defaults to `assumptions.default_monte_carlo_runs`).
        random_seed: RNG seed for reproducible quantiles and win probabilities.
        lang: Output language (`"en"` or `"es"`) for tables and rationale copy.
        reference_year: Starting calendar year used for exact `calendar.monthrange` day counts.
        assumptions: Centralized CFE, sampling-bound, and decision-threshold constants.
        prior_11m_kwh: Optional 11-month trailing meter history to activate 12M rolling CFE DAC mode.
        window_monthly_kwh: Optional per-month STR kWh usage for 12M rolling CFE DAC mode.
        residual_blocks: Optional `(n_blocks, n_months, 2)` array of OOF `(log_adr, logit_occ)`
            residuals to enable dependent block-bootstrap sampling.

    Returns:
        Dictionary of simulation totals, risk quantiles (`P10`, `P50`, `P90`, `5% CVaR`),
        monthly breakdown DataFrame, optional CFE trajectory, and decision recommendation.
    """
    n_runs = int(n_simulations if n_simulations is not None else assumptions.default_monte_carlo_runs)
    ref_year = int(reference_year if reference_year is not None else assumptions.default_reference_year)
    rng = np.random.default_rng(random_seed)
    n_months = len(monthly_forecasts)
    month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN

    # Under a mid-term lease, the tenant pays rent + CFE directly; the owner still covers HOA & utilities.
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
    block_idx = None
    if use_blocks:
        residual_blocks = np.asarray(residual_blocks, dtype=float)
        if residual_blocks.ndim != 3 or residual_blocks.shape[1] != n_months or residual_blocks.shape[2] != 2:
            raise ValueError(
                f"residual_blocks must have shape (n_blocks, {n_months}, 2); got {residual_blocks.shape}."
            )
        # Draw one block index per simulation so all months in a run share one contiguous error path.
        block_idx = rng.integers(0, residual_blocks.shape[0], size=n_runs)

    sim_str_totals = np.zeros(n_runs, dtype=float)
    break_even_occs: List[float] = []
    monthly_breakdown: List[Dict[str, object]] = []

    current_year = ref_year
    prev_month: int | None = None

    for idx, m in enumerate(monthly_forecasts):
        month_num = int(m["month_num"])
        # Advance the calendar year when a window crosses December -> January (e.g. Nov..Apr).
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

        adr_samples, occ_samples = _sample_monthly_adr_and_occ(
            month_forecast=m,
            month_idx=idx,
            n_runs=n_runs,
            rng=rng,
            assumptions=assumptions,
            residual_blocks=residual_blocks if use_blocks else None,
            block_idx=block_idx,
        )

        occupied_nights = float(month_days) * occ_samples
        gross_rev = occupied_nights * adr_samples
        net_after_variable = gross_rev * (1.0 - blended_fee) - (occupied_nights * cleaning_per_night)
        month_net_str = net_after_variable - monthly_fixed_owner_costs - cfe_bill
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
        monthly_breakdown.append(
            _format_monthly_row(
                month_num=month_num,
                month_name=month_name,
                month_days=month_days,
                in_dac_flag=in_dac_flag,
                pred_adr=float(m["pred_adr_mxn"]),
                pred_occ=float(m["pred_occ_rate"]),
                be_occ=be_occ,
                cfe_bill=cfe_bill,
                expected_str_net=float(np.mean(month_net_str)),
                lease_monthly_net=lease_monthly_net,
                assumptions=assumptions,
                lang=lang,
            )
        )

    p10_str, p50_str, p90_str = (float(x) for x in np.percentile(sim_str_totals, [10, 50, 90]))
    mean_str = float(np.mean(sim_str_totals))

    # Conditional Value-at-Risk (5% CVaR): average net cash flow in the worst 5% of runs.
    cutoff_5pct = np.percentile(sim_str_totals, 5)
    cvar_5pct = float(np.mean(sim_str_totals[sim_str_totals <= cutoff_5pct]))
    prob_str_beats_lease = float(np.mean(sim_str_totals > lease_total_net_mxn))

    # Gross monthly lease rent that makes expected lease net equal expected STR net:
    #   n_months * (R_indiff - fixed_costs) = mean_str  =>  R_indiff = mean_str / n_months + fixed_costs
    indifference_monthly_rent_mxn = (mean_str / max(n_months, 1)) + monthly_fixed_owner_costs
    mean_be_occ = float(np.mean(break_even_occs))

    rec_code, recommendation, counter_offer_mxn, rationale = _classify_decision(
        n_months=n_months,
        n_runs=n_runs,
        lease_total_net_mxn=lease_total_net_mxn,
        p50_str=p50_str,
        cvar_5pct=cvar_5pct,
        prob_str_beats_lease=prob_str_beats_lease,
        indifference_monthly_rent_mxn=indifference_monthly_rent_mxn,
        monthly_lease_offer_mxn=costs.monthly_lease_offer_mxn,
        assumptions=assumptions,
        lang=lang,
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
        "suggested_counter_offer_mxn": counter_offer_mxn,
    }
