"""
Phase 3: Financial Decision Engine & Monte Carlo Cash-Flow Simulator.
All unit costs (HOA, cleaning, Wi-Fi, OTA fee %, Facebook direct share, and CFE
electricity) are explicit user inputsâ€”zero hidden cost assumptions.
"""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


# CFE Domestic High-Consumption (DAC) tariff constants in Quintana Roo
CFE_DAC_THRESHOLD_KWH = 400.0
CFE_DAC_BASE_RATE_MXN = 6.08
CFE_IVA_MULTIPLIER = 1.16
DEFAULT_MONTE_CARLO_RUNS = 5000


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


def estimate_cfe_monthly_bill(kwh_used: float, is_summer_month: bool) -> float:
    """
    Calculates monthly CFE electricity bill in MXN from kWh consumption.
    In May-Oct (summer), exceeding 400 kWh triggers the unsubsidized CFE DAC tariff
    plus 16% IVA tax.
    """
    if kwh_used <= 0:
        return 0.0
    if is_summer_month and kwh_used > CFE_DAC_THRESHOLD_KWH:
        return round(kwh_used * CFE_DAC_BASE_RATE_MXN * CFE_IVA_MULTIPLIER, 2)
    subsidized_rate = 1.85
    return round(kwh_used * subsidized_rate * CFE_IVA_MULTIPLIER, 2)


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

    cfe_bill = costs.summer_cfe_monthly_mxn if is_summer_month else costs.winter_cfe_monthly_mxn
    target_gross_contribution = monthly_lease_offer_mxn + cfe_bill
    required_nights = target_gross_contribution / net_rev_per_occupied_night
    raw_occ = required_nights / float(days_in_month)
    return float(np.clip(raw_occ, 0.0, 1.0))


def simulate_lease_vs_str(
    monthly_forecasts: List[Dict[str, float]],
    costs: UserCostInputs,
    n_simulations: int = DEFAULT_MONTE_CARLO_RUNS,
    random_seed: int = 42,
) -> Dict[str, float | str]:
    """
    Runs a calibrated Monte Carlo cash-flow simulation over the exact 3M or 6M lease window.
    Each entry in `monthly_forecasts` provides:
      - month_num (1..12)
      - pred_adr_mxn
      - pred_occ_rate (0..1)
      - adr_std_mxn (predictive uncertainty from backtesting)
      - occ_std (predictive uncertainty from backtesting)
    """
    rng = np.random.default_rng(random_seed)
    n_months = len(monthly_forecasts)

    # Lease Net Income: Tenant pays monthly rent + pays CFE meter directly.
    # Owner still pays HOA and Wi-Fi/Water/Gas.
    monthly_fixed_owner_costs = costs.hoa_monthly_mxn + costs.wifi_gas_water_monthly_mxn
    lease_total_net_mxn = float(
        n_months * (costs.monthly_lease_offer_mxn - monthly_fixed_owner_costs)
    )

    blended_fee = compute_blended_ota_drag(costs.airbnb_fee_pct, costs.facebook_direct_share)
    cleaning_per_night = costs.cleaning_cost_per_stay_mxn / max(costs.avg_stay_nights, 1.0)

    sim_str_totals = np.zeros(n_simulations, dtype=float)
    break_even_occs = []

    for m in monthly_forecasts:
        month_num = int(m["month_num"])
        is_summer = month_num in (5, 6, 7, 8, 9, 10)
        cfe_bill = costs.summer_cfe_monthly_mxn if is_summer else costs.winter_cfe_monthly_mxn

        adr_samples = rng.normal(loc=m["pred_adr_mxn"], scale=max(m.get("adr_std_mxn", 180.0), 1.0), size=n_simulations)
        adr_samples = np.maximum(adr_samples, 300.0)

        occ_samples = rng.normal(loc=m["pred_occ_rate"], scale=max(m.get("occ_std", 0.08), 0.01), size=n_simulations)
        occ_samples = np.clip(occ_samples, 0.05, 0.98)

        occupied_nights = 30.0 * occ_samples
        gross_rev = occupied_nights * adr_samples
        net_after_fees_and_cleaning = gross_rev * (1.0 - blended_fee) - (occupied_nights * cleaning_per_night)
        month_net_str = net_after_fees_and_cleaning - monthly_fixed_owner_costs - cfe_bill
        sim_str_totals += month_net_str

        be_occ = compute_break_even_occupancy(
            adr_mxn=m["pred_adr_mxn"],
            monthly_lease_offer_mxn=costs.monthly_lease_offer_mxn,
            costs=costs,
            is_summer_month=is_summer,
        )
        break_even_occs.append(be_occ)

    p10_str = float(np.percentile(sim_str_totals, 10))
    p50_str = float(np.percentile(sim_str_totals, 50))
    p90_str = float(np.percentile(sim_str_totals, 90))
    mean_str = float(np.mean(sim_str_totals))

    cutoff_5pct = np.percentile(sim_str_totals, 5)
    cvar_5pct = float(np.mean(sim_str_totals[sim_str_totals <= cutoff_5pct]))
    prob_str_beats_lease = float(np.mean(sim_str_totals > lease_total_net_mxn))

    # Indifference Monthly Rent: monthly lease rate where Lease Net == Expected STR Net
    indifference_monthly_rent_mxn = (mean_str / max(n_months, 1)) + monthly_fixed_owner_costs

    if lease_total_net_mxn >= p50_str and prob_str_beats_lease < 0.45:
        recommendation = "ACCEPT CONTRACT"
        counter_offer_mxn = costs.monthly_lease_offer_mxn
    elif prob_str_beats_lease > 0.65:
        recommendation = "KEEP CALENDAR OPEN (STR / FB DIRECT)"
        counter_offer_mxn = round(indifference_monthly_rent_mxn * 1.05, -2)
    else:
        recommendation = "NEGOTIATE HIGHER LEASE RATE"
        counter_offer_mxn = round(max(indifference_monthly_rent_mxn, costs.monthly_lease_offer_mxn * 1.05), -2)

    return {
        "lease_total_net_mxn": round(lease_total_net_mxn, 2),
        "str_expected_net_mxn": round(mean_str, 2),
        "str_p10_net_mxn": round(p10_str, 2),
        "str_p50_net_mxn": round(p50_str, 2),
        "str_p90_net_mxn": round(p90_str, 2),
        "str_cvar_5pct_mxn": round(cvar_5pct, 2),
        "prob_str_beats_lease": round(prob_str_beats_lease, 4),
        "mean_break_even_occupancy": round(float(np.mean(break_even_occs)), 4),
        "indifference_monthly_rent_mxn": round(indifference_monthly_rent_mxn, 2),
        "blended_platform_fee_pct": blended_fee,
        "recommendation": recommendation,
        "suggested_counter_offer_mxn": float(counter_offer_mxn),
    }