"""
Riviera Maya (Playa del Carmen & Tulum) + CDMX Short-Term (Airbnb) vs. Mid/Long-Term (3M / 6M Lease)
Stochastic Yield, CFE DAC Electricity & Seasonal Regime-Switching Optimizer (`rental_strategy_optimizer.py`).

Calibrated to 2026 Market Data:
  - Playa del Carmen (AirDNA 2026: ~6,500 active listings, 58% mean occupancy, $103 ADR, $59 RevPAR)
  - Tulum (AirDNA 2026: ~3,609 active listings, 46% mean occupancy, $125 ADR, $57-$58 RevPAR, 54/100 seasonality)
  - Quintana Roo CFE DAC Tariff & Tax Mechanics (Villas Ambar / StayAtPlaya 2026):
      * Above 400 kWh/month (~13 kWh/day), CFE Tariff 1B switches to unsubsidized DAC (~6.08 MXN/kWh + IVA,
        or ~$260-$380 USD/month in summer A/C electricity absorbed by Airbnb hosts).
      * On 3-month and 6-month leases in Playa del Carmen & Tulum, the TENANT pays CFE electricity via
        check-in/check-out meter readings, eliminating the owner's summer A/C bill shock.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# =============================================================================
# 1. MARKET-CALIBRATED CONFIGURATIONS (PLAYA DEL CARMEN, TULUM, CDMX)
# =============================================================================

@dataclass
class PropertyEconomicsConfig:
    """Unit economics and monthly seasonal curves for Riviera Maya (Playa / Tulum) & CDMX."""
    market_name: str
    base_str_adr_usd: float
    monthly_6m_lease_usd: float
    monthly_3m_lease_usd: float
    ota_and_lodging_tax_pct: float               # Platform fee (~15.5%) + unrecovered QRoo lodging tax drag
    avg_str_stay_nights: float
    net_cleaning_turnover_cost_usd: float        # Unrecovered cleaning, laundry, AC/pool servicing per stay
    str_owner_utilities_winter_usd: float        # Owner-paid Wi-Fi, water, gas + moderate winter CFE (Nov-Apr)
    str_owner_utilities_summer_dac_usd: float    # Owner-paid utilities + heavy summer A/C CFE DAC tariff (May-Oct)
    lease_owner_utilities_monthly_usd: float     # Owner pays only HOA-included water/Wi-Fi; TENANT pays CFE meter!
    fixed_hoa_predial_monthly_usd: float         # Condominium HOA, pool/security, predial & insurance
    mean_occ_by_month: List[float] = field(default_factory=list)
    adr_mult_by_month: List[float] = field(default_factory=list)


def get_riviera_maya_and_cdmx_presets() -> Dict[str, PropertyEconomicsConfig]:
    """
    Returns calibrated 2026 presets for:
      1. Playa del Carmen (2BR Condo in Centro / Gonzalo Guerrero / Playacar)
      2. Tulum (2BR Condo/Villa in Aldea Zama / La Veleta / Region 15)
      3. Mexico City (2BR Apartment in Roma / Condesa / Polanco)
    """
    return {
        "Playa_del_Carmen_2BR": PropertyEconomicsConfig(
            market_name="Playa del Carmen (2BR Centro / Playacar)",
            base_str_adr_usd=103.0,              # AirDNA 2026 Playa del Carmen ADR: $103
            monthly_6m_lease_usd=1350.0,         # ~23,600 MXN/mo + tenant pays CFE electricity
            monthly_3m_lease_usd=1550.0,         # ~27,100 MXN/mo mid-term snowbird/nomad rate + tenant pays CFE
            ota_and_lodging_tax_pct=0.165,       # Platform fee + Quintana Roo lodging tax impact
            avg_str_stay_nights=4.8,
            net_cleaning_turnover_cost_usd=22.0, # Laundry, humid-climate AC filter/drain & turnover wear
            str_owner_utilities_winter_usd=175.0,      # Moderate A/C usage in Nov-Apr + fiber internet/water
            str_owner_utilities_summer_dac_usd=310.0,  # Heavy 24/7 guest A/C usage in May-Oct on CFE DAC (~6.08 MXN/kWh)
            lease_owner_utilities_monthly_usd=45.0,    # Tenant pays CFE electricity meter directly!
            fixed_hoa_predial_monthly_usd=180.0,       # Condo HOA (pool/rooftop/security) + predial
            # Annual mean = 58.0% (matches AirDNA 2026 Playa del Carmen 58% occupancy)
            # Jan-Apr high winter/spring, May-Aug summer/sargassum dip, Sep-Oct hurricane trough, Nov-Dec recovery
            mean_occ_by_month=[0.82, 0.85, 0.78, 0.65, 0.48, 0.44, 0.52, 0.46, 0.30, 0.36, 0.62, 0.68],
            adr_mult_by_month=[1.25, 1.28, 1.18, 1.05, 0.82, 0.78, 0.85, 0.80, 0.70, 0.74, 0.98, 1.22],
        ),
        "Tulum_2BR": PropertyEconomicsConfig(
            market_name="Tulum (2BR Aldea Zama / La Veleta)",
            base_str_adr_usd=125.0,              # AirDNA 2026 Tulum ADR: $125 ($209 in Greater Tulum luxury)
            monthly_6m_lease_usd=1200.0,         # ~21,000 MXN/mo long/mid-term lease + tenant pays CFE electricity
            monthly_3m_lease_usd=1450.0,         # ~25,400 MXN/mo 3-month nomad lease + tenant pays CFE
            ota_and_lodging_tax_pct=0.165,
            avg_str_stay_nights=4.3,
            net_cleaning_turnover_cost_usd=26.0, # Higher Tulum cleaning/laundry/water-hardness maintenance cost
            str_owner_utilities_winter_usd=195.0,      # Fiber internet + winter CFE + water
            str_owner_utilities_summer_dac_usd=355.0,  # Tulum CFE DAC tariff (>400 kWh/mo @ 6.08 MXN/kWh + IVA)
            lease_owner_utilities_monthly_usd=50.0,    # Tenant pays CFE meter directly!
            fixed_hoa_predial_monthly_usd=210.0,       # Tulum HOA (pools, 24/7 security, jungle upkeep)
            # Annual mean = 46.0% (matches AirDNA 2026 Tulum 46% occupancy; steep peak in Jan-Mar, deep trough May-Oct)
            mean_occ_by_month=[0.78, 0.76, 0.68, 0.52, 0.34, 0.28, 0.32, 0.28, 0.20, 0.24, 0.50, 0.62],
            adr_mult_by_month=[1.48, 1.38, 1.24, 1.02, 0.76, 0.70, 0.74, 0.72, 0.64, 0.68, 0.96, 1.36],
        ),
        "CDMX_Roma_Condesa_2BR": PropertyEconomicsConfig(
            market_name="Mexico City (2BR Roma / Condesa)",
            base_str_adr_usd=105.0,
            monthly_6m_lease_usd=1650.0,
            monthly_3m_lease_usd=1850.0,
            ota_and_lodging_tax_pct=0.155,
            avg_str_stay_nights=4.2,
            net_cleaning_turnover_cost_usd=18.0,
            str_owner_utilities_winter_usd=145.0,      # No A/C needed in temperate CDMX
            str_owner_utilities_summer_dac_usd=145.0,
            lease_owner_utilities_monthly_usd=45.0,
            fixed_hoa_predial_monthly_usd=140.0,
            mean_occ_by_month=[0.74, 0.78, 0.76, 0.70, 0.52, 0.46, 0.50, 0.48, 0.44, 0.72, 0.80, 0.78],
            adr_mult_by_month=[1.08, 1.12, 1.10, 1.05, 0.90, 0.86, 0.88, 0.87, 0.85, 1.18, 1.22, 1.15],
        ),
    }


# =============================================================================
# 2. EXACT BREAK-EVEN OCCUPANCY CALCULATOR (WITH SUMMER CFE DAC IMPACT)
# =============================================================================

def compute_breakeven_occupancy(cfg: PropertyEconomicsConfig) -> Dict[str, Any]:
    """
    Computes Naive vs. True Net Break-Even Occupancy in both High Season (Winter) and
    Low/Hot Season (Summer May-Oct, when CFE DAC air-conditioning bills spike and ADRs drop).
    """
    days_per_month = 30.4167
    avg_owner_str_utilities = 0.5 * (cfg.str_owner_utilities_winter_usd + cfg.str_owner_utilities_summer_dac_usd)

    naive_breakeven_6m = cfg.monthly_6m_lease_usd / (days_per_month * cfg.base_str_adr_usd)
    naive_breakeven_3m = cfg.monthly_3m_lease_usd / (days_per_month * cfg.base_str_adr_usd)

    net_rev_per_booked_night = (
        cfg.base_str_adr_usd * (1.0 - cfg.ota_and_lodging_tax_pct)
        - (cfg.net_cleaning_turnover_cost_usd / cfg.avg_str_stay_nights)
    )

    target_net_6m = (cfg.monthly_6m_lease_usd - cfg.lease_owner_utilities_monthly_usd) + avg_owner_str_utilities
    target_net_3m = (cfg.monthly_3m_lease_usd - cfg.lease_owner_utilities_monthly_usd) + avg_owner_str_utilities

    true_breakeven_6m = target_net_6m / (days_per_month * net_rev_per_booked_night)
    true_breakeven_3m = target_net_3m / (days_per_month * net_rev_per_booked_night)

    # Low-season (May-Oct) break-even when ADR is ~0.75x base and CFE DAC utility bill is at summer peak
    summer_adr = cfg.base_str_adr_usd * float(np.mean(cfg.adr_mult_by_month[4:10]))
    summer_net_per_night = (
        summer_adr * (1.0 - cfg.ota_and_lodging_tax_pct)
        - (cfg.net_cleaning_turnover_cost_usd / cfg.avg_str_stay_nights)
    )
    summer_target_6m = (cfg.monthly_6m_lease_usd - cfg.lease_owner_utilities_monthly_usd) + cfg.str_owner_utilities_summer_dac_usd
    summer_breakeven_6m = summer_target_6m / (days_per_month * summer_net_per_night)

    return {
        "market_name": cfg.market_name,
        "base_str_adr_usd": cfg.base_str_adr_usd,
        "net_contribution_per_booked_night_base_usd": round(net_rev_per_booked_night, 2),
        "naive_breakeven_occ_vs_6m_pct": round(naive_breakeven_6m * 100.0, 1),
        "true_annual_breakeven_occ_vs_6m_pct": round(true_breakeven_6m * 100.0, 1),
        "true_annual_breakeven_nights_vs_6m": round(true_breakeven_6m * days_per_month, 1),
        "true_annual_breakeven_occ_vs_3m_pct": round(true_breakeven_3m * 100.0, 1),
        "summer_may_oct_breakeven_occ_vs_6m_pct": round(summer_breakeven_6m * 100.0, 1),
        "summer_may_oct_actual_market_occ_pct": round(float(np.mean(cfg.mean_occ_by_month[4:10]) * 100.0), 1),
    }


# =============================================================================
# 3. 5,000-YEAR MONTE CARLO SIMULATION FOR RIVIERA MAYA & CDMX
# =============================================================================

def compare_rental_strategies(
    cfg: PropertyEconomicsConfig,
    n_simulations: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulates 5 strategies across 5,000 years:
      1. Pure_STR_Airbnb_12M: 12 months on Airbnb (absorbs summer CFE DAC bills + Sep/Oct hurricane trough)
      2. Rolling_6M_Fixed_Lease: Two 6-month leases (tenant pays CFE electricity)
      3. Rolling_3M_MidTerm_Lease: Four 3-month leases (tenant pays CFE electricity)
      4. Hybrid_6M_High_STR_6M_MayOct_Lease: Airbnb in Nov-Apr (6 dry/high months) + 6M Lease in May-Oct
      5. Hybrid_4M_Peak_STR_8M_Lease: Airbnb in Jan-Mar + Dec (4 peak winter months) +Two 4M/8M Leases Apr-Nov
    """
    rng = np.random.default_rng(seed)
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    mean_occ = np.array(cfg.mean_occ_by_month, dtype=float)
    adr_mult = np.array(cfg.adr_mult_by_month, dtype=float)

    # Monthly owner-paid STR utilities (Winter Nov-Apr vs Summer CFE DAC May-Oct)
    str_utils_by_month = np.array([
        cfg.str_owner_utilities_winter_usd if m in [0, 1, 2, 3, 10, 11] else cfg.str_owner_utilities_summer_dac_usd
        for m in range(12)
    ])

    kappa = 12.0
    alpha_m = mean_occ * kappa
    beta_m = (1.0 - mean_occ) * kappa

    occ_sim = rng.beta(alpha_m[None, :], beta_m[None, :], size=(n_simulations, 12))
    adr_noise = rng.lognormal(mean=0.0, sigma=0.07, size=(n_simulations, 12))
    adr_sim = cfg.base_str_adr_usd * adr_mult[None, :] * adr_noise

    booked_nights_sim = np.round(occ_sim * days_in_month[None, :])
    net_per_night_sim = (
        adr_sim * (1.0 - cfg.ota_and_lodging_tax_pct)
        - (cfg.net_cleaning_turnover_cost_usd / cfg.avg_str_stay_nights)
    )

    str_monthly_noi = (
        booked_nights_sim * net_per_night_sim
        - str_utils_by_month[None, :]
        - cfg.fixed_hoa_predial_monthly_usd
    )

    # 1. Pure 12-Month Airbnb
    noi_pure_str = np.sum(str_monthly_noi, axis=1)

    # 2. Rolling 6-Month Fixed Lease (tenant pays CFE)
    gap_days_6m = rng.binomial(1, 0.18, size=(n_simulations, 2)) * rng.integers(4, 11, size=(n_simulations, 2))
    noi_6m_lease = (
        12.0 * (cfg.monthly_6m_lease_usd - cfg.lease_owner_utilities_monthly_usd - cfg.fixed_hoa_predial_monthly_usd)
        - np.sum(gap_days_6m, axis=1) * (cfg.monthly_6m_lease_usd / 30.4)
        - 2.0 * 75.0
    )

    # 3. Rolling 3-Month Mid-Term Lease (tenant pays CFE; higher vacancy friction in May-Oct)
    gap_days_3m = rng.binomial(1, 0.30, size=(n_simulations, 4)) * rng.integers(6, 15, size=(n_simulations, 4))
    noi_3m_lease = (
        12.0 * (cfg.monthly_3m_lease_usd - cfg.lease_owner_utilities_monthly_usd - cfg.fixed_hoa_predial_monthly_usd)
        - np.sum(gap_days_3m, axis=1) * (cfg.monthly_3m_lease_usd / 30.4)
        - 4.0 * 75.0
    )

    # 4. Seasonal Hybrid A: 6M High-Season STR (Nov-Apr) + 6M Low-Season Lease (May-Oct, tenant pays summer CFE!)
    high_6m_idx = [0, 1, 2, 3, 10, 11]
    hybrid_gap_a = rng.binomial(1, 0.22, size=n_simulations) * rng.integers(4, 10, size=n_simulations)
    noi_hybrid_6m_6m = (
        np.sum(str_monthly_noi[:, high_6m_idx], axis=1)
        + 6.0 * (cfg.monthly_6m_lease_usd - cfg.lease_owner_utilities_monthly_usd - cfg.fixed_hoa_predial_monthly_usd)
        - hybrid_gap_a * (cfg.monthly_6m_lease_usd / 30.4)
        - 75.0
    )

    # 5. Seasonal Hybrid B (Peak-4M STR [Dec-Mar] + Two 4M/8M Mid-Term Leases [Apr-Nov])
    peak_4m_idx = [0, 1, 2, 11]
    hybrid_gap_b = rng.binomial(1, 0.25, size=n_simulations) * rng.integers(5, 12, size=n_simulations)
    noi_hybrid_4m_8m = (
        np.sum(str_monthly_noi[:, peak_4m_idx], axis=1)
        + 8.0 * (cfg.monthly_6m_lease_usd - cfg.lease_owner_utilities_monthly_usd - cfg.fixed_hoa_predial_monthly_usd)
        - hybrid_gap_b * (cfg.monthly_6m_lease_usd / 30.4)
        - 90.0
    )

    def _summarize(name: str, arr: np.ndarray, baseline: np.ndarray) -> Dict[str, Any]:
        mean_val = float(np.mean(arr))
        sd_val = float(np.std(arr))
        p5_cvar = float(np.mean(arr[arr <= np.percentile(arr, 5)]))
        return {
            "strategy": name,
            "expected_annual_noi_usd": round(mean_val, 2),
            "expected_monthly_noi_usd": round(mean_val / 12.0, 2),
            "annual_noi_std_usd": round(sd_val, 2),
            "downside_5pct_cvar_annual_usd": round(p5_cvar, 2),
            "prob_beats_6m_lease_pct": round(float(np.mean(arr > baseline) * 100.0), 1),
        }

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fixed_6m_monthly_noi = cfg.monthly_6m_lease_usd - cfg.lease_owner_utilities_monthly_usd - cfg.fixed_hoa_predial_monthly_usd
    monthly_breakdown = []
    for m_idx, m_name in enumerate(month_names):
        m_str_mean = float(np.mean(str_monthly_noi[:, m_idx]))
        monthly_breakdown.append({
            "month": m_name,
            "str_occ_pct": round(float(mean_occ[m_idx] * 100.0), 1),
            "str_adr_usd": round(float(cfg.base_str_adr_usd * adr_mult[m_idx]), 2),
            "str_noi_usd": round(m_str_mean, 2),
            "lease_6m_noi_usd": round(fixed_6m_monthly_noi, 2),
            "optimal_regime": "STR_Airbnb" if m_str_mean > fixed_6m_monthly_noi else "Lease_6M",
        })

    return {
        "breakeven": compute_breakeven_occupancy(cfg),
        "strategies": [
            _summarize("1_Pure_STR_Airbnb_12M", noi_pure_str, noi_6m_lease),
            _summarize("2_Rolling_6M_Fixed_Lease", noi_6m_lease, noi_6m_lease),
            _summarize("3_Rolling_3M_MidTerm_Lease", noi_3m_lease, noi_6m_lease),
            _summarize("4_Hybrid_6M_NovApr_STR_6M_MayOct_Lease", noi_hybrid_6m_6m, noi_6m_lease),
            _summarize("5_Hybrid_4M_DecMar_STR_8M_AprNov_Lease", noi_hybrid_4m_8m, noi_6m_lease),
        ],
        "monthly_regimes": monthly_breakdown,
    }


if __name__ == "__main__":
    presets = get_riviera_maya_and_cdmx_presets()
    all_outputs = {}
    for key, cfg in presets.items():
        all_outputs[key] = compare_rental_strategies(cfg, n_simulations=5000, seed=42)

    print(json.dumps(all_outputs, indent=2))