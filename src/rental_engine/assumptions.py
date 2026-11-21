"""
Explicit Domain, CFE Tariff, Public Market Benchmark & Simulation Assumptions.

Every constant used by the decision engine, CFE billing model, and Monte Carlo
simulator is collected in the `Assumptions` dataclass with documented sources
and access dates.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from typing import Dict, Tuple


# ==============================================================================
# 1. CFE Domestic High-Consumption (DAC) Tariff Limits (12-Month Rolling Average)
# Source: CFE "Tarifa DAC (Doméstica de Alto Consumo)" & Tarifas Domésticas 1..1F
#         https://app.cfe.mx/Aplicaciones/CCFE/Tarifas/TarifasCRECasa/Casa.aspx
#         (accessed 2026-11-15).
# Reclassification rule: A domestic meter is reclassified into Tariff DAC when
# its rolling 12-month average monthly consumption (last 365 billed days / 6
# bimonthly bills) exceeds the threshold for its assigned local tariff.
# ==============================================================================
CFE_TARIFF_DAC_LIMITS_KWH: Dict[str, float] = {
    "1": 250.0,     # Temperate zones
    "1A": 300.0,    # Min summer mean temp 25°C
    "1B": 400.0,    # Min summer mean temp 28°C (user-confirmed tariff for portfolio units)
    "1C": 850.0,    # Min summer mean temp 30°C
    "1D": 1000.0,   # Min summer mean temp 31°C
    "1E": 2000.0,   # Min summer mean temp 32°C
    "1F": 2500.0,   # Min summer mean temp 33°C
}


# ==============================================================================
# 2. Public Market Benchmarks (AirDNA MarketMinder, accessed 2026)
# Preserved as cited public reference figures; unit-level profiles in the demo
# are illustrative synthetic curves around these public market anchors.
# ==============================================================================
AIRDNA_PUBLIC_BENCHMARKS: Dict[str, Dict[str, object]] = {
    "Playa del Carmen": {
        "source": "AirDNA MarketMinder — Playa del Carmen, Q. Roo (accessed 2026)",
        "active_listings": 6500,
        "annual_mean_occupancy": 0.58,
        "average_daily_rate_usd": 103.0,
        "revpar_usd": 59.0,
    },
    "Tulum": {
        "source": "AirDNA MarketMinder — Tulum, Q. Roo (accessed 2026)",
        "active_listings": 3609,
        "annual_mean_occupancy": 0.46,  # Public AirDNA Tulum market-wide occupancy
        "average_daily_rate_usd": 125.0,
        "revpar_usd": 57.5,
        "seasonality_score": 54,
    },
    "CDMX": {
        "source": "AirDNA MarketMinder — Mexico City Roma/Condesa benchmark (accessed 2026)",
        "annual_mean_occupancy": 0.66,
        "average_daily_rate_usd": 105.0,
        "revpar_usd": 69.0,
    },
}


# ==============================================================================
# 3. Illustrative Demo Portfolio Units (No Private Property Identifiers)
# ==============================================================================
PORTFOLIO_UNITS: Dict[str, Dict[str, object]] = {
    "UNIT_A_PLAYA": {
        "name_en": "Unit A Playa — 1BR/1BA",
        "name_es": "Unidad A Playa — 1 Rec / 1 Baño",
        "city": "Playa del Carmen",
        "neighborhood": "Not disclosed",
        "bedrooms": 1,
        "bathrooms": 1,
        "base_adr_mxn": 1650.0,
        "occ_multiplier": 1.00,
        "notes_en": "Illustrative 1BR apartment in Playa del Carmen; steady digital nomad + tourist mix.",
        "notes_es": "Departamento ilustrativo de 1 recámara en Playa del Carmen; demanda constante de nómadas digitales y turistas.",
    },
    "UNIT_B_PLAYA": {
        "name_en": "Unit B Playa — 1BR/1BA",
        "name_es": "Unidad B Playa — 1 Rec / 1 Baño",
        "city": "Playa del Carmen",
        "neighborhood": "Not disclosed",
        "bedrooms": 1,
        "bathrooms": 1,
        "base_adr_mxn": 1750.0,
        "occ_multiplier": 1.02,
        "notes_en": "Illustrative 1BR residential unit in Playa del Carmen; strong Facebook Group mid-term demand.",
        "notes_es": "Unidad residencial ilustrativa de 1 recámara en Playa del Carmen; alta demanda en grupos de Facebook.",
    },
    "UNIT_C_PLAYA": {
        "name_en": "Unit C Playa — 2BR/2BA",
        "name_es": "Unidad C Playa — 2 Rec / 2 Baños",
        "city": "Playa del Carmen",
        "neighborhood": "Not disclosed",
        "bedrooms": 2,
        "bathrooms": 2,
        "base_adr_mxn": 2550.0,
        "occ_multiplier": 0.98,
        "notes_en": "Illustrative 2BR unit; higher winter ADR ceiling, multiple A/C units increase May–Oct CFE DAC exposure.",
        "notes_es": "Unidad ilustrativa de 2 recámaras; mayor tarifa en invierno, múltiples A/C elevan el riesgo de CFE DAC.",
    },
    "UNIT_TULUM": {
        "name_en": "Unit Tulum — 1BR/1BA",
        "name_es": "Unidad Tulum — 1 Rec / 1 Baño",
        "city": "Tulum",
        "neighborhood": "Not disclosed",
        "bedrooms": 1,
        "bathrooms": 1,
        "base_adr_mxn": 1550.0,
        "occ_multiplier": 0.74,
        "notes_en": "Illustrative Tulum 1BR unit anchored to AirDNA's ~46% Tulum market-wide annual occupancy; sharp May–Oct low-season dip.",
        "notes_es": "Unidad ilustrativa en Tulum anclada al ~46% de ocupación media del mercado en Tulum (AirDNA); fuerte caída en mayo–octubre.",
    },
    "MKT_CDMX_1BR": {
        "name_en": "CDMX Benchmark — 1BR/1BA (Roma Norte)",
        "name_es": "Benchmark CDMX — 1 Rec / 1 Baño (Roma Norte)",
        "city": "CDMX",
        "neighborhood": "Roma Norte",
        "bedrooms": 1,
        "bathrooms": 1,
        "base_adr_mxn": 1850.0,
        "occ_multiplier": 1.08,
        "notes_en": "Urban comparison benchmark (~66% annual occupancy per AirDNA, minimal summer A/C utility drag).",
        "notes_es": "Mercado urbano de comparación (~66% ocupación anual según AirDNA, sin gasto elevado de aire acondicionado).",
    },
}

MONTH_NAMES_EN: Dict[int, str] = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

MONTH_NAMES_ES: Dict[int, str] = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}

BASE_SEASONAL_OCC_CURVE: Dict[int, float] = {
    1: 0.82, 2: 0.80, 3: 0.76, 4: 0.68, 5: 0.48, 6: 0.44,
    7: 0.55, 8: 0.50, 9: 0.36, 10: 0.42, 11: 0.64, 12: 0.80,
}

BASE_SEASONAL_ADR_MULT: Dict[int, float] = {
    1: 1.30, 2: 1.25, 3: 1.20, 4: 1.05, 5: 0.82, 6: 0.78,
    7: 0.92, 8: 0.88, 9: 0.74, 10: 0.78, 11: 1.02, 12: 1.28,
}


@dataclass(frozen=True)
class Assumptions:
    """
    Centralized, documented modeling assumptions for CFE billing, Monte Carlo
    cash-flow simulation, and lease-vs-STR decision rules.
    """
    # --- CFE Electricity & DAC Reclassification (CFE Peninsular, accessed 2026-11-15) ---
    cfe_tariff: str = "1B"                         # Default domestic tariff (1B = 400 kWh/mo rolling 12M limit)
    cfe_dac_threshold_kwh: float = 400.0           # Rolling 12-month mean kWh/mo threshold for Tariff 1B
    cfe_dac_fixed_charge_mxn: float = 145.79       # CFE DAC Peninsular monthly fixed charge before IVA (MXN)
    cfe_dac_rate_mxn_per_kwh: float = 6.08         # Unsubsidized DAC Peninsular energy rate before IVA (MXN/kWh)
    cfe_subsidized_rate_mxn_per_kwh: float = 1.85  # Blended subsidized Tariff 1B rate before IVA (MXN/kWh)
    cfe_iva_multiplier: float = 1.16               # Mexican 16% VAT (IVA) applied to CFE electricity bills
    summer_months: Tuple[int, ...] = (5, 6, 7, 8, 9, 10)  # High A/C cooling months in Quintana Roo (May–Oct)
    high_season_months: Tuple[int, ...] = (12, 1, 2, 3, 4)

    # --- Monte Carlo Predictive Uncertainty & Sampling Bounds ---
    default_monte_carlo_runs: int = 5000           # Number of Monte Carlo trajectories per evaluation
    default_reference_year: int = 2026             # Default calendar year for exact days-in-month lookup
    fallback_adr_std_mxn: float = 180.0            # Fallback ADR std (MXN) when backtest residuals are not supplied
    fallback_occ_std: float = 0.08                 # Fallback occupancy std (0..1) when backtest residuals are not supplied
    min_sampled_adr_mxn: float = 300.0             # Lower truncation bound on simulated nightly ADR (MXN)
    min_sampled_occ_rate: float = 0.05             # Lower truncation bound on simulated monthly occupancy (5%)
    max_sampled_occ_rate: float = 0.98             # Upper truncation bound on simulated monthly occupancy (98%)

    # --- Decision Rule Thresholds ---
    accept_lease_max_str_win_prob: float = 0.45    # Recommend ACCEPT CONTRACT when Lease >= P50_STR and P(STR > Lease) < 0.45
    keep_open_min_str_win_prob: float = 0.65       # Recommend KEEP CALENDAR OPEN when P(STR > Lease) > 0.65
    counter_offer_risk_premium: float = 1.05       # 5% risk-premium markup over indifference rent for counter-offers

    def dac_limit_for_tariff(self, tariff: str | None = None) -> float:
        """Returns the 12-month rolling average kWh/month DAC threshold for `tariff`."""
        t = (tariff or self.cfe_tariff).upper().strip()
        return CFE_TARIFF_DAC_LIMITS_KWH.get(t, self.cfe_dac_threshold_kwh)


DEFAULT_ASSUMPTIONS = Assumptions()


def days_in_month(year: int, month: int) -> int:
    """Returns the exact calendar days (28..31) for `(year, month)`."""
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    return calendar.monthrange(int(year), int(month))[1]
