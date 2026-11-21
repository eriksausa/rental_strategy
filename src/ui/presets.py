"""
Preconfigured demo scenarios for quick interactive exploration in the Streamlit UI.

Using a frozen dataclass keeps preset defaults strongly typed and avoids brittle
substring matching against localized dropdown labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ScenarioPreset:
    """Default unit, lease window, offer, and operating costs for a named demo scenario."""

    key: str
    label_en: str
    label_es: str
    unit_id: str
    start_month: int            # 1..12
    contract_months: int        # 3 or 6
    monthly_offer_mxn: float
    hoa_monthly_mxn: float
    wifi_monthly_mxn: float
    cleaning_per_stay_mxn: float
    summer_cfe_monthly_mxn: float

    def label(self, lang: str) -> str:
        """Return the localized selectbox label."""
        return self.label_es if lang == "es" else self.label_en


SCENARIO_PRESETS: Tuple[ScenarioPreset, ...] = (
    ScenarioPreset(
        key="summer_dac_trap",
        label_en="Scenario 1: Summer CFE DAC Trap (Unit Tulum · May–Oct · 6M Offer @ $25,000 MXN/mo)",
        label_es="Escenario 1: Trampa CFE DAC de Verano (Unidad Tulum · May–Oct · Oferta 6 Meses @ $25,000 MXN/mes)",
        unit_id="UNIT_TULUM",
        start_month=5,
        contract_months=6,
        monthly_offer_mxn=25000.0,
        hoa_monthly_mxn=3100.0,
        wifi_monthly_mxn=1100.0,
        cleaning_per_stay_mxn=700.0,
        summer_cfe_monthly_mxn=6000.0,
    ),
    ScenarioPreset(
        key="peak_winter",
        label_en="Scenario 2: Peak Winter High Season (Unit C Playa · Nov–Apr · 6M Offer @ $30,000 MXN/mo)",
        label_es="Escenario 2: Temporada Alta de Invierno (Unidad C Playa · Nov–Abr · Oferta 6 Meses @ $30,000 MXN/mes)",
        unit_id="UNIT_C_PLAYA",
        start_month=11,
        contract_months=6,
        monthly_offer_mxn=30000.0,
        hoa_monthly_mxn=3600.0,
        wifi_monthly_mxn=1300.0,
        cleaning_per_stay_mxn=850.0,
        summer_cfe_monthly_mxn=6500.0,
    ),
    ScenarioPreset(
        key="split_season",
        label_en="Scenario 3: Split Season Close Call (Unit A Playa · Apr–Jun · 3M Offer @ $23,500 MXN/mo)",
        label_es="Escenario 3: Decisión Cerrada en Transición (Unidad A Playa · Abr–Jun · Oferta 3 Meses @ $23,500 MXN/mes)",
        unit_id="UNIT_A_PLAYA",
        start_month=4,
        contract_months=3,
        monthly_offer_mxn=23500.0,
        hoa_monthly_mxn=2800.0,
        wifi_monthly_mxn=1100.0,
        cleaning_per_stay_mxn=650.0,
        summer_cfe_monthly_mxn=5800.0,
    ),
)
