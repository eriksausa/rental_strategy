"""
Sidebar input controls for the Rental Strategy Optimizer.

Collects all property, lease offer, operating expense, booking channel, CFE electricity,
and uncertainty-method parameters into a single typed `SidebarInputs` dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import streamlit as st

from src.rental_engine.assumptions import (
    DEFAULT_ASSUMPTIONS,
    MONTH_NAMES_EN,
    MONTH_NAMES_ES,
    PORTFOLIO_UNITS,
)
from src.rental_engine.engine import UserCostInputs

from .i18n import pick, sidebar_author_markdown
from .presets import ScenarioPreset


@dataclass(frozen=True)
class SidebarInputs:
    """Validated user inputs returned by `render_sidebar()`."""

    unit_id: str
    start_month: int
    contract_months: int
    months_window: List[int]
    use_block_bootstrap: bool
    costs: UserCostInputs
    prior_11m_kwh: Optional[List[float]]
    window_monthly_kwh: Optional[List[float]]


def render_sidebar(preset: ScenarioPreset, lang: str) -> SidebarInputs:
    """Render all sidebar widgets seeded by `preset` and return structured `SidebarInputs`."""
    unit_keys = list(PORTFOLIO_UNITS.keys())
    name_key = "name_es" if lang == "es" else "name_en"
    month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN
    unit_labels = [str(PORTFOLIO_UNITS[k][name_key]) for k in unit_keys]
    default_unit_idx = unit_keys.index(preset.unit_id)
    default_dur_idx = 0 if preset.contract_months == 3 else 1

    with st.sidebar:
        st.markdown("### 👨‍💻 Created By")
        st.markdown(sidebar_author_markdown(lang))

        # --- Section 1: Apartment, Lease Window & Uncertainty Mode ---
        st.header(pick("1. Apartment & Lease Offer", "1. Unidad y Oferta", lang))
        selected_label = st.selectbox(
            pick("Select Property", "Seleccionar Propiedad", lang),
            unit_labels,
            index=default_unit_idx,
        )
        selected_unit_id = unit_keys[unit_labels.index(selected_label)]

        start_month = st.selectbox(
            pick("Lease Start Month", "Mes de Inicio del Contrato", lang),
            options=list(range(1, 13)),
            format_func=lambda m: f"{m:02d} - {month_map[m]}",
            index=preset.start_month - 1,
        )
        contract_months = st.radio(
            pick("Contract Duration (Months)", "Duración del Contrato (Meses)", lang),
            [3, 6],
            index=default_dur_idx,
            horizontal=True,
        )
        use_block_bootstrap = st.toggle(
            pick(
                "Block-bootstrap uncertainty (dependent OOF residuals)",
                "Incertidumbre por bootstrap de bloques (residuos OOF dependientes)",
                lang,
            ),
            value=True,
            help=pick(
                "Resamples consecutive-horizon backtest residual blocks to preserve month-to-month "
                "persistence and ADR/occupancy error correlation. Turn off to compare against "
                "independent monthly normal draws.",
                "Remuestrea bloques consecutivos de residuos del backtest para conservar la persistencia "
                "entre meses y la correlación ADR/ocupación. Desactívalo para comparar con normales independientes.",
                lang,
            ),
        )
        lease_offer_mxn = st.number_input(
            pick("Tenant Monthly Offer (MXN)", "Oferta Mensual del Inquilino (MXN)", lang),
            min_value=5000.0,
            max_value=150000.0,
            value=preset.monthly_offer_mxn,
            step=500.0,
        )

        # --- Section 2: Fixed & Variable Operating Expenses ---
        st.header(pick("2. Operating Expenses", "2. Gastos Operativos", lang))
        st.caption(
            pick(
                "Enter the unit's operating expenses (defaults are illustrative):",
                "Ingresa los gastos operativos del departamento (valores iniciales ilustrativos):",
                lang,
            )
        )
        hoa_mxn = st.number_input(
            pick("Monthly HOA / Maintenance (MXN)", "Mantenimiento / HOA Mensual (MXN)", lang),
            min_value=0.0,
            value=preset.hoa_monthly_mxn,
            step=100.0,
        )
        wifi_mxn = st.number_input(
            pick("Monthly Wi-Fi, Water & Gas (MXN)", "Internet, Agua y Gas Mensual (MXN)", lang),
            min_value=0.0,
            value=preset.wifi_monthly_mxn,
            step=100.0,
        )
        cleaning_mxn = st.number_input(
            pick("Unrecovered Turnover Cleaning / Stay (MXN)", "Costo de Limpieza por Estancia (MXN)", lang),
            min_value=0.0,
            value=preset.cleaning_per_stay_mxn,
            step=50.0,
        )
        avg_stay = st.number_input(
            pick("Average Stay Length (Nights)", "Estancia Promedio (Noches)", lang),
            min_value=1.0,
            max_value=30.0,
            value=4.5,
            step=0.5,
        )

        # --- Section 3: Booking Channel Mix & CFE Electricity ---
        st.header(pick("3. Channel Mix & CFE Electricity", "3. Canales y Luz CFE", lang))
        fb_share_pct = st.slider(
            pick(
                "Facebook Group Direct Bookings (% at 0% Fee)",
                "% Reservas Directas en Facebook (0% Comisión)",
                lang,
            ),
            min_value=0,
            max_value=100,
            value=60,
        )
        airbnb_fee_pct = st.slider(
            pick("Airbnb Platform + Q. Roo Tax (%)", "Comisión Airbnb + Impuesto Q. Roo (%)", lang),
            min_value=10.0,
            max_value=25.0,
            value=18.0,
            step=0.5,
        )

        cfe_mode = st.radio(
            pick("CFE Electricity Input Mode", "Modo de Estimación CFE", lang),
            options=(
                ["Recibo Mensual Directo (MXN)", "Simulador kWh Promedio Móvil 12M (Tarifa 1B)"]
                if lang == "es"
                else ["Direct Monthly Bills (MXN)", "12M Rolling-Average kWh Mode (Tariff 1B)"]
            ),
        )
        use_kwh_mode = "kWh" in cfe_mode

        prior_11m_kwh: Optional[List[float]] = None
        window_monthly_kwh: Optional[List[float]] = None
        months_window = [((int(start_month) - 1 + i) % 12) + 1 for i in range(int(contract_months))]

        if use_kwh_mode:
            prior_avg_kwh = st.slider(
                pick(
                    "Prior 11-Month Avg Consumption (kWh/mo)",
                    "Promedio kWh/mes (11 meses previos)",
                    lang,
                ),
                min_value=150,
                max_value=900,
                value=390,
                step=10,
            )
            winter_kwh = st.slider(
                pick(
                    "Winter STR Monthly Usage (Nov–Apr, kWh)",
                    "Consumo Mensual Invierno en Renta Corta (kWh)",
                    lang,
                ),
                min_value=150,
                max_value=800,
                value=280,
                step=20,
            )
            summer_kwh = st.slider(
                pick(
                    "Summer STR Monthly Usage (May–Oct, kWh)",
                    "Consumo Mensual Verano A/C en Renta Corta (kWh)",
                    lang,
                ),
                min_value=300,
                max_value=1200,
                value=780,
                step=20,
            )
            prior_11m_kwh = [float(prior_avg_kwh)] * 11
            window_monthly_kwh = [
                float(summer_kwh if m in DEFAULT_ASSUMPTIONS.summer_months else winter_kwh)
                for m in months_window
            ]
            winter_cfe = 1600.0
            summer_cfe = preset.summer_cfe_monthly_mxn
        else:
            winter_cfe = st.number_input(
                pick(
                    "Winter Monthly CFE Bill on STR (Nov–Apr, MXN)",
                    "Recibo CFE Invierno en Renta Corta (Nov–Abr, MXN)",
                    lang,
                ),
                min_value=0.0,
                value=1600.0,
                step=200.0,
            )
            summer_cfe = st.number_input(
                pick(
                    "Summer Monthly CFE DAC Bill on STR (May–Oct, MXN)",
                    "Recibo CFE DAC Verano en Renta Corta (May–Oct, MXN)",
                    lang,
                ),
                min_value=0.0,
                value=preset.summer_cfe_monthly_mxn,
                step=200.0,
            )

    costs = UserCostInputs(
        unit_id=selected_unit_id,
        contract_months=int(contract_months),
        monthly_lease_offer_mxn=float(lease_offer_mxn),
        hoa_monthly_mxn=float(hoa_mxn),
        wifi_gas_water_monthly_mxn=float(wifi_mxn),
        cleaning_cost_per_stay_mxn=float(cleaning_mxn),
        avg_stay_nights=float(avg_stay),
        airbnb_fee_pct=float(airbnb_fee_pct) / 100.0,
        facebook_direct_share=float(fb_share_pct) / 100.0,
        winter_cfe_monthly_mxn=float(winter_cfe),
        summer_cfe_monthly_mxn=float(summer_cfe),
    )
    return SidebarInputs(
        unit_id=selected_unit_id,
        start_month=int(start_month),
        contract_months=int(contract_months),
        months_window=months_window,
        use_block_bootstrap=bool(use_block_bootstrap),
        costs=costs,
        prior_11m_kwh=prior_11m_kwh,
        window_monthly_kwh=window_monthly_kwh,
    )
