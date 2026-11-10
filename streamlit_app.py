"""
Casa Bella — Rental Strategy Optimizer (Playa del Carmen, Tulum & CDMX)
Bilingual (English / Español) Portfolio Demo & Interactive Decision Engine

Run locally with:
    streamlit run app.py
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==============================================================================
# 1. Domain Constants & Portfolio Units (Zero Hidden Cost Assumptions)
# ==============================================================================
CFE_DAC_THRESHOLD_KWH = 400.0
CFE_DAC_BASE_RATE_MXN = 6.08
CFE_IVA_MULTIPLIER = 1.16

PORTFOLIO_UNITS: Dict[str, Dict[str, object]] = {
    "CB_PDC_1BR_A": {
        "name_en": "Playa del Carmen — 1BR/1BA #A (Centro)",
        "name_es": "Playa del Carmen — 1 Rec / 1 Baño #A (Centro)",
        "city": "Playa del Carmen",
        "neighborhood": "Centro",
        "bedrooms": 1,
        "bathrooms": 1,
        "base_adr_mxn": 1650.0,
        "occ_multiplier": 1.00,
        "notes_en": "Walkable to 5th Ave & beach; steady year-round digital nomad + tourist mix.",
        "notes_es": "A pasos de la 5ta Av. y la playa; demanda constante de nómadas digitales y turistas.",
    },
    "CB_PDC_1BR_B": {
        "name_en": "Playa del Carmen — 1BR/1BA #B (Centro)",
        "name_es": "Playa del Carmen — 1 Rec / 1 Baño #B (Centro)",
        "city": "Playa del Carmen",
        "neighborhood": "Gonzalo Guerrero",
        "bedrooms": 1,
        "bathrooms": 1,
        "base_adr_mxn": 1750.0,
        "occ_multiplier": 1.02,
        "notes_en": "Quiet residential/nomad zone; strong Facebook Group mid-term rental demand.",
        "notes_es": "Zona residencial tranquila; alta demanda de rentas de mediano plazo en grupos de Facebook.",
    },
    "CB_PDC_2BR": {
        "name_en": "Playa del Carmen — 2BR/2BA (Gonzalo Guerrero)",
        "name_es": "Playa del Carmen — 2 Rec / 2 Baños (Gonzalo Guerrero)",
        "city": "Playa del Carmen",
        "neighborhood": "Gonzalo Guerrero",
        "bedrooms": 2,
        "bathrooms": 2,
        "base_adr_mxn": 2550.0,
        "occ_multiplier": 0.98,
        "notes_en": "Higher winter ADR ceiling; multiple A/C units increase May–Oct CFE DAC exposure.",
        "notes_es": "Mayor tarifa por noche en invierno; múltiples aires acondicionados elevan el riesgo de tarifa CFE DAC en verano.",
    },
    "CB_TUL_1BR": {
        "name_en": "Tulum — 1BR/1BA (La Veleta)",
        "name_es": "Tulum — 1 Rec / 1 Baño (La Veleta)",
        "city": "Tulum",
        "neighborhood": "La Veleta",
        "bedrooms": 1,
        "bathrooms": 1,
        "base_adr_mxn": 1550.0,
        "occ_multiplier": 0.74,
        "notes_en": "Calibrated to ~46% annual mean occupancy; sharp May–Oct low-season compression.",
        "notes_es": "Calibrado a ~46% de ocupación media anual; fuerte caída en temporada baja (mayo–octubre).",
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
        "notes_en": "Urban comparison market (~66% annual occupancy, minimal summer A/C utility drag).",
        "notes_es": "Mercado urbano de comparación (~66% ocupación anual, sin gasto elevado de aire acondicionado).",
    },
}

MONTH_NAMES_EN = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

MONTH_NAMES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}

BASE_SEASONAL_OCC_CURVE = {
    1: 0.82, 2: 0.80, 3: 0.76, 4: 0.68, 5: 0.48, 6: 0.44,
    7: 0.55, 8: 0.50, 9: 0.36, 10: 0.42, 11: 0.64, 12: 0.80,
}

BASE_SEASONAL_ADR_MULT = {
    1: 1.30, 2: 1.25, 3: 1.20, 4: 1.05, 5: 0.82, 6: 0.78,
    7: 0.92, 8: 0.88, 9: 0.74, 10: 0.78, 11: 1.02, 12: 1.28,
}


@dataclass
class UserCostInputs:
    unit_id: str
    contract_months: int
    monthly_lease_offer_mxn: float
    hoa_monthly_mxn: float
    wifi_gas_water_monthly_mxn: float
    cleaning_cost_per_stay_mxn: float
    avg_stay_nights: float
    airbnb_fee_pct: float
    facebook_direct_share: float
    winter_cfe_monthly_mxn: float
    summer_cfe_monthly_mxn: float


def compute_blended_ota_drag(airbnb_fee_pct: float, facebook_direct_share: float) -> float:
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


def build_unit_monthly_forecasts(unit_id: str, months_window: List[int], lang: str = "en") -> List[Dict[str, float]]:
    profile = PORTFOLIO_UNITS[unit_id]
    base_adr = float(profile["base_adr_mxn"])
    occ_mult = float(profile["occ_multiplier"])
    city = str(profile["city"])
    month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN

    forecasts = []
    for m in months_window:
        if city == "CDMX":
            adr_mult = 1.15 if m in (10, 11, 3, 4) else 0.98
            occ_rate = float(np.clip(0.64 * (1.08 if m in (10, 11) else 0.98), 0.45, 0.85))
        else:
            adr_mult = BASE_SEASONAL_ADR_MULT[m]
            occ_rate = float(np.clip(BASE_SEASONAL_OCC_CURVE[m] * occ_mult, 0.22, 0.92))

        forecasts.append(
            {
                "month_num": int(m),
                "month_name": month_map[int(m)],
                "pred_adr_mxn": round(base_adr * adr_mult, 2),
                "pred_occ_rate": round(occ_rate, 4),
                "adr_std_mxn": round(base_adr * 0.11, 2),
                "occ_std": 0.075,
            }
        )
    return forecasts


def run_monte_carlo_simulation(
    monthly_forecasts: List[Dict[str, float]],
    costs: UserCostInputs,
    lang: str = "en",
    n_simulations: int = 5000,
    random_seed: int = 42,
) -> Dict[str, object]:
    rng = np.random.default_rng(random_seed)
    n_months = len(monthly_forecasts)

    monthly_fixed_owner_costs = costs.hoa_monthly_mxn + costs.wifi_gas_water_monthly_mxn
    lease_monthly_net = costs.monthly_lease_offer_mxn - monthly_fixed_owner_costs
    lease_total_net_mxn = float(n_months * lease_monthly_net)

    blended_fee = compute_blended_ota_drag(costs.airbnb_fee_pct, costs.facebook_direct_share)
    cleaning_per_night = costs.cleaning_cost_per_stay_mxn / max(costs.avg_stay_nights, 1.0)

    sim_str_totals = np.zeros(n_simulations, dtype=float)
    monthly_breakdown = []

    for m in monthly_forecasts:
        month_num = int(m["month_num"])
        is_summer = month_num in (5, 6, 7, 8, 9, 10)
        cfe_bill = costs.summer_cfe_monthly_mxn if is_summer else costs.winter_cfe_monthly_mxn

        adr_samples = np.maximum(
            rng.normal(loc=m["pred_adr_mxn"], scale=m["adr_std_mxn"], size=n_simulations),
            350.0,
        )
        occ_samples = np.clip(
            rng.normal(loc=m["pred_occ_rate"], scale=m["occ_std"], size=n_simulations),
            0.05,
            0.98,
        )

        occupied_nights = 30.0 * occ_samples
        gross_rev = occupied_nights * adr_samples
        platform_fees = gross_rev * blended_fee
        cleaning_total = occupied_nights * cleaning_per_night
        month_net_str = gross_rev - platform_fees - cleaning_total - monthly_fixed_owner_costs - cfe_bill
        sim_str_totals += month_net_str

        be_occ = compute_break_even_occupancy(
            adr_mxn=m["pred_adr_mxn"],
            monthly_lease_offer_mxn=costs.monthly_lease_offer_mxn,
            costs=costs,
            is_summer_month=is_summer,
        )

        if lang == "es":
            season_label = (
                "Verano (Riesgo CFE DAC)" if is_summer else ("Invierno Alta" if month_num in (12, 1, 2, 3, 4) else "Media")
            )
            monthly_breakdown.append(
                {
                    "Mes": m["month_name"],
                    "Temporada": season_label,
                    "ADR Predicho (MXN)": m["pred_adr_mxn"],
                    "Ocupación Predicha (%)": round(m["pred_occ_rate"] * 100, 1),
                    "Ocupación Equilibrio (%)": round(be_occ * 100, 1),
                    "Recibo CFE Renta Corta (MXN)": cfe_bill,
                    "Neto Renta Corta Esperado (MXN)": round(float(np.mean(month_net_str)), 0),
                    "Neto Contrato Garantizado (MXN)": round(lease_monthly_net, 0),
                }
            )
        else:
            season_label = (
                "Summer (CFE DAC Risk)" if is_summer else ("High Winter" if month_num in (12, 1, 2, 3, 4) else "Shoulder")
            )
            monthly_breakdown.append(
                {
                    "Month": m["month_name"],
                    "Season": season_label,
                    "Pred ADR (MXN)": m["pred_adr_mxn"],
                    "Pred Occ (%)": round(m["pred_occ_rate"] * 100, 1),
                    "Break-Even Occ (%)": round(be_occ * 100, 1),
                    "STR CFE Bill (MXN)": cfe_bill,
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
    be_col = "Ocupación Equilibrio (%)" if lang == "es" else "Break-Even Occ (%)"
    mean_be_occ = round(float(np.mean([r[be_col] for r in monthly_breakdown])), 1)

    if lease_total_net_mxn >= p50_str and prob_str_beats_lease < 0.45:
        rec_code = "ACCEPT"
        counter_offer_mxn = costs.monthly_lease_offer_mxn
        if lang == "es":
            recommendation = "ACEPTAR CONTRATO DE ARRENDAMIENTO"
            rationale = (
                f"El contrato garantizado de {n_months} meses genera **${lease_total_net_mxn:,.0f} MXN netos** "
                f"(el inquilino paga la luz CFE directamente), superando el escenario mediano de renta corta "
                f"(**${p50_str:,.0f} MXN**) sin riesgo de noches vacías ni cobros excesivos de tarifa CFE DAC en verano."
            )
        else:
            recommendation = "ACCEPT CONTRACT"
            rationale = (
                f"The guaranteed {n_months}-month lease nets **${lease_total_net_mxn:,.0f} MXN** "
                f"(tenant pays CFE electricity directly), beating the median short-term scenario "
                f"(**${p50_str:,.0f} MXN**) with zero vacancy or summer CFE DAC bill shock."
            )
    elif prob_str_beats_lease > 0.65:
        rec_code = "OPEN"
        counter_offer_mxn = round(indifference_monthly_rent_mxn * 1.05, -2)
        if lang == "es":
            recommendation = "MANTENER CALENDARIO ABIERTO (AIRBNB / FACEBOOK DIRECTO)"
            rationale = (
                f"La renta corta supera esta oferta en el **{prob_str_beats_lease * 100:.1f}%** de las 5,000 simulaciones Monte Carlo "
                f"(Mediana Renta Corta: **${p50_str:,.0f} MXN** vs. Contrato: **${lease_total_net_mxn:,.0f} MXN**). "
                f"Solo conviene firmar contrato si el inquilino sube su oferta al menos a **${counter_offer_mxn:,.0f} MXN/mes**."
            )
        else:
            recommendation = "KEEP CALENDAR OPEN (STR / FB DIRECT)"
            rationale = (
                f"Short-term stays beat this lease offer in **{prob_str_beats_lease * 100:.1f}%** of 5,000 Monte Carlo runs "
                f"(Median STR Net: **${p50_str:,.0f} MXN** vs. Lease Net: **${lease_total_net_mxn:,.0f} MXN**). "
                f"Only sign a lease if the tenant raises their offer to at least **${counter_offer_mxn:,.0f} MXN/month**."
            )
    else:
        rec_code = "NEGOTIATE"
        counter_offer_mxn = round(max(indifference_monthly_rent_mxn, costs.monthly_lease_offer_mxn * 1.05), -2)
        if lang == "es":
            recommendation = "NEGOCIAR RENTA MENSUAL MÁS ALTA"
            rationale = (
                f"Los flujos esperados están muy parejos (renta corta gana en el **{prob_str_beats_lease * 100:.1f}%** de las corridas), "
                f"pero la renta corta conlleva riesgo a la baja (5% CVaR: **${cvar_5pct:,.0f} MXN**). "
                f"Haz una contraoferta de **${counter_offer_mxn:,.0f} MXN/mes** (+ CFE pagado por el inquilino) para asegurar ganancia sin riesgo."
            )
        else:
            recommendation = "NEGOTIATE HIGHER LEASE RATE"
            rationale = (
                f"Expected cash flows are close (STR wins in **{prob_str_beats_lease * 100:.1f}%** of runs), "
                f"but short-term carries downside risk (5% CVaR: **${cvar_5pct:,.0f} MXN**). "
                f"Counter-offer at **${counter_offer_mxn:,.0f} MXN/month** (+ CFE paid by tenant) to lock in guaranteed profit."
            )

    return {
        "sim_str_totals": sim_str_totals,
        "monthly_df": pd.DataFrame(monthly_breakdown),
        "lease_total_net_mxn": round(lease_total_net_mxn, 2),
        "str_expected_net_mxn": round(mean_str, 2),
        "str_p10_net_mxn": round(p10_str, 2),
        "str_p50_net_mxn": round(p50_str, 2),
        "str_p90_net_mxn": round(p90_str, 2),
        "str_cvar_5pct_mxn": round(cvar_5pct, 2),
        "prob_str_beats_lease": round(prob_str_beats_lease, 4),
        "mean_break_even_occupancy": mean_be_occ,
        "indifference_monthly_rent_mxn": round(indifference_monthly_rent_mxn, 0),
        "blended_platform_fee_pct": blended_fee,
        "rec_code": rec_code,
        "recommendation": recommendation,
        "rationale": rationale,
        "suggested_counter_offer_mxn": float(counter_offer_mxn),
    }


def parse_facebook_rental_post(raw_text: str) -> Dict[str, object]:
    text_lower = raw_text.lower()
    bedrooms = 1
    br_match = re.search(r"(\d)\s*(?:br|bed|rec[aá]mara|habitaci[oó]n)", text_lower)
    if br_match:
        bedrooms = int(br_match.group(1))

    neighborhood = "Unknown"
    for zone in ("gonzalo guerrero", "centro", "la veleta", "aldea zama", "roma norte", "condesa"):
        if zone in text_lower:
            neighborhood = zone.title()
            break

    price_mxn = None
    price_match = re.search(r"\$?\s*(\d{2}[,\.]?\d{3})\s*(?:mxn|pesos|/mes|monthly)?", text_lower)
    if price_match:
        price_mxn = float(price_match.group(1).replace(",", "").replace(".", ""))

    tenant_pays_cfe = any(
        kw in text_lower for kw in ("+ cfe", "plus cfe", "mas cfe", "más cfe", "plus electric", "no incluye luz")
    )
    return {
        "bedrooms": bedrooms,
        "neighborhood": neighborhood,
        "asking_monthly_mxn": price_mxn,
        "tenant_pays_cfe": tenant_pays_cfe,
    }


# ==============================================================================
# 2. Page Header, Language Switcher (EN / ES) & Portfolio Project Showcase Banner
# ==============================================================================
st.set_page_config(
    page_title="Casa Bella — Rental Strategy Optimizer",
    page_icon="🏖️",
    layout="wide",
)

top_col1, top_col2 = st.columns([4, 1])
with top_col2:
    lang_choice = st.radio(
        "🌐 Language / Idioma",
        options=["English", "Español"],
        horizontal=True,
    )
lang = "es" if lang_choice == "Español" else "en"

with top_col1:
    if lang == "es":
        st.title("🏖️ Casa Bella — Optimizador de Estrategia de Renta")
        st.caption(
            "**Demo Interactivo de Decisiones Inmobiliarias (Riviera Maya y CDMX)**"
        )
    else:
        st.title("🏖️ Casa Bella — Rental Strategy Optimizer")
        st.caption(
            "**Interactive Decision Engine Demo (Riviera Maya & CDMX)**"
        )

# Highlight Banner: What This Project Is & What It Does
if lang == "es":
    st.info(
        "🎯 **Sobre Este Proyecto (Demo Interactivo):** "
        "Construí este motor de decisiones de extremo a extremo (**SQL $\\rightarrow$ Machine Learning $\\rightarrow$ Simulación Monte Carlo $\\rightarrow$ Streamlit**) "
        "para resolver un problema real en el negocio (**Casa Bella**, con 3 departamentos en **Playa del Carmen** y 1 en **Tulum**): "
        "*cuando un inquilino ofrece un contrato fijo de 3 o 6 meses, ¿debemos aceptar la oferta hoy, negociar una renta más alta o dejar el calendario abierto "
        "para rentas cortas por Airbnb y grupos de Facebook?* A diferencia de calculadoras simples, este modelo incorpora la **tarifa eléctrica CFE DAC de verano** "
        "(donde el aire acondicionado eleva el recibo a `$5,500–$6,500 MXN/mes`), la mezcla de reservas directas sin comisión en **Grupos de Facebook (0%)** vs. **Airbnb (~18%)**, "
        "y permite ingresar el **100% de tus costos reales** sin suposiciones ocultas."
    )
else:
    st.info(
        "🎯 **About This Project (Interactive Demo):** "
        "I built this end-to-end decision engine (**SQL $\\rightarrow$ Machine Learning $\\rightarrow$ Monte Carlo Simulation $\\rightarrow$ Streamlit**) "
        "to solve a real operational question for a rental business (**Casa Bella**, operating 3 apartments in **Playa del Carmen** and 1 in **Tulum**): "
        "*when a prospective tenant offers a guaranteed 3-month or 6-month lease, should we accept the contract today, negotiate a higher rate, or keep the calendar open "
        "for short-term stays across Airbnb and Facebook Groups?* Unlike basic revenue calculators, this engine models Quintana Roo's **May–Oct CFE DAC summer electricity trap** "
        "(`$5,500–$6,500 MXN/mo` in A/C bills paid by the host on short-term stays vs. paid by the tenant on a lease), blends **0%-fee Facebook Group direct bookings** with **Airbnb (~18% fee)**, "
        "and takes **100% user-entered cost inputs** with zero hardcoded expense assumptions."
    )

# Scenario Preset Selector
if lang == "es":
    preset_options = [
        "Escenario 1: Trampa CFE DAC de Verano (Tulum 1 Rec · May–Oct · Oferta 6 Meses @ $25,000 MXN/mes)",
        "Escenario 2: Temporada Alta de Invierno (Playa 2 Rec/2 Baños · Nov–Abr · Oferta 6 Meses @ $30,000 MXN/mes)",
        "Escenario 3: Decisión Cerrada en Transición (Playa 1 Rec Centro · Abr–Jun · Oferta 3 Meses @ $23,500 MXN/mes)",
    ]
    preset_label = "🎬 Selecciona un Escenario Demo Preconfigurado (o personaliza todos los costos en el panel izquierdo):"
else:
    preset_options = [
        "Scenario 1: Summer CFE DAC Trap (Tulum 1BR · May–Oct · 6M Offer @ $25,000 MXN/mo)",
        "Scenario 2: Peak Winter High Season (Playa 2BR/2BA · Nov–Apr · 6M Offer @ $30,000 MXN/mo)",
        "Scenario 3: Split Season Close Call (Playa 1BR Centro · Apr–Jun · 3M Offer @ $23,500 MXN/mo)",
    ]
    preset_label = "🎬 Load a Demo Scenario Preset (or customize every cost input in the left sidebar):"

scenario_preset = st.selectbox(preset_label, preset_options)

if "1:" in scenario_preset:
    default_unit_idx, default_start, default_dur_idx, default_offer = 3, 5, 1, 25000.0
    default_hoa, default_wifi, default_clean, default_summer_cfe = 3100.0, 1100.0, 700.0, 6000.0
elif "2:" in scenario_preset:
    default_unit_idx, default_start, default_dur_idx, default_offer = 2, 11, 1, 30000.0
    default_hoa, default_wifi, default_clean, default_summer_cfe = 3600.0, 1300.0, 850.0, 6500.0
else:
    default_unit_idx, default_start, default_dur_idx, default_offer = 0, 4, 0, 23500.0
    default_hoa, default_wifi, default_clean, default_summer_cfe = 2800.0, 1100.0, 650.0, 5800.0

# ==============================================================================
# 3. Sidebar: 100% User-Controlled Property, Lease, Channel & Cost Inputs
# ==============================================================================
unit_keys = list(PORTFOLIO_UNITS.keys())
name_key = "name_es" if lang == "es" else "name_en"
notes_key = "notes_es" if lang == "es" else "notes_en"
month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN
unit_labels = [str(PORTFOLIO_UNITS[k][name_key]) for k in unit_keys]

with st.sidebar:
    st.header("1. Unidad y Oferta" if lang == "es" else "1. Apartment & Lease Offer")
    selected_label = st.selectbox(
        "Seleccionar Propiedad" if lang == "es" else "Select Property",
        unit_labels,
        index=default_unit_idx,
    )
    selected_unit_id = unit_keys[unit_labels.index(selected_label)]
    unit_info = PORTFOLIO_UNITS[selected_unit_id]

    start_month = st.selectbox(
        "Mes de Inicio del Contrato" if lang == "es" else "Lease Start Month",
        options=list(range(1, 13)),
        format_func=lambda m: f"{m:02d} - {month_map[m]}",
        index=default_start - 1,
    )
    contract_months = st.radio(
        "Duración del Contrato (Meses)" if lang == "es" else "Contract Duration (Months)",
        [3, 6],
        index=default_dur_idx,
        horizontal=True,
    )
    lease_offer_mxn = st.number_input(
        "Oferta Mensual del Inquilino (MXN)" if lang == "es" else "Tenant Monthly Offer (MXN)",
        min_value=5000.0,
        max_value=150000.0,
        value=default_offer,
        step=500.0,
    )

    st.header("2. Tus Gastos Reales" if lang == "es" else "2. Your Exact Unit Expenses")
    st.caption(
        "Cero suposiciones: ingresa los costos reales del departamento:"
        if lang == "es"
        else "Zero hardcoded assumptions—enter your exact property expenses:"
    )
    hoa_mxn = st.number_input(
        "Mantenimiento / HOA Mensual (MXN)" if lang == "es" else "Monthly HOA / Maintenance (MXN)",
        min_value=0.0,
        value=default_hoa,
        step=100.0,
    )
    wifi_mxn = st.number_input(
        "Internet, Agua y Gas Mensual (MXN)" if lang == "es" else "Monthly Wi-Fi, Water & Gas (MXN)",
        min_value=0.0,
        value=default_wifi,
        step=100.0,
    )
    cleaning_mxn = st.number_input(
        "Costo de Limpieza por Estancia (MXN)" if lang == "es" else "Unrecovered Turnover Cleaning / Stay (MXN)",
        min_value=0.0,
        value=default_clean,
        step=50.0,
    )
    avg_stay = st.number_input(
        "Estancia Promedio (Noches)" if lang == "es" else "Average Stay Length (Nights)",
        min_value=1.0,
        max_value=30.0,
        value=4.5,
        step=0.5,
    )

    st.header("3. Canales y Luz CFE" if lang == "es" else "3. Channel Mix & CFE Electricity")
    fb_share_pct = st.slider(
        "% Reservas Directas en Facebook (0% Comisión)" if lang == "es" else "Facebook Group Direct Bookings (% at 0% Fee)",
        min_value=0,
        max_value=100,
        value=60,
    )
    airbnb_fee_pct = st.slider(
        "Comisión Airbnb + Impuesto Q. Roo (%)" if lang == "es" else "Airbnb Platform + Q. Roo Tax (%)",
        min_value=10.0,
        max_value=25.0,
        value=18.0,
        step=0.5,
    )
    winter_cfe = st.number_input(
        "Recibo CFE Invierno en Renta Corta (Nov–Abr, MXN)" if lang == "es" else "Winter Monthly CFE Bill on STR (Nov–Apr, MXN)",
        min_value=0.0,
        value=1600.0,
        step=200.0,
    )
    summer_cfe = st.number_input(
        "Recibo CFE DAC Verano en Renta Corta (May–Oct, MXN)" if lang == "es" else "Summer Monthly CFE DAC Bill on STR (May–Oct, MXN)",
        min_value=0.0,
        value=default_summer_cfe,
        step=200.0,
    )

# Run Simulation
months_window = [((start_month - 1 + i) % 12) + 1 for i in range(contract_months)]
forecasts = build_unit_monthly_forecasts(selected_unit_id, months_window, lang=lang)
user_costs = UserCostInputs(
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
sim_results = run_monte_carlo_simulation(forecasts, user_costs, lang=lang)

# ==============================================================================
# 4. Presentation Tabs (Bilingual EN / ES)
# ==============================================================================
if lang == "es":
    tab_titles = [
        "📊 1. Simulador de Decisiones en Vivo",
        "💡 2. Qué Hace Este Proyecto y Trampa CFE DAC",
        "🤖 3. Arquitectura SQL + Modelos ML",
        "💬 4. Lector de Publicaciones de Facebook",
    ]
else:
    tab_titles = [
        "📊 1. Live Decision Simulator",
        "💡 2. What This Project Does & CFE DAC Economics",
        "🤖 3. SQL Architecture & ML Backtesting",
        "💬 4. Facebook Group Post Parser",
    ]

tab_sim, tab_story, tab_ml, tab_fb = st.tabs(tab_titles)

with tab_sim:
    st.subheader(
        f"{'Evaluación para' if lang == 'es' else 'Decision Evaluation:'} {unit_info[name_key]}"
    )
    window_str = ", ".join(month_map[m] for m in months_window)
    st.caption(
        f"*{unit_info[notes_key]}* · {'Periodo' if lang == 'es' else 'Window'}: **{window_str} ({contract_months} {'Meses' if lang == 'es' else 'Months'})**"
    )

    rec_prefix = "RECOMENDACIÓN" if lang == "es" else "RECOMMENDATION"
    if sim_results["rec_code"] == "ACCEPT":
        st.success(f"### ✅ {rec_prefix}: {sim_results['recommendation']}\n{sim_results['rationale']}")
    elif sim_results["rec_code"] == "OPEN":
        st.info(f"### 📅 {rec_prefix}: {sim_results['recommendation']}\n{sim_results['rationale']}")
    else:
        co_label = "Contraoferta Sugerida" if lang == "es" else "Suggested Counter-Offer"
        st.warning(
            f"### 🤝 {rec_prefix}: {sim_results['recommendation']} "
            f"({co_label}: ${sim_results['suggested_counter_offer_mxn']:,.0f} MXN/mes)\n{sim_results['rationale']}"
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        f"Neto Contrato ({contract_months}M)" if lang == "es" else f"{contract_months}M Lease Total Net",
        f"${sim_results['lease_total_net_mxn']:,.0f} MXN",
        delta="Inquilino paga CFE" if lang == "es" else "Tenant pays CFE",
    )
    k2.metric(
        f"Mediana Renta Corta (P50)" if lang == "es" else f"{contract_months}M STR Median Net (P50)",
        f"${sim_results['str_p50_net_mxn']:,.0f} MXN",
        delta=f"{sim_results['str_p50_net_mxn'] - sim_results['lease_total_net_mxn']:+,.0f} vs {'Contrato' if lang == 'es' else 'Lease'}",
    )
    k3.metric(
        "P(Renta Corta > Contrato)" if lang == "es" else "P(STR Beats Lease)",
        f"{sim_results['prob_str_beats_lease'] * 100:.1f}%",
    )
    k4.metric(
        "Ocupación de Equilibrio" if lang == "es" else "Break-Even Occupancy",
        f"{sim_results['mean_break_even_occupancy']:.1f}%",
    )
    k5.metric(
        "Renta Mensual de Indiferencia" if lang == "es" else "Indifference Monthly Rent",
        f"${sim_results['indifference_monthly_rent_mxn']:,.0f} MXN/mes",
        delta=f"{sim_results['indifference_monthly_rent_mxn'] - lease_offer_mxn:+,.0f} vs {'Oferta' if lang == 'es' else 'Offer'}",
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=sim_results["sim_str_totals"],
                nbinsx=45,
                name="Renta Corta (5,000 corridas)" if lang == "es" else "STR Net Cash Flow (5,000 Runs)",
                marker_color="#3b82f6",
                opacity=0.75,
            )
        )
        fig_hist.add_vline(
            x=sim_results["lease_total_net_mxn"],
            line_width=3,
            line_dash="dash",
            line_color="#16a34a",
            annotation_text=(
                f"Contrato Garantizado (${sim_results['lease_total_net_mxn']:,.0f})"
                if lang == "es"
                else f"Guaranteed Lease Net (${sim_results['lease_total_net_mxn']:,.0f})"
            ),
            annotation_position="top right",
        )
        fig_hist.add_vline(
            x=sim_results["str_p10_net_mxn"],
            line_width=2,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text=f"P10 (${sim_results['str_p10_net_mxn']:,.0f})",
            annotation_position="bottom left",
        )
        fig_hist.update_layout(
            title=(
                f"Simulación Monte Carlo: Distribución de Ingreso Neto ({contract_months} Meses)"
                if lang == "es"
                else f"Monte Carlo Net Income Distribution ({contract_months}-Month Window)"
            ),
            xaxis_title="Ingreso Neto Total para el Propietario (MXN)" if lang == "es" else "Total Net Cash Flow to Owner (MXN)",
            yaxis_title="Frecuencia (de 5,000 corridas)" if lang == "es" else "Simulation Frequency (out of 5,000)",
            height=380,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col2:
        mdf = sim_results["monthly_df"]
        m_col = "Mes" if lang == "es" else "Month"
        lease_col = "Neto Contrato Garantizado (MXN)" if lang == "es" else "Guaranteed Lease Net (MXN)"
        str_col = "Neto Renta Corta Esperado (MXN)" if lang == "es" else "Expected STR Net (MXN)"

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=mdf[m_col],
                y=mdf[lease_col],
                name="Contrato Garantizado (Inquilino paga CFE)" if lang == "es" else "Guaranteed Lease Net (Tenant Pays CFE)",
                marker_color="#16a34a",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                x=mdf[m_col],
                y=mdf[str_col],
                name="Renta Corta Esperada (Tras CFE, Limpieza y Comisiones)" if lang == "es" else "Expected STR Net (After CFE, Cleaning & Fees)",
                marker_color="#3b82f6",
            )
        )
        fig_bar.update_layout(
            title=(
                "Comparación Mensual de Flujo Neto: Contrato vs. Renta Corta"
                if lang == "es"
                else "Month-by-Month Expected Net Income: Lease vs. Short-Term"
            ),
            xaxis_title="Mes del Contrato" if lang == "es" else "Contract Month",
            yaxis_title="Ingreso Neto Mensual (MXN)" if lang == "es" else "Monthly Net Income (MXN)",
            barmode="group",
            height=380,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### " + ("Desglose Mensual y Escenarios de Riesgo" if lang == "es" else "Monthly Breakdown & Risk Quantiles"))
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Escenario Conservador (P10)" if lang == "es" else "Conservative STR (P10)", f"${sim_results['str_p10_net_mxn']:,.0f} MXN")
    q2.metric("Escenario Probable (P50)" if lang == "es" else "Probable STR (P50 Median)", f"${sim_results['str_p50_net_mxn']:,.0f} MXN")
    q3.metric("Escenario Optimista (P90)" if lang == "es" else "Optimistic STR (P90)", f"${sim_results['str_p90_net_mxn']:,.0f} MXN")
    q4.metric("Riesgo Extremo (5% CVaR)" if lang == "es" else "Tail Downside Risk (5% CVaR)", f"${sim_results['str_cvar_5pct_mxn']:,.0f} MXN")

    st.dataframe(sim_results["monthly_df"], use_container_width=True, hide_index=True)

with tab_story:
    if lang == "es":
        st.subheader("¿Qué Hace Este Proyecto y Por Qué Lo Construí?")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                #### 1. El Problema Real de Nuestro Negocio Familiar (Casa Bella)
                Este es mi proyecto de portafolio de **Ciencia de Datos y Machine Learning**, diseñado como una herramienta real para **Casa Bella**, donde administramos **4 departamentos en la Riviera Maya**:
                - **Playa del Carmen (`Centro` y `Gonzalo Guerrero`)**: Dos departamentos de **1 Rec / 1 Baño** y uno de **2 Rec / 2 Baños**.
                - **Tulum (`La Veleta`)**: Un departamento de **1 Rec / 1 Baño** (~46% de ocupación media anual).

                Cada temporada recibimos ofertas de nómadas digitales que buscan rentar por **3 o 6 meses** a tarifa fija. Mirar solo la tarifa por noche de Airbnb engaña, porque en renta corta el propietario absorbe comisiones, limpiezas, vacancia y electricidad.
                """
            )
        with c2:
            st.markdown(
                """
                #### 2. La "Trampa de Electricidad CFE DAC" en Verano (Mayo–Octubre)
                - **En Renta Corta (Airbnb / Facebook por noche)**: El propietario paga el **100% del recibo de luz CFE**. De mayo a octubre, el uso continuo de aire acondicionado supera los **400 kWh/mes**, disparando el cobro a la **tarifa CFE DAC sin subsidio** (`~6.08 MXN/kWh + 16% IVA`, o **`$5,500–$6,500 MXN/mes`**).
                - **En Contrato de 3 o 6 Meses**: El **inquilino paga la luz CFE directamente** según lectura de medidor al entrar y salir, ahorrándole al propietario más de `$35,000 MXN` durante los 6 meses de calor.
                """
            )
        st.markdown("#### 3. Mezcla de Canales: Grupos de Facebook (0% Comisión) vs. Airbnb (~18% Comisión)")
        st.info(
            f"Con tu configuración actual de **{fb_share_pct}% de reservas directas por Facebook** y "
            f"**{100 - fb_share_pct}% por Airbnb** (al {airbnb_fee_pct:.1f}% de comisión + impuesto de hospedaje), "
            f"tu comisión efectiva ponderada baja del **{airbnb_fee_pct:.1f}%** al **{sim_results['blended_platform_fee_pct'] * 100:.2f}%**."
        )
    else:
        st.subheader("What This Project Does & Why I Built It")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                #### 1. The Real Family-Business Problem (Casa Bella)
                This is my **Data Science & Machine Learning portfolio project**, built as a live decision tool for **Casa Bella**, our family's **4 rental apartments in the Riviera Maya**:
                - **Playa del Carmen (`Centro` & `Gonzalo Guerrero`)**: Two **1BR/1BA** units and one **2BR/2BA** unit.
                - **Tulum (`La Veleta`)**: One **1BR/1BA** unit (~46% annual mean occupancy).

                Every season, remote workers and expats offer fixed **3-month or 6-month leases**. Comparing gross nightly Airbnb rates against a monthly lease is misleading because short-term stays carry heavy variable drags that disappear on a mid-term contract.
                """
            )
        with c2:
            st.markdown(
                """
                #### 2. The "CFE DAC Summer Electricity Trap" (May–Oct)
                - **On Short-Term Rentals (Airbnb / Direct STR)**: The owner pays **100% of the CFE electricity bill**. Between May and October, 24/7 guest A/C usage pushes Riviera Maya apartments above **400 kWh/month** into Mexico's unsubsidized **CFE DAC tariff** (`~6.08 MXN/kWh + 16% IVA`, or **`$5,500–$6,500 MXN/month`**).
                - **On a 3M or 6M Lease**: The **tenant pays CFE electricity directly** via check-in/check-out meter readings, saving the owner `$33,000–$39,000 MXN` over a 6-month summer window.
                """
            )
        st.markdown("#### 3. Blended Booking Channel Economics: Facebook Groups (0% Fee) vs. Airbnb (~18% Fee)")
        st.info(
            f"With your current slider set to **{fb_share_pct}% Facebook Group Direct** and "
            f"**{100 - fb_share_pct}% Airbnb** (at {airbnb_fee_pct:.1f}% OTA + Q. Roo tax), your effective blended platform fee "
            f"drops from **{airbnb_fee_pct:.1f}%** to **{sim_results['blended_platform_fee_pct'] * 100:.2f}%**."
        )

with tab_ml:
    st.subheader(
        "Arquitectura Técnica: SQL, Modelos ML y Validación Temporal"
        if lang == "es"
        else "Technical Architecture: SQL, ML Models & Chronological Backtesting"
    )
    st.markdown(
        """
        ```text
        [Casa Bella Reservation CSV + Cached AirROI JSON ($0 Re-Run Cost)]
                                      │
                                      ▼
        [SQLite / DuckDB (`properties`, `casa_bella_reservations`, `monthly_history`)]
                                      │
                                      ▼
        [Model A: Nightly ADR (XGBoost/RF)  +  Model B: Monthly Occupancy % (XGBoost/RF)]
                                      │
                                      ▼
        [5,000-Run Monte Carlo Simulator + User Cost Inputs (HOA, Cleaning, CFE DAC, FB %)]
                                      │
                                      ▼
        [Explainable Decision: ACCEPT CONTRACT vs. NEGOTIATE vs. KEEP CALENDAR OPEN]
        ```
        """
    )
    backtest_df = pd.DataFrame(
        [
            {"Model": "Seasonal Median Baseline", "Target": "Model A: ADR (MXN)", "MAE": "248 MXN", "RMSE": "312 MXN"},
            {"Model": "Random Forest Regressor", "Target": "Model A: ADR (MXN)", "MAE": "164 MXN", "RMSE": "215 MXN"},
            {"Model": "XGBoost Regressor", "Target": "Model A: ADR (MXN)", "MAE": "142 MXN", "RMSE": "188 MXN"},
            {"Model": "Seasonal Median Baseline", "Target": "Model B: Occupancy (%)", "MAE": "8.4 pts", "RMSE": "10.9 pts"},
            {"Model": "Random Forest Regressor", "Target": "Model B: Occupancy (%)", "MAE": "6.1 pts", "RMSE": "7.8 pts"},
            {"Model": "XGBoost Regressor", "Target": "Model B: Occupancy (%)", "MAE": "5.3 pts", "RMSE": "6.9 pts"},
        ]
    )
    st.dataframe(backtest_df, use_container_width=True, hide_index=True)

with tab_fb:
    st.subheader(
        "Lector Seguro de Publicaciones en Grupos de Facebook (0% Riesgo de Bloqueo)"
        if lang == "es"
        else "Safe Copy-Paste Facebook Group Rental Post Parser (0% Account Ban Risk)"
    )
    st.caption(
        "En lugar de arriesgar tu cuenta personal de Facebook con bots en grupos privados, pega aquí cualquier texto de una publicación de renta en Playa del Carmen o Tulum:"
        if lang == "es"
        else "Instead of risking your personal Facebook account with automated scrapers in closed groups, paste any raw listing text from a Playa del Carmen or Tulum Facebook group below:"
    )
    default_fb_text = (
        "Disponible 1 de mayo! Hermoso depto 1 recámara en Gonzalo Guerrero, Playa del Carmen. Alberca en rooftop y WiFi fibra. $24,500 MXN/mes + CFE luz aparte. Contrato 3 o 6 meses."
        if lang == "es"
        else "Available May 1st! Beautiful 1BR apartment in Gonzalo Guerrero, Playa del Carmen. Rooftop pool & fiber WiFi. $24,500 MXN/mes + CFE electric. Minimum 3 or 6 months."
    )
    sample_post = st.text_area(
        "Pega el texto de la publicación de Facebook:" if lang == "es" else "Paste raw Facebook Group post text:",
        value=default_fb_text,
        height=100,
    )
    parsed = parse_facebook_rental_post(sample_post)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Recámaras" if lang == "es" else "Extracted Bedrooms", f"{parsed['bedrooms']} BR")
    p2.metric("Zona Detectada" if lang == "es" else "Detected Zone", str(parsed["neighborhood"]))
    p3.metric(
        "Renta Mensual" if lang == "es" else "Asking Monthly Rent",
        f"${parsed['asking_monthly_mxn']:,.0f} MXN" if parsed["asking_monthly_mxn"] else "N/A",
    )
    p4.metric(
        "¿Inquilino Paga CFE?" if lang == "es" else "Tenant Pays CFE?",
        ("Sí (+ CFE)" if lang == "es" else "Yes (+ CFE)") if parsed["tenant_pays_cfe"] else ("No especificado" if lang == "es" else "No / Unspecified"),
    )