"""
Rental Strategy Optimizer (Playa del Carmen, Tulum & CDMX)
Bilingual (English / Español) Portfolio Demo & Interactive Decision Engine

Single entry point — imports all domain assumptions, CFE 12-month rolling DAC
billing, ML forecasting models, and Monte Carlo simulation from `src.rental_engine`.

Run locally with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.rental_engine.assumptions import (
    AIRDNA_PUBLIC_BENCHMARKS,
    CFE_TARIFF_DAC_LIMITS_KWH,
    DEFAULT_ASSUMPTIONS,
    MONTH_NAMES_EN,
    MONTH_NAMES_ES,
    PORTFOLIO_UNITS,
)
from src.rental_engine.cfe import evaluate_12m_dac_trajectory
from src.rental_engine.data import load_training_dataframe
from src.rental_engine.engine import (
    UserCostInputs,
    simulate_lease_vs_str,
)
from src.rental_engine.fb_parser import parse_facebook_rental_post
from src.rental_engine.models import (
    TrainedModelBundle,
    predict_unit_monthly_forecasts,
    train_and_evaluate_models,
)


@st.cache_resource(show_spinner=False)
def get_trained_model_bundle() -> TrainedModelBundle:
    """
    Loads the synthetic SQLite panel and fits the Seasonal Median Baseline,
    Random Forest, and XGBoost models with a strict chronological split
    (train <= 2025-06, test > 2025-06). Cached across reruns.
    """
    df = load_training_dataframe()
    return train_and_evaluate_models(df, cutoff_year_month="2025-06")


# ==============================================================================
# 1. Page Header, Language Switcher (EN / ES) & Demo Disclaimer Banner
# ==============================================================================
st.set_page_config(
    page_title="Rental Strategy Optimizer — Riviera Maya & CDMX",
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

st.markdown(
    """
    <style>
    .header-title {
        font-size: 3.0rem !important;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }
    .header-caption {
        font-size: 1.15rem !important;
        line-height: 1.6;
        color: #6b7280;
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with top_col1:
    if lang == "es":
        st.markdown(
            "<h1 class='header-title'>🏖️ Optimizador de Estrategia de Renta</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='header-caption'><strong>Creado por Erik Šauša, Dipl.-Ing. (M.Sc.), B.Sc.</strong><br>"
            "<em>Maestría en Ciencias en Telecomunicaciones · Licenciatura en Ingeniería Eléctrica y Tecnologías de la Información (TU Wien)</em><br>"
            "<strong>🚧 Motor Interactivo de Decisiones Inmobiliarias (Playa del Carmen, Tulum y CDMX)</strong></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h1 class='header-title'>🏖️ Rental Strategy Optimizer</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='header-caption'><strong>Created by Erik Šauša, Dipl.-Ing. (M.Sc.), B.Sc.</strong><br>"
            "<em>Master of Science in Telecommunications · Bachelor of Science in Electrical Engineering and Information Technology (TU Wien)</em><br>"
            "<strong>🚧 Interactive Decision Engine Demo (Playa del Carmen, Tulum & CDMX)</strong></div>",
            unsafe_allow_html=True,
        )

# Explicit Demo & Synthetic Data Warning Banner
if lang == "es":
    st.warning(
        "⚠️ **Demo Público (Datos Sintéticos Ilustrativos):** "
        "Para proteger la privacidad del negocio y de los huéspedes, este repositorio utiliza **datos mensuales sintéticos** "
        "(`source = 'synthetic_demo'`) para 4 unidades ilustrativas (**Unidad A Playa**, **Unidad B Playa**, **Unidad C Playa** y **Unidad Tulum**). "
        "Las cifras de mercado (como el ~46% de ocupación anual en Tulum y ~58% en Playa del Carmen) provienen de benchmarks públicos de **AirDNA (2026)**."
    )
    st.info(
        "🎯 **Sobre Este Proyecto:** "
        "Motor de decisiones de extremo a extremo (**SQL $\\rightarrow$ Modelos ML con Validación Temporal $\\rightarrow$ Simulación Monte Carlo $\\rightarrow$ Streamlit**) "
        "diseñado para responder una pregunta operativa real: *cuando un inquilino ofrece un contrato fijo de 3 o 6 meses, ¿conviene aceptar la oferta hoy, "
        "negociar una renta más alta o dejar el calendario abierto para rentas cortas por Airbnb y grupos de Facebook?* "
        "Modela la regla real de reclasificación **CFE DAC por promedio móvil de 12 meses** (Tarifa 1B: límite de `400 kWh/mes`), "
        "la mezcla de reservas directas en **Grupos de Facebook (0% comisión)** vs. **Airbnb (~18%)**, y el número exacto de días calendario de cada mes."
    )
else:
    st.warning(
        "⚠️ **Public Demo (Illustrative Synthetic Inputs):** "
        "To protect business and guest privacy, this repository uses **synthetic monthly history** (`source = 'synthetic_demo'`) "
        "for 4 anonymized units (**Unit A Playa**, **Unit B Playa**, **Unit C Playa**, and **Unit Tulum**). "
        "Market-level reference figures (such as Tulum's ~46% and Playa del Carmen's ~58% annual mean occupancy) are cited from public **AirDNA (2026)** benchmarks."
    )
    st.info(
        "🎯 **About This Project:** "
        "End-to-end decision engine (**SQL $\\rightarrow$ Chronological ML Models $\\rightarrow$ Monte Carlo Simulation $\\rightarrow$ Streamlit**) "
        "built to answer a real operational question: *when a prospective tenant offers a guaranteed 3-month or 6-month lease, should we accept the contract today, "
        "negotiate a higher rate, or keep the calendar open for short-term stays across Airbnb and Facebook Groups?* "
        "Models CFE's **12-month rolling average DAC electricity reclassification rule** (Tariff 1B: `400 kWh/month` rolling threshold), "
        "blends **0%-fee Facebook Group direct bookings** with **Airbnb (~18% fee)**, and uses exact calendar days per month."
    )

# Train / fetch cached ML models
model_bundle = get_trained_model_bundle()

# Scenario Preset Selector
if lang == "es":
    preset_options = [
        "Escenario 1: Trampa CFE DAC de Verano (Unidad Tulum · May–Oct · Oferta 6 Meses @ $25,000 MXN/mes)",
        "Escenario 2: Temporada Alta de Invierno (Unidad C Playa · Nov–Abr · Oferta 6 Meses @ $30,000 MXN/mes)",
        "Escenario 3: Decisión Cerrada en Transición (Unidad A Playa · Abr–Jun · Oferta 3 Meses @ $23,500 MXN/mes)",
    ]
    preset_label = "🎬 Selecciona un Escenario Demo Preconfigurado (o personaliza los costos en el panel izquierdo):"
else:
    preset_options = [
        "Scenario 1: Summer CFE DAC Trap (Unit Tulum · May–Oct · 6M Offer @ $25,000 MXN/mo)",
        "Scenario 2: Peak Winter High Season (Unit C Playa · Nov–Apr · 6M Offer @ $30,000 MXN/mo)",
        "Scenario 3: Split Season Close Call (Unit A Playa · Apr–Jun · 3M Offer @ $23,500 MXN/mo)",
    ]
    preset_label = "🎬 Load a Demo Scenario Preset (or customize any cost input in the left sidebar):"

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
# 2. Sidebar: Property, Lease, Channel & CFE Inputs
# ==============================================================================
unit_keys = list(PORTFOLIO_UNITS.keys())
name_key = "name_es" if lang == "es" else "name_en"
notes_key = "notes_es" if lang == "es" else "notes_en"
month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN
unit_labels = [str(PORTFOLIO_UNITS[k][name_key]) for k in unit_keys]

with st.sidebar:
    st.markdown("### 👨‍💻 Created By")
    if lang == "es":
        st.markdown(
            """
            **Erik Šauša, Dipl.-Ing. (M.Sc.), B.Sc.**
            - 🎓 **Maestría en Ciencias en Telecomunicaciones** *(TU Wien)*
            - 🎓 **Licenciatura en Ingeniería Eléctrica y Tecnología de la Información** *(TU Wien)*
            - 💼 **Científico de Datos & Ingeniero de ML**
            """
        )
    else:
        st.markdown(
            """
            **Erik Šauša, Dipl.-Ing. (M.Sc.), B.Sc.**
            - 🎓 **Master of Science in Telecommunications** *(TU Wien)*
            - 🎓 **Bachelor of Science in Electrical Engineering and Information Technology** *(TU Wien)*
            - 💼 **Data Scientist & ML Engineer**
            """
        )

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

    st.header("2. Gastos Operativos" if lang == "es" else "2. Operating Expenses")
    st.caption(
        "Ingresa los gastos operativos del departamento (valores iniciales ilustrativos):"
        if lang == "es"
        else "Enter the unit's operating expenses (defaults are illustrative):"
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

    cfe_mode = st.radio(
        "Modo de Estimación CFE" if lang == "es" else "CFE Electricity Input Mode",
        options=(
            ["Recibo Mensual Directo (MXN)", "Simulador kWh Promedio Móvil 12M (Tarifa 1B)"]
            if lang == "es"
            else ["Direct Monthly Bills (MXN)", "12M Rolling-Average kWh Mode (Tariff 1B)"]
        ),
    )
    use_kwh_mode = "kWh" in cfe_mode

    prior_11m_kwh = None
    window_monthly_kwh = None
    if use_kwh_mode:
        prior_avg_kwh = st.slider(
            "Promedio kWh/mes (11 meses previos)" if lang == "es" else "Prior 11-Month Avg Consumption (kWh/mo)",
            min_value=150,
            max_value=900,
            value=390,
            step=10,
        )
        winter_kwh = st.slider(
            "Consumo Mensual Invierno en Renta Corta (kWh)" if lang == "es" else "Winter STR Monthly Usage (Nov–Apr, kWh)",
            min_value=150,
            max_value=800,
            value=280,
            step=20,
        )
        summer_kwh = st.slider(
            "Consumo Mensual Verano A/C en Renta Corta (kWh)" if lang == "es" else "Summer STR Monthly Usage (May–Oct, kWh)",
            min_value=300,
            max_value=1200,
            value=780,
            step=20,
        )
        winter_cfe = 1600.0
        summer_cfe = default_summer_cfe
    else:
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

# ==============================================================================
# 3. Forecast & Monte Carlo Execution (via `src.rental_engine`)
# ==============================================================================
months_window = [((start_month - 1 + i) % 12) + 1 for i in range(contract_months)]
forecasts = predict_unit_monthly_forecasts(
    model_bundle, selected_unit_id, months_window, lang=lang
)

if use_kwh_mode:
    prior_11m_kwh = [float(prior_avg_kwh)] * 11
    window_monthly_kwh = [
        float(summer_kwh if m in DEFAULT_ASSUMPTIONS.summer_months else winter_kwh)
        for m in months_window
    ]

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

sim_results = simulate_lease_vs_str(
    forecasts,
    user_costs,
    lang=lang,
    prior_11m_kwh=prior_11m_kwh,
    window_monthly_kwh=window_monthly_kwh,
)

# ==============================================================================
# 4. Presentation Tabs (Bilingual EN / ES)
# ==============================================================================
if lang == "es":
    tab_titles = [
        "📊 1. Simulador de Decisiones en Vivo",
        "💡 2. Regla CFE DAC (Promedio 12M) y Supuestos",
        "🤖 3. Arquitectura SQL + Backtesting ML Real",
        "💬 4. Lector de Publicaciones de Facebook",
    ]
else:
    tab_titles = [
        "📊 1. Live Decision Simulator",
        "💡 2. CFE DAC 12M Rolling Rule & Assumptions",
        "🤖 3. SQL Architecture & Computed ML Backtesting",
        "💬 4. Facebook Group Post Parser",
    ]

tab_sim, tab_story, tab_ml, tab_fb = st.tabs(tab_titles)

with tab_sim:
    st.subheader(
        f"{'Evaluación para' if lang == 'es' else 'Decision Evaluation:'} {unit_info[name_key]}"
    )
    window_str = ", ".join(month_map[m] for m in months_window)
    st.caption(
        f"*{unit_info[notes_key]}* · {'Periodo' if lang == 'es' else 'Window'}: "
        f"**{window_str} ({contract_months} {'Meses' if lang == 'es' else 'Months'})** · "
        f"{'Modelo Activo' if lang == 'es' else 'Active Forecast Model'}: **{model_bundle.active_model_name}** "
        f"(σ_ADR = {model_bundle.adr_residual_std_mxn:.0f} MXN, σ_Occ = {model_bundle.occ_residual_std * 100:.1f} pts)"
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
        "Mediana Renta Corta (P50)" if lang == "es" else f"{contract_months}M STR Median Net (P50)",
        f"${sim_results['str_p50_net_mxn']:,.0f} MXN",
        delta=f"{sim_results['str_p50_net_mxn'] - sim_results['lease_total_net_mxn']:+,.0f} vs {'Contrato' if lang == 'es' else 'Lease'}",
    )
    k3.metric(
        "P(Renta Corta > Contrato)" if lang == "es" else "P(STR Beats Lease)",
        f"{sim_results['prob_str_beats_lease'] * 100:.1f}%",
    )
    k4.metric(
        "Ocupación de Equilibrio" if lang == "es" else "Break-Even Occupancy",
        f"{sim_results['mean_break_even_occupancy_pct']:.1f}%",
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
                "Comparación Mensual de Flujo Neto (Días Exactos del Mes)"
                if lang == "es"
                else "Month-by-Month Expected Net Income (Exact Calendar Days)"
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

    if sim_results["cfe_trajectory"] is not None:
        st.markdown(
            "##### ⚡ "
            + (
                "Trayectoria CFE Promedio Móvil 12 Meses (Tarifa 1B: Límite 400 kWh/mes)"
                if lang == "es"
                else "12-Month Rolling Average CFE DAC Trajectory (Tariff 1B: 400 kWh/mo Limit)"
            )
        )
        st.dataframe(pd.DataFrame(sim_results["cfe_trajectory"]), use_container_width=True, hide_index=True)

with tab_story:
    if lang == "es":
        st.subheader("Regla Oficial CFE DAC (Promedio Móvil 12 Meses) y Supuestos Explícitos")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                #### 1. El Problema Operativo en la Riviera Maya
                Este proyecto evalúa **4 departamentos ilustrativos en la Riviera Maya** (basados en la estructura de un negocio familiar real, anonimizados para este demo público):
                - **Playa del Carmen**: **Unidad A Playa** y **Unidad B Playa** (**1 Rec / 1 Baño**) y **Unidad C Playa** (**2 Rec / 2 Baños**).
                - **Tulum**: **Unidad Tulum** (**1 Rec / 1 Baño**, anclada al ~46% de ocupación media anual del mercado de Tulum según **AirDNA 2026**).

                Cada temporada, trabajadores remotos ofrecen contratos de **3 o 6 meses** a tarifa fija. Comparar solo la tarifa por noche de Airbnb contra una renta mensual lleva a malas decisiones si no se descuentan comisiones, limpiezas, vacancia y electricidad.
                """
            )
        with c2:
            st.markdown(
                """
                #### 2. Cómo Funciona Realmente la Tarifa CFE DAC (Promedio Móvil 12 Meses)
                - **Regla de los 12 Meses (CFE Tarifa 1B = 400 kWh/mes)**: La CFE **no** reclasifica un medidor por un solo mes alto. Calcula el **promedio móvil de los últimos 12 meses** (6 bimestres). Si ese promedio supera los **400 kWh/mes** en Tarifa 1B, el medidor cae en **Tarifa DAC sin subsidio** (`~6.08 MXN/kWh + 16% IVA`).
                - **Persistencia en Invierno**: Una vez en DAC, **incluso los meses de invierno** se cobran a tarifa DAC hasta que el promedio de 12 meses vuelva a bajar de `400 kWh/mes`.
                - **Efecto del Inquilino en el Medidor**: En un contrato de 3 o 6 meses el inquilino paga su recibo CFE, pero su consumo en kWh queda registrado en el historial de 12 meses del medidor.
                """
            )
    else:
        st.subheader("Official CFE DAC 12-Month Rolling Rule & Explicit Model Assumptions")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                #### 1. The Operational Problem in the Riviera Maya
                This project evaluates **4 anonymized portfolio units in the Riviera Maya** (modeled on a family-run rental operation, with synthetic history for privacy):
                - **Playa del Carmen**: **Unit A Playa** and **Unit B Playa** (**1BR/1BA**) and **Unit C Playa** (**2BR/2BA**).
                - **Tulum**: **Unit Tulum** (**1BR/1BA**, anchored to Tulum's ~46% market-wide annual occupancy per **AirDNA 2026**).

                Every season, remote workers offer fixed **3-month or 6-month leases**. Comparing gross nightly Airbnb rates against a monthly lease is misleading because short-term stays carry variable costs that disappear on a mid-term contract.
                """
            )
        with c2:
            st.markdown(
                """
                #### 2. How CFE DAC Reclassification Actually Works (12-Month Rolling Average)
                - **12-Month Rolling Rule (CFE Tariff 1B = 400 kWh/mo)**: CFE does **not** trigger DAC from a single month in isolation. It computes the **trailing 12-month average** consumption (6 bimonthly bills). When that rolling mean exceeds **400 kWh/month** under Tariff 1B, the meter loses government subsidy and enters **DAC** (`~6.08 MXN/kWh + 16% IVA`).
                - **Winter Persistence**: Once reclassified into DAC, **winter months are also billed at the DAC rate** until the 12-month rolling average falls back to or below `400 kWh/month`.
                - **Lease Meter Carryover**: On a 3M/6M lease, the tenant pays CFE directly, saving the owner `$33,000–$39,000 MXN` in summer bills—though heavy tenant A/C kWh still counts toward the meter's 12-month rolling window.
                """
            )

    st.markdown(
        "#### 3. "
        + (
            "Tabla de Supuestos Explícitos (`src/rental_engine/assumptions.py`)"
            if lang == "es"
            else "Explicit Modeling Assumptions (`src/rental_engine/assumptions.py`)"
        )
    )
    assumptions_table = pd.DataFrame(
        [
            {
                "Parameter": "cfe_tariff / cfe_dac_threshold_kwh",
                "Value": f"Tariff {DEFAULT_ASSUMPTIONS.cfe_tariff} ({DEFAULT_ASSUMPTIONS.cfe_dac_threshold_kwh:.0f} kWh/mo rolling 12M mean)",
                "Source / Rationale": "CFE Tarifas Domésticas & DAC (accessed 2026-11-15); 1C = 850 kWh/mo also supported",
            },
            {
                "Parameter": "cfe_dac_rate_mxn_per_kwh / cfe_subsidized_rate_mxn_per_kwh",
                "Value": f"{DEFAULT_ASSUMPTIONS.cfe_dac_rate_mxn_per_kwh:.2f} / {DEFAULT_ASSUMPTIONS.cfe_subsidized_rate_mxn_per_kwh:.2f} MXN/kWh (+16% IVA)",
                "Source / Rationale": "CFE Peninsular DAC vs. blended subsidized Tariff 1B energy rate",
            },
            {
                "Parameter": "adr_std_mxn / occ_std (Monte Carlo spread)",
                "Value": f"Computed from out-of-time backtest residuals ({model_bundle.adr_residual_std_mxn:.1f} MXN / {model_bundle.occ_residual_std * 100:.2f} pts)",
                "Source / Rationale": f"Out-of-time test split (year_month > {model_bundle.cutoff_year_month}); fallback {DEFAULT_ASSUMPTIONS.fallback_adr_std_mxn:.0f} MXN / {DEFAULT_ASSUMPTIONS.fallback_occ_std * 100:.0f} pts",
            },
            {
                "Parameter": "min_sampled_adr_mxn / occ clip bounds",
                "Value": f">= {DEFAULT_ASSUMPTIONS.min_sampled_adr_mxn:.0f} MXN | [{DEFAULT_ASSUMPTIONS.min_sampled_occ_rate * 100:.0f}%, {DEFAULT_ASSUMPTIONS.max_sampled_occ_rate * 100:.0f}%]",
                "Source / Rationale": "Prevents negative or >100% physical occupancy in normal draws",
            },
            {
                "Parameter": "Decision thresholds (ACCEPT / KEEP OPEN / Markup)",
                "Value": f"P(STR>Lease) < {DEFAULT_ASSUMPTIONS.accept_lease_max_str_win_prob:.2f} | > {DEFAULT_ASSUMPTIONS.keep_open_min_str_win_prob:.2f} | {DEFAULT_ASSUMPTIONS.counter_offer_risk_premium:.2f}x",
                "Source / Rationale": "Risk-averse decision boundaries & 5% counter-offer premium over indifference rent",
            },
            {
                "Parameter": "Public Market Benchmarks (Playa / Tulum / CDMX)",
                "Value": "Playa 58% occ ($103 ADR) · Tulum 46% occ ($125 ADR) · CDMX 66% occ ($105 ADR)",
                "Source / Rationale": "AirDNA MarketMinder public market summaries (accessed 2026)",
            },
        ]
    )
    st.dataframe(assumptions_table, use_container_width=True, hide_index=True)

with tab_ml:
    st.subheader(
        "Arquitectura SQL, Entrenamiento Real y Validación Temporal"
        if lang == "es"
        else "SQL Architecture, Real Model Training & Chronological Backtesting"
    )
    st.markdown(
        """
        ```text
        [SQLite Synthetic Demo Panel (`properties`, `reservations`, `monthly_history`, `market_benchmarks`)]
                                      │
                                      ▼
        [Chronological Split (`src/rental_engine/models.py`): Train <= 2025-06 (90 rows) | Test > 2025-06 (30 rows)]
                                      │
                                      ▼
        [Fit & Evaluate: Seasonal Median Baseline vs. RandomForestRegressor vs. XGBRegressor]
                                      │
                                      ▼
        [Extract Out-of-Time Residual Stds (σ_ADR, σ_Occ) + Point Forecasts -> 5,000-Run Monte Carlo Engine]
                                      │
                                      ▼
        [Explainable Decision: ACCEPT CONTRACT vs. NEGOTIATE vs. KEEP CALENDAR OPEN]
        ```
        """
    )
    st.caption(
        "📊 **Nota Metodológica Honesta:** Las métricas de la tabla inferior se calculan en vivo ejecutando `.fit()` en `year_month <= 2025-06` "
        "y evaluando en `year_month > 2025-06` sobre el panel sintético (`source = 'synthetic_demo'`). Demuestran que el pipeline temporal sin fuga de datos "
        "funciona de extremo a extremo, no precisión sobre datos privados."
        if lang == "es"
        else "📊 **Methodological Note:** The metrics below are computed live by running `.fit()` on `year_month <= 2025-06` and evaluating on "
        "`year_month > 2025-06` over the synthetic demo panel (`source = 'synthetic_demo'`). They verify the leak-free chronological pipeline and supply the "
        "residual standard deviations used in Monte Carlo, rather than claiming accuracy on private booking records."
    )
    st.dataframe(model_bundle.backtest_metrics_df, use_container_width=True, hide_index=True)

with tab_fb:
    st.subheader(
        "Lector Seguro de Publicaciones en Grupos de Facebook (0% Riesgo de Bloqueo)"
        if lang == "es"
        else "Safe Copy-Paste Facebook Group Rental Post Parser (0% Account Ban Risk)"
    )
    st.caption(
        "En lugar de usar bots en grupos privados, pega aquí cualquier texto de una publicación de renta en Playa del Carmen o Tulum:"
        if lang == "es"
        else "Instead of running automated scrapers in closed groups, paste any raw listing text from a Playa del Carmen or Tulum Facebook group below:"
    )
    default_fb_text = (
        "Disponible 1 de mayo! Hermoso depto 1 recámara en Aldea Zama, Tulum. Alberca en rooftop y WiFi fibra. $24,500 MXN/mes + CFE luz aparte. Contrato 3 o 6 meses."
        if lang == "es"
        else "Available May 1st! Beautiful 1BR apartment in Aldea Zama, Tulum. Rooftop pool & fiber WiFi. $24,500 MXN/mes + CFE electric. Minimum 3 or 6 months."
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
