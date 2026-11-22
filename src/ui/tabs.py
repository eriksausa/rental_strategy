"""
Tab renderers for the four sections of the Rental Strategy Optimizer dashboard.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.rental_engine.assumptions import (
    DEFAULT_ASSUMPTIONS,
    MONTH_NAMES_EN,
    MONTH_NAMES_ES,
    PORTFOLIO_UNITS,
)
from src.rental_engine.fb_parser import parse_facebook_rental_post
from src.rental_engine.forecast_service import ForecastArtifacts
from src.rental_engine.reporting import MODEL_LABELS

from .charts import build_mc_histogram, build_monthly_comparison_chart
from .i18n import pick
from .sidebar import SidebarInputs


def render_simulator_tab(
    inputs: SidebarInputs,
    sim_results: Dict[str, object],
    artifacts: ForecastArtifacts,
    wrapped_year_end: bool,
    lang: str,
) -> None:
    """Tab 1: Recommendation banner, KPI cards, Monte Carlo charts, and monthly breakdown."""
    unit_info = PORTFOLIO_UNITS[inputs.unit_id]
    name_key = "name_es" if lang == "es" else "name_en"
    notes_key = "notes_es" if lang == "es" else "notes_en"
    month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN

    if wrapped_year_end:
        st.info(
            pick(
                "This lease window crosses past the 12-month walk-forward horizon; months past December "
                "use the unit's 12-month seasonal walk-forward forecast paired with h=1..N residual blocks.",
                "Esta ventana de contrato cruza más allá del horizonte walk-forward de 12 meses; los meses "
                "posteriores a diciembre usan el pronóstico estacional de 12 meses con bloques de residuos h=1..N.",
                lang,
            )
        )

    chosen = artifacts.meta["chosen_models"]
    adr_model_label = MODEL_LABELS.get(chosen["adr"], chosen["adr"])
    occ_model_label = MODEL_LABELS.get(chosen["occ"], chosen["occ"])
    uncertainty_label = (
        pick(
            "block bootstrap of walk-forward residuals",
            "bootstrap de bloques de residuos (walk-forward)",
            lang,
        )
        if sim_results["uncertainty_method"] == "block_bootstrap"
        else pick(
            "independent normals (marginal walk-forward spreads)",
            "normales independientes (dispersión marginal walk-forward)",
            lang,
        )
    )
    st.caption(
        f"{pick('Uncertainty method', 'Método de incertidumbre', lang)}: **{uncertainty_label}** · "
        f"{pick('Chosen models', 'Modelos elegidos', lang)}: "
        f"ADR = **{adr_model_label}**, {pick('Occupancy', 'Ocupación', lang)} = **{occ_model_label}**"
    )

    st.subheader(f"{pick('Decision Evaluation:', 'Evaluación para', lang)} {unit_info[name_key]}")
    window_str = ", ".join(month_map[m] for m in inputs.months_window)
    st.caption(
        f"*{unit_info[notes_key]}* · {pick('Window', 'Periodo', lang)}: "
        f"**{window_str} ({inputs.contract_months} {pick('Months', 'Meses', lang)})**"
    )

    # Recommendation banner
    rec_prefix = pick("RECOMMENDATION", "RECOMENDACIÓN", lang)
    if sim_results["rec_code"] == "ACCEPT":
        st.success(f"### ✅ {rec_prefix}: {sim_results['recommendation']}\n{sim_results['rationale']}")
    elif sim_results["rec_code"] == "OPEN":
        st.info(f"### 📅 {rec_prefix}: {sim_results['recommendation']}\n{sim_results['rationale']}")
    else:
        co_label = pick("Suggested Counter-Offer", "Contraoferta Sugerida", lang)
        st.warning(
            f"### 🤝 {rec_prefix}: {sim_results['recommendation']} "
            f"({co_label}: ${sim_results['suggested_counter_offer_mxn']:,.0f} MXN/mes)\n{sim_results['rationale']}"
        )

    # Headline KPI row
    lease_net = float(sim_results["lease_total_net_mxn"])
    p50_net = float(sim_results["str_p50_net_mxn"])
    indiff_rent = float(sim_results["indifference_monthly_rent_mxn"])
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        pick(f"{inputs.contract_months}M Lease Total Net", f"Neto Contrato ({inputs.contract_months}M)", lang),
        f"${lease_net:,.0f} MXN",
        delta=pick("Tenant pays CFE", "Inquilino paga CFE", lang),
    )
    k2.metric(
        pick(f"{inputs.contract_months}M STR Median Net (P50)", "Mediana Renta Corta (P50)", lang),
        f"${p50_net:,.0f} MXN",
        delta=f"{p50_net - lease_net:+,.0f} vs {pick('Lease', 'Contrato', lang)}",
    )
    k3.metric(
        pick("P(STR Beats Lease)", "P(Renta Corta > Contrato)", lang),
        f"{float(sim_results['prob_str_beats_lease']) * 100:.1f}%",
    )
    k4.metric(
        pick("Break-Even Occupancy", "Ocupación de Equilibrio", lang),
        f"{float(sim_results['mean_break_even_occupancy_pct']):.1f}%",
    )
    k5.metric(
        pick("Indifference Monthly Rent", "Renta Mensual de Indiferencia", lang),
        f"${indiff_rent:,.0f} MXN/mes",
        delta=f"{indiff_rent - inputs.costs.monthly_lease_offer_mxn:+,.0f} vs {pick('Offer', 'Oferta', lang)}",
    )

    # Interactive Plotly charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            build_mc_histogram(sim_results, inputs.contract_months, lang),
            use_container_width=True,
        )
    with chart_col2:
        st.plotly_chart(build_monthly_comparison_chart(sim_results, lang), use_container_width=True)

    # Risk quantiles & month-by-month table
    st.markdown(
        "#### " + pick("Monthly Breakdown & Risk Quantiles", "Desglose Mensual y Escenarios de Riesgo", lang)
    )
    q1, q2, q3, q4 = st.columns(4)
    q1.metric(
        pick("Conservative STR (P10)", "Escenario Conservador (P10)", lang),
        f"${sim_results['str_p10_net_mxn']:,.0f} MXN",
    )
    q2.metric(
        pick("Probable STR (P50 Median)", "Escenario Probable (P50)", lang),
        f"${sim_results['str_p50_net_mxn']:,.0f} MXN",
    )
    q3.metric(
        pick("Optimistic STR (P90)", "Escenario Optimista (P90)", lang),
        f"${sim_results['str_p90_net_mxn']:,.0f} MXN",
    )
    q4.metric(
        pick("Tail Downside Risk (5% CVaR)", "Riesgo Extremo (5% CVaR)", lang),
        f"${sim_results['str_cvar_5pct_mxn']:,.0f} MXN",
    )
    st.dataframe(sim_results["monthly_df"], use_container_width=True, hide_index=True)

    if sim_results["cfe_trajectory"] is not None:
        st.markdown(
            "##### ⚡ "
            + pick(
                "12-Month Rolling Average CFE DAC Trajectory (Tariff 1B: 400 kWh/mo Limit)",
                "Trayectoria CFE Promedio Móvil 12 Meses (Tarifa 1B: Límite 400 kWh/mes)",
                lang,
            )
        )
        st.dataframe(pd.DataFrame(sim_results["cfe_trajectory"]), use_container_width=True, hide_index=True)


def render_cfe_assumptions_tab(artifacts: ForecastArtifacts, lang: str) -> None:
    """Tab 2: Official CFE 12-month rolling DAC rule explanation and explicit assumptions table."""
    st.subheader(
        pick(
            "Official CFE DAC 12-Month Rolling Rule & Explicit Model Assumptions",
            "Regla Oficial CFE DAC (Promedio Móvil 12 Meses) y Supuestos Explícitos",
            lang,
        )
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            pick(
                en="""
                #### 1. The Operational Problem in the Riviera Maya
                This project evaluates **4 anonymized portfolio units in the Riviera Maya** (modeled on a real rental business, with synthetic history for privacy):
                - **Playa del Carmen**: **Unit A Playa** and **Unit B Playa** (**1BR/1BA**) and **Unit C Playa** (**2BR/2BA**).
                - **Tulum**: **Unit Tulum** (**1BR/1BA**, anchored to Tulum's ~46% market-wide annual occupancy per **AirDNA 2026**).

                Every season, remote workers offer fixed **3-month or 6-month leases**. Comparing gross nightly Airbnb rates against a monthly lease is misleading because short-term stays carry variable costs that disappear on a mid-term contract.
                """,
                es="""
                #### 1. El Problema Operativo en la Riviera Maya
                Este proyecto evalúa **4 departamentos ilustrativos en la Riviera Maya** (basados en la estructura de un negocio real de rentas, anonimizados para este demo público):
                - **Playa del Carmen**: **Unidad A Playa** y **Unidad B Playa** (**1 Rec / 1 Baño**) y **Unidad C Playa** (**2 Rec / 2 Baños**).
                - **Tulum**: **Unidad Tulum** (**1 Rec / 1 Baño**, anclada al ~46% de ocupación media anual del mercado de Tulum según **AirDNA 2026**).

                Cada temporada, trabajadores remotos ofrecen contratos de **3 o 6 meses** a tarifa fija. Comparar solo la tarifa por noche de Airbnb contra una renta mensual lleva a malas decisiones si no se descuentan comisiones, limpiezas, vacancia y electricidad.
                """,
                lang=lang,
            )
        )
    with c2:
        st.markdown(
            pick(
                en="""
                #### 2. How CFE DAC Reclassification Actually Works (12-Month Rolling Average)
                - **12-Month Rolling Rule (CFE Tariff 1B = 400 kWh/mo)**: CFE does **not** trigger DAC from a single month in isolation. It computes the **trailing 12-month average** consumption (6 bimonthly bills). When that rolling mean exceeds **400 kWh/month** under Tariff 1B, the meter loses government subsidy and enters **DAC** (`~6.08 MXN/kWh + 16% IVA`).
                - **Winter Persistence**: Once reclassified into DAC, **winter months are also billed at the DAC rate** until the 12-month rolling average falls back to or below `400 kWh/month`.
                - **Lease Meter Carryover**: On a 3M/6M lease, the tenant pays CFE directly, saving the owner `$33,000–$39,000 MXN` in summer bills—though heavy tenant A/C kWh still counts toward the meter's 12-month rolling window.
                """,
                es="""
                #### 2. Cómo Funciona Realmente la Tarifa CFE DAC (Promedio Móvil 12 Meses)
                - **Regla de los 12 Meses (CFE Tarifa 1B = 400 kWh/mes)**: La CFE **no** reclasifica un medidor por un solo mes alto. Calcula el **promedio móvil de los últimos 12 meses** (6 bimestres). Si ese promedio supera los **400 kWh/mes** en Tarifa 1B, el medidor cae en **Tarifa DAC sin subsidio** (`~6.08 MXN/kWh + 16% IVA`).
                - **Persistencia en Invierno**: Una vez en DAC, **incluso los meses de invierno** se cobran a tarifa DAC hasta que el promedio de 12 meses vuelva a bajar de `400 kWh/mes`.
                - **Efecto del Inquilino en el Medidor**: En un contrato de 3 o 6 meses el inquilino paga su recibo CFE, pero su consumo en kWh queda registrado en el historial de 12 meses del medidor.
                """,
                lang=lang,
            )
        )

    st.markdown(
        "#### 3. "
        + pick(
            "Explicit Modeling Assumptions (`src/rental_engine/assumptions.py`)",
            "Tabla de Supuestos Explícitos (`src/rental_engine/assumptions.py`)",
            lang,
        )
    )
    n_oof = int(artifacts.meta.get("n_oof_rows", 0))
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
                "Parameter": "Forecast uncertainty (block bootstrap)",
                "Value": f"Centered OOF residuals ({n_oof:,} unit-month forecasts, horizons 1–12)",
                "Source / Rationale": "Resamples contiguous residual blocks to keep ADR/occupancy error correlation & persistence",
            },
            {
                "Parameter": "min_sampled_adr_mxn / occ clip bounds",
                "Value": f">= {DEFAULT_ASSUMPTIONS.min_sampled_adr_mxn:.0f} MXN | [{DEFAULT_ASSUMPTIONS.min_sampled_occ_rate * 100:.0f}%, {DEFAULT_ASSUMPTIONS.max_sampled_occ_rate * 100:.0f}%]",
                "Source / Rationale": "Prevents unrealistic negative prices or >100% physical occupancy",
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


def render_ml_tab(artifacts: Optional[ForecastArtifacts], lang: str) -> None:
    """Tab 3: Walk-forward pipeline architecture, backtest accuracy, and price-elasticity table."""
    st.subheader(
        pick(
            "Walk-Forward Forecasting Architecture, Backtest & Price Elasticity",
            "Arquitectura Walk-Forward, Backtest de Origen Móvil y Elasticidad de Precio",
            lang,
        )
    )
    st.markdown(
        """
        ```text
        [12-Unit × 48-Month Panel (`synthetic_panel.py`, or private reservations via `sql/schema_and_eda.sql`)]
                                                │
                                                ▼
        [Rolling-Origin Backtest (`forecasting.py`): 17 origins, horizons 1–12, strict `target <= origin`]
                                                │
                                                ▼
        [Compare 3 Baselines vs. Random Forest & Gradient Boosting → Select lowest transformed-scale RMSE]
                                                │
                                                ▼
        [Save Point Forecasts (`forecasts.csv`) + Centered OOF Residuals (`oof_residuals.csv`)]
                                                │
                                                ▼
        [5,000-Run Block-Bootstrap Monte Carlo (`engine.py`) → ACCEPT / NEGOTIATE / KEEP OPEN]
        ```
        """
    )

    if artifacts is None:
        st.info(
            pick(
                "Run `python scripts/run_forecast_pipeline.py` to generate backtest metrics and artifacts.",
                "Ejecuta `python scripts/run_forecast_pipeline.py` para generar métricas y artefactos.",
                lang,
            )
        )
        return

    wf = artifacts.meta
    rows = []
    for var, label, scale in (("adr", "ADR (MXN)", 1.0), ("occ", "Occupancy (pts)", 100.0)):
        for model_name, metrics in wf["metrics"][var]["all"].items():
            rows.append(
                {
                    "Target": label,
                    "Model": MODEL_LABELS.get(model_name, model_name),
                    "MAE": round(metrics["mae"] * scale, 2),
                    "RMSE": round(metrics["rmse"] * scale, 2),
                    "Skill vs best baseline (%)": round(metrics["skill_vs_best_baseline"] * 100.0, 1),
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    floor = wf.get("occupancy_mae_noise_floor")
    if floor is not None:
        st.caption(
            pick(
                f"{wf['n_units']} synthetic units × {wf['n_months']} months; rolling-origin with 1–12 month horizons "
                f"({wf['n_oof_rows']:,} OOF unit-month forecasts per model). Occupancy noise floor: MAE ≈ {floor * 100:.1f} pts "
                "(a perfect forecast of the latent probability). Positive skill = beats the best baseline.",
                f"{wf['n_units']} unidades sintéticas × {wf['n_months']} meses; origen móvil con horizontes 1–12 meses "
                f"({wf['n_oof_rows']:,} pronósticos OOF por modelo). Piso de ruido de ocupación: MAE ≈ {floor * 100:.1f} pts "
                "(un pronóstico perfecto de la probabilidad latente). Skill positivo = supera al mejor modelo base.",
                lang,
            )
        )

    st.markdown("#### " + pick("Price Elasticity of Occupancy", "Elasticidad de Precio de la Ocupación", lang))
    el = wf["elasticity"]
    el_df = pd.DataFrame(
        [
            {"Specification": "Naive pooled OLS", "beta": f"{el['naive_pooled_ols']}"},
            {"Specification": "Unit + calendar-month + trend controls", "beta": f"{el['unit_month_trend_controls']}"},
            {"Specification": "Unit + year-month fixed effects", "beta": f"{el['twfe_unit_and_year_month']}"},
            {
                "Specification": "Unit x calendar-month + year-month fixed effects (headline)",
                "beta": f"{el['headline_estimate']} (95% CI {el['headline_ci95_bootstrap'][0]} to {el['headline_ci95_bootstrap'][1]})",
            },
            {"Specification": "True value in the generator", "beta": f"{el['true_value']}"},
        ]
    )
    st.dataframe(el_df, use_container_width=True, hide_index=True)
    st.caption(
        pick(
            "The naive regression has the wrong sign (high season raises both prices and occupancy); "
            "unit × calendar-month + year-month fixed effects recover the true price response.",
            "La regresión ingenua tiene el signo equivocado (temporada alta eleva precio y ocupación); "
            "los efectos fijos de unidad × mes + año-mes recuperan la respuesta verdadera.",
            lang,
        )
    )


def render_fb_parser_tab(lang: str) -> None:
    """Tab 4: Safe copy-paste Facebook Group rental listing parser."""
    st.subheader(
        pick(
            "Safe Copy-Paste Facebook Group Rental Post Parser (0% Account Ban Risk)",
            "Lector Seguro de Publicaciones en Grupos de Facebook (0% Riesgo de Bloqueo)",
            lang,
        )
    )
    st.caption(
        pick(
            "Instead of running automated scrapers in closed groups, paste any raw listing text "
            "from a Playa del Carmen or Tulum Facebook group below:",
            "En lugar de usar bots en grupos privados, pega aquí cualquier texto de una publicación "
            "de renta en Playa del Carmen o Tulum:",
            lang,
        )
    )
    default_fb_text = pick(
        en="Available May 1st! Beautiful 1BR apartment in Aldea Zama, Tulum. Rooftop pool & fiber WiFi. $24,500 MXN/mes + CFE electric. Minimum 3 or 6 months.",
        es="Disponible 1 de mayo! Hermoso depto 1 recámara en Aldea Zama, Tulum. Alberca en rooftop y WiFi fibra. $24,500 MXN/mes + CFE luz aparte. Contrato 3 o 6 meses.",
        lang=lang,
    )
    sample_post = st.text_area(
        pick("Paste raw Facebook Group post text:", "Pega el texto de la publicación de Facebook:", lang),
        value=default_fb_text,
        height=100,
    )
    parsed = parse_facebook_rental_post(sample_post)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(pick("Extracted Bedrooms", "Recámaras", lang), f"{parsed['bedrooms']} BR")
    p2.metric(pick("Detected Zone", "Zona Detectada", lang), str(parsed["neighborhood"]))
    p3.metric(
        pick("Asking Monthly Rent", "Renta Mensual", lang),
        f"${parsed['asking_monthly_mxn']:,.0f} MXN" if parsed["asking_monthly_mxn"] else "N/A",
    )
    p4.metric(
        pick("Tenant Pays CFE?", "¿Inquilino Paga CFE?", lang),
        pick("Yes (+ CFE)", "Sí (+ CFE)", lang)
        if parsed["tenant_pays_cfe"]
        else pick("No / Unspecified", "No especificado", lang),
    )
