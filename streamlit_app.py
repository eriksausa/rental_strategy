"""
Rental Strategy Optimizer (Playa del Carmen, Tulum & CDMX)
Bilingual (English / Español) Interactive Decision Engine.

Thin Streamlit entry point: loads precomputed walk-forward forecasting artifacts
from `artifacts/` and `reports/`, collects user inputs via `src.ui.sidebar`,
runs the Monte Carlo engine in `src.rental_engine.engine`, and renders the four
dashboard tabs in `src.ui.tabs`.

Run locally with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from src.rental_engine import forecast_service
from src.rental_engine.engine import simulate_lease_vs_str
from src.ui.i18n import HEADER_CSS, demo_banners, header_html, pick, tab_titles
from src.ui.presets import SCENARIO_PRESETS
from src.ui.sidebar import render_sidebar
from src.ui.tabs import (
    render_cfe_assumptions_tab,
    render_fb_parser_tab,
    render_ml_tab,
    render_simulator_tab,
)

ROOT_DIR = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def get_forecast_artifacts() -> Optional[forecast_service.ForecastArtifacts]:
    """Load walk-forward forecasts and OOF residual blocks once per Streamlit session."""
    try:
        return forecast_service.load_artifacts(ROOT_DIR)
    except (FileNotFoundError, KeyError, ValueError):
        return None


def main() -> None:
    st.set_page_config(
        page_title="Rental Strategy Optimizer — Riviera Maya & CDMX",
        page_icon="🏖️",
        layout="wide",
    )

    # 1. Language Switcher & Page Header
    top_col1, top_col2 = st.columns([4, 1])
    with top_col2:
        lang_choice = st.radio("🌐 Language / Idioma", options=["English", "Español"], horizontal=True)
    lang = "es" if lang_choice == "Español" else "en"

    st.markdown(HEADER_CSS, unsafe_allow_html=True)
    title_html, caption_html = header_html(lang)
    with top_col1:
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

    warning_msg, overview_msg = demo_banners(lang)
    st.warning(warning_msg)
    st.info(overview_msg)

    # 2. Load Walk-Forward Forecasting Artifacts
    artifacts = get_forecast_artifacts()
    if artifacts is None:
        st.error(
            pick(
                "Forecast artifacts not found. Run `python scripts/run_forecast_pipeline.py` first.",
                "No se encontraron artefactos de pronóstico. Ejecuta `python scripts/run_forecast_pipeline.py` primero.",
                lang,
            )
        )
        return

    # 3. Scenario Preset & Sidebar Inputs
    preset_labels = [p.label(lang) for p in SCENARIO_PRESETS]
    selected_preset_label = st.selectbox(
        pick(
            "🎬 Load a Demo Scenario Preset (or customize any cost input in the left sidebar):",
            "🎬 Selecciona un Escenario Demo Preconfigurado (o personaliza los costos en el panel izquierdo):",
            lang,
        ),
        preset_labels,
    )
    preset = SCENARIO_PRESETS[preset_labels.index(selected_preset_label)]
    inputs = render_sidebar(preset, lang)

    # 4. Resolve Walk-Forward Window & Run 5,000-Run Monte Carlo Simulation
    forecasts, residual_blocks, wrapped_year_end = forecast_service.forecast_calendar_window(
        artifacts, inputs.unit_id, inputs.months_window, lang=lang
    )
    sim_results = simulate_lease_vs_str(
        forecasts,
        inputs.costs,
        lang=lang,
        prior_11m_kwh=inputs.prior_11m_kwh,
        window_monthly_kwh=inputs.window_monthly_kwh,
        residual_blocks=residual_blocks if inputs.use_block_bootstrap else None,
    )

    # 5. Render Presentation Tabs
    tab_sim, tab_cfe, tab_ml, tab_fb = st.tabs(tab_titles(lang))
    with tab_sim:
        render_simulator_tab(inputs, sim_results, artifacts, wrapped_year_end, lang)
    with tab_cfe:
        render_cfe_assumptions_tab(artifacts, lang)
    with tab_ml:
        render_ml_tab(artifacts, lang)
    with tab_fb:
        render_fb_parser_tab(lang)


if __name__ == "__main__":
    main()
