"""
Bilingual (English / Spanish) text helpers and static copy blocks for the Streamlit UI.
"""
from __future__ import annotations

from typing import List


def pick(en: str, es: str, lang: str) -> str:
    """Return `es` when `lang == 'es'`, otherwise `en`."""
    return es if lang == "es" else en


HEADER_CSS = """
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
"""


def header_html(lang: str) -> tuple[str, str]:
    """Return `(title_html, caption_html)` for the page banner."""
    if lang == "es":
        title = "<h1 class='header-title'>🏖️ Optimizador de Estrategia de Renta</h1>"
        caption = (
            "<div class='header-caption'><strong>Creado por Erik Šauša, Dipl.-Ing. (M.Sc.), B.Sc.</strong><br>"
            "<em>Maestría en Ciencias en Telecomunicaciones · Licenciatura en Ingeniería Eléctrica y Tecnologías de la Información (TU Wien)</em><br>"
            "<strong>🚧 Motor Interactivo de Decisiones Inmobiliarias (Playa del Carmen, Tulum y CDMX)</strong></div>"
        )
        return title, caption

    title = "<h1 class='header-title'>🏖️ Rental Strategy Optimizer</h1>"
    caption = (
        "<div class='header-caption'><strong>Created by Erik Šauša, Dipl.-Ing. (M.Sc.), B.Sc.</strong><br>"
        "<em>Master of Science in Telecommunications · Bachelor of Science in Electrical Engineering and Information Technology (TU Wien)</em><br>"
        "<strong>🚧 Interactive Decision Engine Demo (Playa del Carmen, Tulum & CDMX)</strong></div>"
    )
    return title, caption


def demo_banners(lang: str) -> tuple[str, str]:
    """Return `(synthetic_warning, project_overview_info)` banner strings."""
    warning = pick(
        en=(
            "⚠️ **Public Demo (Illustrative Synthetic Inputs):** "
            "To protect business and guest privacy, this repository uses a **synthetic 12-unit × 48-month panel** "
            "(`src/rental_engine/synthetic_panel.py`) covering 4 anonymized units (**Unit A Playa**, **Unit B Playa**, "
            "**Unit C Playa**, and **Unit Tulum**) plus a CDMX benchmark. "
            "Market-level reference figures (such as Tulum's ~46% and Playa del Carmen's ~58% annual mean occupancy) "
            "are cited from public **AirDNA (2026)** benchmarks."
        ),
        es=(
            "⚠️ **Demo Público (Datos Sintéticos Ilustrativos):** "
            "Para proteger la privacidad del negocio y de los huéspedes, este repositorio utiliza un **panel sintético de 12 unidades × 48 meses** "
            "(`src/rental_engine/synthetic_panel.py`) con 4 unidades ilustrativas (**Unidad A Playa**, **Unidad B Playa**, "
            "**Unidad C Playa** y **Unidad Tulum**) más un benchmark en CDMX. "
            "Las cifras de mercado (~46% de ocupación anual en Tulum y ~58% en Playa del Carmen) provienen de benchmarks públicos de **AirDNA (2026)**."
        ),
        lang=lang,
    )
    overview = pick(
        en=(
            "🎯 **About This Project:** "
            "End-to-end decision engine (**Walk-Forward ML Backtest $\\rightarrow$ Block-Bootstrap Monte Carlo $\\rightarrow$ Streamlit**) "
            "built to answer a real operational question: *when a prospective tenant offers a guaranteed 3-month or 6-month lease, "
            "should we accept the contract today, negotiate a higher rate, or keep the calendar open for short-term stays across Airbnb and Facebook Groups?* "
            "Models CFE's **12-month rolling average DAC electricity reclassification rule** (Tariff 1B: `400 kWh/month` rolling threshold), "
            "blends **0%-fee Facebook Group direct bookings** with **Airbnb (~18% fee)**, and uses exact calendar days per month."
        ),
        es=(
            "🎯 **Sobre Este Proyecto:** "
            "Motor de decisiones de extremo a extremo (**Backtest Walk-Forward ML $\\rightarrow$ Monte Carlo con Bootstrap de Bloques $\\rightarrow$ Streamlit**) "
            "diseñado para responder una pregunta operativa real: *cuando un inquilino ofrece un contrato fijo de 3 o 6 meses, ¿conviene aceptar la oferta hoy, "
            "negociar una renta más alta o dejar el calendario abierto para rentas cortas por Airbnb y grupos de Facebook?* "
            "Modela la regla real de reclasificación **CFE DAC por promedio móvil de 12 meses** (Tarifa 1B: límite de `400 kWh/mes`), "
            "la mezcla de reservas directas en **Grupos de Facebook (0% comisión)** vs. **Airbnb (~18%)**, y el número exacto de días calendario de cada mes."
        ),
        lang=lang,
    )
    return warning, overview


def sidebar_author_markdown(lang: str) -> str:
    """Return bilingual author bio Markdown for the sidebar header."""
    if lang == "es":
        return (
            "**Erik Šauša, Dipl.-Ing. (M.Sc.), B.Sc.**\n"
            "- 🎓 **Maestría en Ciencias en Telecomunicaciones** *(TU Wien)*\n"
            "- 🎓 **Licenciatura en Ingeniería Eléctrica y Tecnología de la Información** *(TU Wien)*\n"
            "- 💼 **Científico de Datos & Ingeniero de ML**"
        )
    return (
        "**Erik Šauša, Dipl.-Ing. (M.Sc.), B.Sc.**\n"
        "- 🎓 **Master of Science in Telecommunications** *(TU Wien)*\n"
        "- 🎓 **Bachelor of Science in Electrical Engineering and Information Technology** *(TU Wien)*\n"
        "- 💼 **Data Scientist & ML Engineer**"
    )


def tab_titles(lang: str) -> List[str]:
    """Return localized titles for the four main dashboard tabs."""
    if lang == "es":
        return [
            "📊 1. Simulador de Decisiones en Vivo",
            "💡 2. Regla CFE DAC (Promedio 12M) y Supuestos",
            "🤖 3. Backtest Walk-Forward y Elasticidad de Precio",
            "💬 4. Lector de Publicaciones de Facebook",
        ]
    return [
        "📊 1. Live Decision Simulator",
        "💡 2. CFE DAC 12M Rolling Rule & Assumptions",
        "🤖 3. Walk-Forward Backtest & Price Elasticity",
        "💬 4. Facebook Group Post Parser",
    ]
