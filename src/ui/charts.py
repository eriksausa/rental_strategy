"""
Plotly chart builders for the Live Decision Simulator tab.
"""
from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go

from .i18n import pick


def build_mc_histogram(sim_results: Dict[str, object], contract_months: int, lang: str) -> go.Figure:
    """Build the Monte Carlo total net cash-flow histogram with Lease and P10 reference lines."""
    lease_net = float(sim_results["lease_total_net_mxn"])
    p10_net = float(sim_results["str_p10_net_mxn"])

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sim_results["sim_str_totals"],
            nbinsx=45,
            name=pick("STR Net Cash Flow (5,000 Runs)", "Renta Corta (5,000 corridas)", lang),
            marker_color="#3b82f6",
            opacity=0.75,
        )
    )
    fig.add_vline(
        x=lease_net,
        line_width=3,
        line_dash="dash",
        line_color="#16a34a",
        annotation_text=pick(
            f"Guaranteed Lease Net (${lease_net:,.0f})",
            f"Contrato Garantizado (${lease_net:,.0f})",
            lang,
        ),
        annotation_position="top right",
    )
    fig.add_vline(
        x=p10_net,
        line_width=2,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text=f"P10 (${p10_net:,.0f})",
        annotation_position="bottom left",
    )
    fig.update_layout(
        title=pick(
            f"Monte Carlo Net Income Distribution ({contract_months}-Month Window)",
            f"Simulación Monte Carlo: Distribución de Ingreso Neto ({contract_months} Meses)",
            lang,
        ),
        xaxis_title=pick(
            "Total Net Cash Flow to Owner (MXN)",
            "Ingreso Neto Total para el Propietario (MXN)",
            lang,
        ),
        yaxis_title=pick(
            "Simulation Frequency (out of 5,000)",
            "Frecuencia (de 5,000 corridas)",
            lang,
        ),
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def build_monthly_comparison_chart(sim_results: Dict[str, object], lang: str) -> go.Figure:
    """Build a grouped bar chart comparing guaranteed lease net vs. expected STR net per month."""
    mdf = sim_results["monthly_df"]
    m_col = pick("Month", "Mes", lang)
    lease_col = pick("Guaranteed Lease Net (MXN)", "Neto Contrato Garantizado (MXN)", lang)
    str_col = pick("Expected STR Net (MXN)", "Neto Renta Corta Esperado (MXN)", lang)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=mdf[m_col],
            y=mdf[lease_col],
            name=pick(
                "Guaranteed Lease Net (Tenant Pays CFE)",
                "Contrato Garantizado (Inquilino paga CFE)",
                lang,
            ),
            marker_color="#16a34a",
        )
    )
    fig.add_trace(
        go.Bar(
            x=mdf[m_col],
            y=mdf[str_col],
            name=pick(
                "Expected STR Net (After CFE, Cleaning & Fees)",
                "Renta Corta Esperada (Tras CFE, Limpieza y Comisiones)",
                lang,
            ),
            marker_color="#3b82f6",
        )
    )
    fig.update_layout(
        title=pick(
            "Month-by-Month Expected Net Income (Exact Calendar Days)",
            "Comparación Mensual de Flujo Neto (Días Exactos del Mes)",
            lang,
        ),
        xaxis_title=pick("Contract Month", "Mes del Contrato", lang),
        yaxis_title=pick("Monthly Net Income (MXN)", "Ingreso Neto Mensual (MXN)", lang),
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig
