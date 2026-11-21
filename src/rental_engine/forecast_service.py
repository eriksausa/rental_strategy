"""
Bridge between the saved forecasting artifacts and the decision engine.

    artifacts/forecasts.csv       point forecasts (h = 1..12 months past the last observed month)
    artifacts/oof_residuals.csv   out-of-fold backtest residuals of the chosen models
    reports/forecast_metrics.json backtest metrics and metadata

Usage
-----
    art = load_artifacts(Path("."))
    monthly, blocks = forecast_window(art, "CB_TUL_1BR", start_month="2026-05", n_months=6)
    result = simulate_lease_vs_str(monthly, costs, residual_blocks=blocks)

`blocks` has shape (n_blocks, n_months, 2). Each block is a window of consecutive horizons
taken from one unit at one backtest origin, holding the (log-ADR, log-odds-occupancy)
forecast errors. Resampling whole blocks keeps the persistence across months and the
dependence between ADR and occupancy that independent normal draws would lose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MAX_HORIZON = 12


@dataclass
class ForecastArtifacts:
    forecasts: pd.DataFrame
    residuals: pd.DataFrame
    meta: Dict[str, object]


def load_artifacts(root: Path) -> ForecastArtifacts:
    root = Path(root)
    forecasts = pd.read_csv(root / "artifacts" / "forecasts.csv")
    residuals = pd.read_csv(root / "artifacts" / "oof_residuals.csv")
    meta = json.loads((root / "reports" / "forecast_metrics.json").read_text(encoding="utf-8"))
    return ForecastArtifacts(forecasts=forecasts, residuals=residuals, meta=meta)


def months_between(start: str, end: str) -> int:
    a, b = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    return int((b - a).n)


def window_start_month(last_observed: str, months_window: List[int]) -> Optional[str]:
    """Map a window of calendar months (e.g. [5, 6, 7, 8, 9, 10]) to the first matching
    'YYYY-MM' within 1..12 months after `last_observed`, or None if it does not fit."""
    last = pd.Period(last_observed, freq="M")
    for h in range(1, MAX_HORIZON + 1):
        if (last + h).month == int(months_window[0]) and h + len(months_window) - 1 <= MAX_HORIZON:
            return str(last + h)
    return None


def residual_blocks(residuals: pd.DataFrame, first_h: int, n_months: int) -> Optional[np.ndarray]:
    """Windows of residuals for horizons first_h .. first_h + n_months - 1, shape (B, n_months, 2)."""
    hs = list(range(first_h, first_h + n_months))
    piv_a = residuals.pivot_table(index=["unit_id", "origin"], columns="h", values="res_adr")
    piv_o = residuals.pivot_table(index=["unit_id", "origin"], columns="h", values="res_occ")
    if not set(hs).issubset(piv_a.columns):
        return None
    a = piv_a[hs].dropna()
    o = piv_o[hs].loc[a.index].dropna()
    a = a.loc[o.index]
    if len(a) == 0:
        return None
    return np.stack([a.to_numpy(), o.to_numpy()], axis=2)


def forecast_window(
    art: ForecastArtifacts, unit_id: str, start_month: str, n_months: int
) -> Tuple[List[Dict[str, float]], Optional[np.ndarray]]:
    """Monthly forecast dicts for a lease window plus matching residual blocks.

    `start_month` is 'YYYY-MM'. The window must fall within 1..12 months after the last
    observed month of the data the models were trained on.
    """
    last = str(art.meta["last_observed_month"])
    h0 = months_between(last, start_month)
    if h0 < 1 or h0 + n_months - 1 > MAX_HORIZON:
        raise ValueError(
            f"Window {start_month} (+{n_months} months) is outside the forecast range: "
            f"forecasts cover months 1..{MAX_HORIZON} after {last}."
        )
    rows = art.forecasts[
        (art.forecasts["unit_id"] == unit_id) & (art.forecasts["h"] >= h0) & (art.forecasts["h"] < h0 + n_months)
    ].sort_values("h")
    if len(rows) != n_months:
        raise ValueError(f"No forecasts found for unit {unit_id!r} in that window.")

    sd_by_h = art.residuals.groupby("h")[["res_adr", "res_occ"]].std()
    monthly: List[Dict[str, float]] = []
    for _, r in rows.iterrows():
        sd_adr = float(sd_by_h.loc[int(r["h"]), "res_adr"])
        sd_occ = float(sd_by_h.loc[int(r["h"]), "res_occ"])
        p = float(r["pred_occ_rate"])
        monthly.append(
            {
                "month_num": int(r["month_num"]),
                "year_month": str(r["year_month"]),
                "h": int(r["h"]),
                "pred_adr_mxn": float(r["pred_adr_mxn"]),
                "pred_occ_rate": p,
                # Marginal spreads, used only when no residual blocks are supplied.
                "adr_std_mxn": float(r["pred_adr_mxn"]) * sd_adr,
                "occ_std": p * (1.0 - p) * sd_occ,
            }
        )
    return monthly, residual_blocks(art.residuals, h0, n_months)
