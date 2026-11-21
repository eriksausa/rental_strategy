"""
Bridge between saved walk-forward forecasting artifacts and the decision engine.

Expected artifact layout (produced by `scripts/run_forecast_pipeline.py`):
    artifacts/forecasts.csv         Point forecasts for horizons h = 1..12 after `last_observed_month`
    artifacts/oof_residuals.csv     Centered OOF transformed-scale residuals of the chosen models
    reports/forecast_metrics.json   Backtest metrics, chosen models, elasticity, and metadata

Usage
-----
    art = load_artifacts(Path("."))
    monthly, blocks = forecast_window(art, "UNIT_TULUM", start_month="2026-05", n_months=6)
    result = simulate_lease_vs_str(monthly, costs, residual_blocks=blocks)

`blocks` has shape `(n_blocks, n_months, 2)`. Each block is a slice of consecutive horizons
taken from one unit at one backtest origin, holding `(log-ADR, log-odds-occupancy)` errors.
Resampling whole blocks preserves both month-to-month persistence and ADR/occupancy error
correlation that independent normal draws discard.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .assumptions import MONTH_NAMES_EN, MONTH_NAMES_ES

MAX_HORIZON = 12


@dataclass
class ForecastArtifacts:
    """In-memory bundle of walk-forward forecasts, OOF residuals, and run metadata."""

    forecasts: pd.DataFrame
    residuals: pd.DataFrame
    meta: Dict[str, object]


def load_artifacts(root: Path) -> ForecastArtifacts:
    """Load `forecasts.csv`, `oof_residuals.csv`, and `forecast_metrics.json` under `root`."""
    root = Path(root)
    forecasts = pd.read_csv(root / "artifacts" / "forecasts.csv")
    residuals = pd.read_csv(root / "artifacts" / "oof_residuals.csv")
    meta = json.loads((root / "reports" / "forecast_metrics.json").read_text(encoding="utf-8"))
    return ForecastArtifacts(forecasts=forecasts, residuals=residuals, meta=meta)


def months_between(start: str, end: str) -> int:
    """Return the signed integer number of calendar months from `start` to `end` (`YYYY-MM`)."""
    a, b = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    return int((b - a).n)


def window_start_month(last_observed: str, months_window: Sequence[int]) -> Optional[str]:
    """Map a calendar-month sequence (e.g. `[5, 6, 7, 8, 9, 10]`) to the first matching `YYYY-MM`
    that fits completely inside horizons `1..12` after `last_observed`, or `None` if it overflows."""
    last = pd.Period(last_observed, freq="M")
    for h in range(1, MAX_HORIZON + 1):
        if (last + h).month == int(months_window[0]) and h + len(months_window) - 1 <= MAX_HORIZON:
            return str(last + h)
    return None


def residual_blocks(residuals: pd.DataFrame, first_h: int, n_months: int) -> Optional[np.ndarray]:
    """Assemble consecutive-horizon OOF residual blocks of shape `(n_blocks, n_months, 2)`.

    Channel 0 is the log-ADR residual; channel 1 is the logit-occupancy residual.
    Only `(unit_id, origin)` pairs with complete observations across all requested horizons
    are retained so every resampled trajectory is contiguous in time.
    """
    horizons = list(range(first_h, first_h + n_months))
    piv_adr = residuals.pivot_table(index=["unit_id", "origin"], columns="h", values="res_adr")
    piv_occ = residuals.pivot_table(index=["unit_id", "origin"], columns="h", values="res_occ")
    if not set(horizons).issubset(piv_adr.columns):
        return None

    adr_blocks = piv_adr[horizons].dropna()
    occ_blocks = piv_occ[horizons].loc[adr_blocks.index].dropna()
    adr_blocks = adr_blocks.loc[occ_blocks.index]
    if len(adr_blocks) == 0:
        return None
    return np.stack([adr_blocks.to_numpy(), occ_blocks.to_numpy()], axis=2)


def _row_to_monthly_forecast(
    row: pd.Series,
    sd_by_h: pd.DataFrame,
    month_map: Dict[int, str],
) -> Dict[str, object]:
    """Convert one forecast CSV row into the dict format expected by `simulate_lease_vs_str`."""
    h_val = int(row["h"])
    m_num = int(row["month_num"])
    pred_adr = float(row["pred_adr_mxn"])
    pred_occ = float(row["pred_occ_rate"])
    # Clamp horizon lookup to available OOF residual horizons (e.g. during `--quick` smoke runs).
    h_lookup = h_val if h_val in sd_by_h.index else int(sd_by_h.index.max())
    sd_log_adr = float(sd_by_h.loc[h_lookup, "res_adr"])
    sd_logit_occ = float(sd_by_h.loc[h_lookup, "res_occ"])

    # Delta-method marginal standard deviations on level scales (used only when
    # running the independent-normal comparison without residual blocks):
    #   d(exp(x))/dx = exp(x) -> sd(ADR) ≈ ADR * sd(log_ADR)
    #   d(sigmoid(z))/dz = p(1 - p) -> sd(occ) ≈ p * (1 - p) * sd(logit_occ)
    return {
        "month_num": m_num,
        "month_name": month_map[m_num],
        "year_month": str(row["year_month"]),
        "h": h_val,
        "pred_adr_mxn": pred_adr,
        "pred_occ_rate": pred_occ,
        "adr_std_mxn": pred_adr * sd_log_adr,
        "occ_std": pred_occ * (1.0 - pred_occ) * sd_logit_occ,
    }


def forecast_window(
    art: ForecastArtifacts,
    unit_id: str,
    start_month: str,
    n_months: int,
    lang: str = "en",
) -> Tuple[List[Dict[str, object]], Optional[np.ndarray]]:
    """Build monthly forecast dicts and horizon-aligned residual blocks for an exact `YYYY-MM` window.

    Raises:
        ValueError: If the requested window falls outside horizons `1..12` after
            `art.meta["last_observed_month"]` or if `unit_id` is not present in `art.forecasts`.
    """
    last = str(art.meta["last_observed_month"])
    first_h = months_between(last, start_month)
    if first_h < 1 or first_h + n_months - 1 > MAX_HORIZON:
        raise ValueError(
            f"Window {start_month} (+{n_months} months) is outside the forecast range: "
            f"forecasts cover months 1..{MAX_HORIZON} after {last}."
        )

    rows = art.forecasts[
        (art.forecasts["unit_id"] == unit_id)
        & (art.forecasts["h"] >= first_h)
        & (art.forecasts["h"] < first_h + n_months)
    ].sort_values("h")
    if len(rows) != n_months:
        raise ValueError(f"No forecasts found for unit {unit_id!r} in that window.")

    month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN
    sd_by_h = art.residuals.groupby("h")[["res_adr", "res_occ"]].std()
    monthly = [_row_to_monthly_forecast(r, sd_by_h, month_map) for _, r in rows.iterrows()]
    return monthly, residual_blocks(art.residuals, first_h, n_months)


def forecast_calendar_window(
    art: ForecastArtifacts,
    unit_id: str,
    months_window: Sequence[int],
    lang: str = "en",
) -> Tuple[List[Dict[str, object]], Optional[np.ndarray], bool]:
    """Resolve any 3M or 6M calendar-month window (including year-end wraps like Nov-Apr).

    When the window fits inside horizons `1..12` after `last_observed_month`, delegates directly
    to `forecast_window()` (`wrapped_year_end=False`). When a lease crosses past December beyond
    horizon 12, maps each calendar month to that unit's 12-month walk-forward forecast and pairs
    it with `h = 1..n_months` OOF residual blocks (`wrapped_year_end=True`).
    """
    last = str(art.meta["last_observed_month"])
    start_ym = window_start_month(last, months_window)
    if start_ym is not None:
        monthly, blocks = forecast_window(art, unit_id, start_ym, len(months_window), lang=lang)
        return monthly, blocks, False

    unit_rows = art.forecasts[art.forecasts["unit_id"] == unit_id]
    if unit_rows.empty:
        raise ValueError(f"Unknown unit_id {unit_id!r} in forecast artifacts.")

    month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN
    sd_by_h = art.residuals.groupby("h")[["res_adr", "res_occ"]].std()
    by_month = {int(r["month_num"]): r for _, r in unit_rows.iterrows()}

    monthly = [_row_to_monthly_forecast(by_month[int(m)], sd_by_h, month_map) for m in months_window]
    blocks = residual_blocks(art.residuals, first_h=1, n_months=len(months_window))
    return monthly, blocks, True
