"""
Forecasting ADR and occupancy for rental units, with honest walk-forward evaluation.

Design
------
* Targets are modelled on transformed scales: log(ADR) and logit(occupancy).
* One multi-horizon "direct" model per target. Each training row is an
  (origin, horizon h) pair: "standing at month `origin`, predict month `origin + h`".
  Features only use data observed at or before `origin`:
      lag12      value of the target month one year earlier
      um         the unit's average for the target calendar month over all past years
      recent     mean of the last 3 observed months
      yoy        mean year-over-year change over the last 3 observed months
      dev        mean deviation of the last 3 months from the unit's own seasonal norm
      month sin/cos of the TARGET month, horizon h, unit attributes
  ML models learn the deviation from the unit's seasonal norm (`um`), so they only have
  to learn what the seasonal average does not already explain.
* Baselines that any ML model must beat:
      seasonal_naive         same month last year
      seasonal_naive_drift   same month last year + recent year-over-year change
      unit_month_mean        unit's average for that calendar month so far
* Evaluation is a rolling-origin (walk-forward) backtest. At each origin the models
  are trained ONLY on pairs whose target month is already observed at that origin.
* Out-of-fold (OOF) residuals of the chosen model are saved. The decision engine
  resamples WINDOWS of these residuals (block bootstrap), so uncertainty keeps the
  month-to-month persistence and the ADR/occupancy dependence seen in the backtest.

Everything here runs on whatever panel it is given. With the synthetic demo panel the
reported numbers describe the generator (see src/synthetic_data.py), not the real market.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

MAX_H = 12
MIN_ORIGIN = 14           # need 12 months of history plus a 3-month window for YoY features
BASELINES = ("seasonal_naive", "seasonal_naive_drift", "unit_month_mean")
ML_MODELS = ("random_forest", "gradient_boosting")
ALL_MODELS = BASELINES + ML_MODELS
CITIES = ("Playa del Carmen", "Tulum", "CDMX")
FEATURES = [
    "month_sin", "month_cos", "h", "bedrooms", "city_pdc", "city_tul", "city_cdmx",
    "lag_adr", "lag_occ", "rec_adr", "rec_occ", "yoy_adr", "yoy_occ",
    "um_adr", "um_occ", "dev_adr", "dev_occ",
]
H_BUCKETS = {"h1-3": (1, 3), "h4-6": (4, 6), "h7-12": (7, 12)}


# --------------------------------------------------------------------------- transforms
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def smooth_occ(occ, days):
    """Keeps 0% / 100% months finite on the log-odds scale."""
    return (occ * days + 0.5) / (days + 1.0)


def _prepare(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = panel.sort_values(["unit_id", "year_month"]).copy()
    months = sorted(df["year_month"].unique())
    units = df["unit_id"].unique()
    if len(df) != len(units) * len(months):
        raise ValueError("forecasting requires a balanced panel (same months for every unit).")
    df["y_adr"] = np.log(df["adr_mxn"].astype(float))
    df["y_occ"] = logit(smooth_occ(df["occupancy_rate"].astype(float), df["days_in_month"].astype(float)))
    return df, months


# --------------------------------------------------------------------------- pairs / features
def build_pairs(
    panel: pd.DataFrame,
    max_h: int = MAX_H,
    min_origin: int = MIN_ORIGIN,
    include_future: bool = False,
) -> pd.DataFrame:
    """One row per (unit, origin, horizon). Features use data at or before `origin` only.

    With include_future=True, rows for the LAST observed origin are also created for
    targets beyond the data (labels are NaN); these are used to produce final forecasts.
    """
    df, months = _prepare(panel)
    T = len(months)
    last_period = pd.Period(months[-1], freq="M")
    month_nums = np.array([int(m[5:7]) for m in months])

    def month_of(idx: int) -> int:
        return int(month_nums[idx]) if idx < T else int((last_period + (idx - (T - 1))).month)

    has_ptrue = "p_true" in df.columns
    rows = []
    for uid, g in df.groupby("unit_id", sort=True):
        g = g.set_index("year_month").loc[months]
        ya, yo = g["y_adr"].to_numpy(), g["y_occ"].to_numpy()
        adr_level, occ_level = g["adr_mxn"].to_numpy(), g["occupancy_rate"].to_numpy()
        ptrue = g["p_true"].to_numpy() if has_ptrue else None
        bedrooms, city = int(g["bedrooms"].iloc[0]), g["city"].iloc[0]
        for O in range(min_origin, T):
            rec_a, rec_o = ya[O - 2:O + 1].mean(), yo[O - 2:O + 1].mean()
            yoy_a = (ya[O - 2:O + 1] - ya[O - 14:O - 11]).mean()
            yoy_o = (yo[O - 2:O + 1] - yo[O - 14:O - 11]).mean()
            dev_a, dev_o = [], []
            for j in range(O - 2, O + 1):
                idx_j = [k for k in range(O + 1) if month_nums[k] == month_nums[j]]
                dev_a.append(ya[j] - ya[idx_j].mean())
                dev_o.append(yo[j] - yo[idx_j].mean())
            dev_a, dev_o = float(np.mean(dev_a)), float(np.mean(dev_o))
            for h in range(1, max_h + 1):
                tgt = O + h
                if tgt > T - 1 and not (include_future and O == T - 1):
                    continue
                m = month_of(tgt)
                same_month_idx = [j for j in range(O + 1) if month_nums[j] == m]
                observed = tgt <= T - 1
                rows.append(
                    {
                        "unit_id": uid, "origin": O, "target": tgt, "h": h,
                        "target_month": str(last_period + (tgt - (T - 1))),
                        "month_num": m,
                        "month_sin": np.sin(2 * np.pi * m / 12.0), "month_cos": np.cos(2 * np.pi * m / 12.0),
                        "bedrooms": bedrooms,
                        "city_pdc": int(city == CITIES[0]), "city_tul": int(city == CITIES[1]),
                        "city_cdmx": int(city == CITIES[2]),
                        "lag_adr": ya[tgt - 12], "lag_occ": yo[tgt - 12],
                        "rec_adr": rec_a, "rec_occ": rec_o, "yoy_adr": yoy_a, "yoy_occ": yoy_o,
                        "um_adr": ya[same_month_idx].mean(), "um_occ": yo[same_month_idx].mean(),
                        "dev_adr": dev_a, "dev_occ": dev_o,
                        "y_adr": ya[tgt] if observed else np.nan,
                        "y_occ": yo[tgt] if observed else np.nan,
                        "actual_adr_mxn": adr_level[tgt] if observed else np.nan,
                        "actual_occ": occ_level[tgt] if observed else np.nan,
                        "p_true": (ptrue[tgt] if (observed and ptrue is not None) else np.nan),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- models
def _make_model(name: str, seed: int, n_estimators: int):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=3, max_features=0.6, random_state=seed, n_jobs=1
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed
        )
    raise ValueError(f"unknown model {name}")


def fit_predict(
    name: str, train: pd.DataFrame, test: pd.DataFrame, var: str, seed: int = 0, n_estimators: int = 150
) -> np.ndarray:
    """Predict the TRANSFORMED target (log ADR or log-odds occupancy) for `test` rows."""
    lag = test[f"lag_{var}"].to_numpy()
    if name == "seasonal_naive":
        return lag
    if name == "seasonal_naive_drift":
        return lag + test[f"yoy_{var}"].to_numpy()
    if name == "unit_month_mean":
        return test[f"um_{var}"].to_numpy()
    model = _make_model(name, seed, n_estimators)
    anchor_train, anchor_test = train[f"um_{var}"], test[f"um_{var}"].to_numpy()
    model.fit(train[FEATURES], train[f"y_{var}"] - anchor_train)
    return anchor_test + model.predict(test[FEATURES])


# --------------------------------------------------------------------------- walk-forward
def walk_forward(
    pairs: pd.DataFrame,
    test_origins: Sequence[int],
    models: Sequence[str] = ALL_MODELS,
    seed: int = 0,
    n_estimators: int = 150,
) -> pd.DataFrame:
    """Rolling-origin backtest. At origin O, train only on pairs whose target is <= O."""
    out = []
    for O in test_origins:
        train = pairs[pairs["target"] <= O]
        test = pairs[(pairs["origin"] == O) & pairs["y_adr"].notna()]
        if train.empty or test.empty:
            continue
        for var in ("adr", "occ"):
            for name in models:
                pred = fit_predict(name, train, test, var, seed=seed, n_estimators=n_estimators)
                actual_t = test[f"y_{var}"].to_numpy()
                frame = test[["unit_id", "origin", "target", "h", "target_month", "p_true"]].copy()
                frame["var"], frame["model"] = var, name
                frame["pred_t"], frame["actual_t"] = pred, actual_t
                if var == "adr":
                    frame["pred_level"], frame["actual_level"] = np.exp(pred), test["actual_adr_mxn"].to_numpy()
                else:
                    frame["pred_level"], frame["actual_level"] = inv_logit(pred), test["actual_occ"].to_numpy()
                out.append(frame)
    return pd.concat(out, ignore_index=True)


def _metrics_for(frame: pd.DataFrame, var: str) -> Dict[str, float]:
    err = frame["pred_level"] - frame["actual_level"]
    res = {
        "n": int(len(frame)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "rmse_transformed": float(np.sqrt(((frame["pred_t"] - frame["actual_t"]) ** 2).mean())),
    }
    if var == "adr":
        res["wape"] = float(err.abs().sum() / frame["actual_level"].abs().sum())
    return res


def summarize(oof: pd.DataFrame) -> Dict[str, object]:
    """Metrics by variable x model x horizon bucket, plus skill versus the best baseline."""
    table: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for var in ("adr", "occ"):
        table[var] = {}
        sub = oof[oof["var"] == var]
        for bucket, (lo, hi) in {**H_BUCKETS, "all": (1, MAX_H)}.items():
            sb = sub[(sub["h"] >= lo) & (sub["h"] <= hi)]
            per_model = {m: _metrics_for(sb[sb["model"] == m], var) for m in sb["model"].unique()}
            best_base = min(
                (per_model[m]["rmse_transformed"] for m in BASELINES if m in per_model), default=np.nan
            )
            for m, d in per_model.items():
                d["skill_vs_best_baseline"] = float(1.0 - d["rmse_transformed"] / best_base)
            table[var][bucket] = per_model
    # Noise floor for occupancy: the error of a PERFECT forecaster of the latent probability.
    floor = None
    occ_rows = oof[(oof["var"] == "occ") & (oof["model"] == "seasonal_naive")]
    if occ_rows["p_true"].notna().all() and len(occ_rows):
        floor = float((occ_rows["actual_level"] - occ_rows["p_true"]).abs().mean())
    return {"metrics": table, "occupancy_mae_noise_floor": floor}


def choose_models(oof: pd.DataFrame) -> Dict[str, str]:
    """Pick, per target, the model with the lowest overall transformed-scale RMSE."""
    chosen = {}
    for var in ("adr", "occ"):
        sub = oof[oof["var"] == var]
        scores = {
            m: float(np.sqrt(((g["pred_t"] - g["actual_t"]) ** 2).mean())) for m, g in sub.groupby("model")
        }
        chosen[var] = min(scores, key=scores.get)
    return chosen


def oof_residuals(
    oof: pd.DataFrame, chosen: Dict[str, str], center: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Residuals (actual - predicted, transformed scale) of the chosen model for each target.

    With center=True the mean residual is removed so that bootstrapping the residuals adds
    spread but not a bias that is specific to the backtest window. The removed bias is
    returned (and reported) separately.
    """
    parts = []
    for var in ("adr", "occ"):
        sub = oof[(oof["var"] == var) & (oof["model"] == chosen[var])]
        parts.append(
            sub.assign(res=sub["actual_t"] - sub["pred_t"])[["unit_id", "origin", "h", "res"]].rename(
                columns={"res": f"res_{var}"}
            )
        )
    res = parts[0].merge(parts[1], on=["unit_id", "origin", "h"])
    bias = {"adr": float(res["res_adr"].mean()), "occ": float(res["res_occ"].mean())}
    if center:
        res["res_adr"] -= bias["adr"]
        res["res_occ"] -= bias["occ"]
    return res, bias


# --------------------------------------------------------------------------- final forecasts
def final_forecasts(
    panel: pd.DataFrame, pairs_all: pd.DataFrame, chosen: Dict[str, str], seed: int = 0, n_estimators: int = 150
) -> pd.DataFrame:
    """Forecast h=1..12 months past the last observed month for every unit."""
    last_obs = int(pairs_all.loc[pairs_all["y_adr"].notna(), "target"].max())
    train = pairs_all[pairs_all["y_adr"].notna()]
    future = pairs_all[(pairs_all["origin"] == last_obs) & pairs_all["y_adr"].isna()].copy()
    for var in ("adr", "occ"):
        future[f"pred_{var}_t"] = fit_predict(chosen[var], train, future, var, seed=seed, n_estimators=n_estimators)
    out = future[["unit_id", "target_month", "month_num", "h"]].copy()
    out = out.rename(columns={"target_month": "year_month"})
    out["pred_log_adr"] = future["pred_adr_t"].to_numpy()
    out["pred_logit_occ"] = future["pred_occ_t"].to_numpy()
    out["pred_adr_mxn"] = np.round(np.exp(out["pred_log_adr"]), 2)
    out["pred_occ_rate"] = np.round(inv_logit(out["pred_logit_occ"]), 4)
    return out.sort_values(["unit_id", "h"]).reset_index(drop=True)


# --------------------------------------------------------------------------- pipeline
def run_pipeline(
    panel: pd.DataFrame,
    out_dir: Optional[Path] = None,
    seed: int = 0,
    n_estimators: int = 150,
    first_test_origin: Optional[int] = None,
    models: Sequence[str] = ALL_MODELS,
    save_oof_predictions: bool = False,
) -> Dict[str, object]:
    """Backtest, choose models, build final forecasts and residuals; optionally write artifacts."""
    pairs_all = build_pairs(panel, include_future=True)
    labelled = pairs_all[pairs_all["y_adr"].notna()]
    T = int(labelled["target"].max()) + 1
    if first_test_origin is None:
        first_test_origin = max(MIN_ORIGIN + 12, T - 18)
    origins = list(range(first_test_origin, T - 1))
    oof = walk_forward(labelled, origins, models=models, seed=seed, n_estimators=n_estimators)
    summary = summarize(oof)
    chosen = choose_models(oof)
    resid, bias = oof_residuals(oof, chosen)
    forecasts = final_forecasts(panel, pairs_all, chosen, seed=seed, n_estimators=n_estimators)
    months = sorted(panel["year_month"].unique())
    result = {
        "last_observed_month": months[-1],
        "n_units": int(panel["unit_id"].nunique()),
        "n_months": len(months),
        "test_origins": [int(origins[0]), int(origins[-1])] if origins else [],
        "n_oof_rows": int(len(oof) // len(models) // 2),
        "chosen_models": chosen,
        "residual_bias_removed_transformed_scale": bias,
        "synthetic_data": bool(panel.attrs.get("synthetic", False)),
        "true_params": panel.attrs.get("true_params"),
        **summary,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (out_dir / "reports").mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(out_dir / "artifacts" / "forecasts.csv", index=False)
        resid.to_csv(out_dir / "artifacts" / "oof_residuals.csv", index=False)
        if save_oof_predictions:  # ~2 MB; off by default so the repo stays small
            oof.drop(columns=["p_true"]).to_csv(out_dir / "artifacts" / "oof_predictions.csv", index=False)
    return {"result": result, "forecasts": forecasts, "residuals": resid, "oof": oof}
