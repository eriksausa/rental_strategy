"""
Forecasting ADR and occupancy for rental units with leak-free walk-forward evaluation.

Design
------
* Targets are modeled on transformed scales: `log(ADR)` and `logit(occupancy)`.
* One multi-horizon "direct" model per target. Each training row is an `(origin, h)` pair:
  "standing at month `origin`, forecast month `origin + h`". Every feature uses ONLY data
  observed at or before `origin`:
      lag_{var}  same calendar month one year before the target (`target - 12`)
      um_{var}   unit's expanding average for the target calendar month up to `origin`
      rec_{var}  mean of the last 3 observed months (`origin-2 .. origin`)
      yoy_{var}  mean year-over-year change over the last 3 observed months
      dev_{var}  mean deviation of the last 3 observed months from each month's expanding mean
      month_sin / month_cos of the TARGET month, horizon `h`, bedrooms, city indicators
* ML models learn the *residual* around the unit's seasonal norm (`y - um`), so tree splits
  only need to capture deviations from baseline seasonality.
* Baselines that any ML model must beat:
      seasonal_naive         same month last year
      seasonal_naive_drift   same month last year + recent 3-month YoY change
      unit_month_mean        unit's expanding average for that calendar month
* Evaluation uses a rolling-origin (walk-forward) backtest: at each origin `O`, models train
  strictly on pairs whose `target <= O` (no look-ahead).
* Out-of-fold (OOF) residuals of the winning models are saved so the Monte Carlo engine can
  resample consecutive-horizon windows (block bootstrap), preserving both serial persistence
  and ADR/occupancy error correlation.

Everything here runs on whatever balanced panel it is given. With the synthetic demo panel,
the reported metrics describe the generator (`src/rental_engine/synthetic_panel.py`), not
the real market.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from .transforms import inv_logit, logit, smooth_occupancy

MAX_H = 12
# Need 12 months of lag history plus a 3-month YoY window: (origin - 2) - 12 >= 0 -> origin >= 14.
MIN_ORIGIN = 14
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


def _prepare_panel(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate panel balance and compute transformed targets `y_adr` and `y_occ`."""
    df = panel.sort_values(["unit_id", "year_month"]).copy()
    months = sorted(df["year_month"].unique())
    units = df["unit_id"].unique()
    if len(df) != len(units) * len(months):
        raise ValueError("forecasting requires a balanced panel (same months for every unit).")

    df["y_adr"] = np.log(df["adr_mxn"].astype(float))
    df["y_occ"] = logit(
        smooth_occupancy(df["occupancy_rate"].astype(float), df["days_in_month"].astype(float))
    )
    return df, months


def _origin_summary_features(
    log_adr: np.ndarray,
    logit_occ: np.ndarray,
    month_nums: np.ndarray,
    origin: int,
) -> Dict[str, float]:
    """Compute recent-level, YoY-drift, and seasonal-deviation features using data <= `origin`."""
    recent_slice = slice(origin - 2, origin + 1)
    prior_year_slice = slice(origin - 14, origin - 11)

    rec_adr = float(log_adr[recent_slice].mean())
    rec_occ = float(logit_occ[recent_slice].mean())
    yoy_adr = float((log_adr[recent_slice] - log_adr[prior_year_slice]).mean())
    yoy_occ = float((logit_occ[recent_slice] - logit_occ[prior_year_slice]).mean())

    # Average deviation of the last 3 observed months from each calendar month's expanding mean.
    dev_adr_vals: List[float] = []
    dev_occ_vals: List[float] = []
    for idx in range(origin - 2, origin + 1):
        same_cal_month = [k for k in range(origin + 1) if month_nums[k] == month_nums[idx]]
        dev_adr_vals.append(log_adr[idx] - log_adr[same_cal_month].mean())
        dev_occ_vals.append(logit_occ[idx] - logit_occ[same_cal_month].mean())

    return {
        "rec_adr": rec_adr,
        "rec_occ": rec_occ,
        "yoy_adr": yoy_adr,
        "yoy_occ": yoy_occ,
        "dev_adr": float(np.mean(dev_adr_vals)),
        "dev_occ": float(np.mean(dev_occ_vals)),
    }


def build_pairs(
    panel: pd.DataFrame,
    max_h: int = MAX_H,
    min_origin: int = MIN_ORIGIN,
    include_future: bool = False,
) -> pd.DataFrame:
    """Build direct multi-horizon `(unit, origin, h)` feature/target rows without look-ahead.

    Args:
        panel: Balanced unit-month panel DataFrame.
        max_h: Maximum forecast horizon in months (default 12).
        min_origin: Earliest month index used as a forecast origin (default 14).
        include_future: When True, also emits rows at the final observed origin for horizons
            `1..max_h` past the end of the panel (with NaN labels) for forward forecasting.
    """
    df, months = _prepare_panel(panel)
    n_months = len(months)
    last_period = pd.Period(months[-1], freq="M")
    month_nums = np.array([int(ym[5:7]) for ym in months])

    def calendar_month_of(time_idx: int) -> int:
        if time_idx < n_months:
            return int(month_nums[time_idx])
        return int((last_period + (time_idx - (n_months - 1))).month)

    has_ptrue = "p_true" in df.columns
    rows: List[Dict[str, object]] = []

    for unit_id, group in df.groupby("unit_id", sort=True):
        unit_series = group.set_index("year_month").loc[months]
        log_adr = unit_series["y_adr"].to_numpy()
        logit_occ = unit_series["y_occ"].to_numpy()
        adr_level = unit_series["adr_mxn"].to_numpy()
        occ_level = unit_series["occupancy_rate"].to_numpy()
        ptrue = unit_series["p_true"].to_numpy() if has_ptrue else None
        bedrooms = int(unit_series["bedrooms"].iloc[0])
        city = unit_series["city"].iloc[0]

        for origin in range(min_origin, n_months):
            origin_feats = _origin_summary_features(log_adr, logit_occ, month_nums, origin)

            for horizon in range(1, max_h + 1):
                target_idx = origin + horizon
                is_observed = target_idx <= n_months - 1
                if not is_observed and not (include_future and origin == n_months - 1):
                    continue

                target_cal_month = calendar_month_of(target_idx)
                same_month_past = [j for j in range(origin + 1) if month_nums[j] == target_cal_month]
                rows.append(
                    {
                        "unit_id": unit_id,
                        "origin": origin,
                        "target": target_idx,
                        "h": horizon,
                        "target_month": str(last_period + (target_idx - (n_months - 1))),
                        "month_num": target_cal_month,
                        "month_sin": np.sin(2.0 * np.pi * target_cal_month / 12.0),
                        "month_cos": np.cos(2.0 * np.pi * target_cal_month / 12.0),
                        "bedrooms": bedrooms,
                        "city_pdc": int(city == CITIES[0]),
                        "city_tul": int(city == CITIES[1]),
                        "city_cdmx": int(city == CITIES[2]),
                        # Because horizon <= 12, target_idx - 12 is always <= origin (never leaks future).
                        "lag_adr": log_adr[target_idx - 12],
                        "lag_occ": logit_occ[target_idx - 12],
                        **origin_feats,
                        "um_adr": log_adr[same_month_past].mean(),
                        "um_occ": logit_occ[same_month_past].mean(),
                        "y_adr": log_adr[target_idx] if is_observed else np.nan,
                        "y_occ": logit_occ[target_idx] if is_observed else np.nan,
                        "actual_adr_mxn": adr_level[target_idx] if is_observed else np.nan,
                        "actual_occ": occ_level[target_idx] if is_observed else np.nan,
                        "p_true": (ptrue[target_idx] if (is_observed and ptrue is not None) else np.nan),
                    }
                )
    return pd.DataFrame(rows)


def _make_model(name: str, seed: int, n_estimators: int):
    """Instantiate one of the supported tree-ensemble regressors."""
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            max_features=0.6,
            random_state=seed,
            n_jobs=1,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=seed,
        )
    raise ValueError(f"Unknown model name: {name!r}")


def fit_predict(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    var: str,
    seed: int = 0,
    n_estimators: int = 150,
) -> np.ndarray:
    """Fit `name` on `train` and return transformed-scale predictions (`log(ADR)` or `logit(occ)`)."""
    lag = test[f"lag_{var}"].to_numpy()
    if name == "seasonal_naive":
        return lag
    if name == "seasonal_naive_drift":
        return lag + test[f"yoy_{var}"].to_numpy()
    if name == "unit_month_mean":
        return test[f"um_{var}"].to_numpy()

    # ML models predict the deviation around the unit's expanding calendar-month average (`um`).
    model = _make_model(name, seed, n_estimators)
    anchor_train = train[f"um_{var}"]
    anchor_test = test[f"um_{var}"].to_numpy()
    model.fit(train[FEATURES], train[f"y_{var}"] - anchor_train)
    return anchor_test + model.predict(test[FEATURES])


def walk_forward(
    pairs: pd.DataFrame,
    test_origins: Sequence[int],
    models: Sequence[str] = ALL_MODELS,
    seed: int = 0,
    n_estimators: int = 150,
) -> pd.DataFrame:
    """Run a rolling-origin backtest where origin `O` trains strictly on `target <= O`."""
    frames: List[pd.DataFrame] = []
    for origin in test_origins:
        # Strict cutoff on `target <= origin` ensures no future label leaks into training.
        train = pairs[pairs["target"] <= origin]
        test = pairs[(pairs["origin"] == origin) & pairs["y_adr"].notna()]
        if train.empty or test.empty:
            continue

        for var in ("adr", "occ"):
            for name in models:
                pred_t = fit_predict(name, train, test, var, seed=seed, n_estimators=n_estimators)
                actual_t = test[f"y_{var}"].to_numpy()

                frame = test[["unit_id", "origin", "target", "h", "target_month", "p_true"]].copy()
                frame["var"] = var
                frame["model"] = name
                frame["pred_t"] = pred_t
                frame["actual_t"] = actual_t
                if var == "adr":
                    frame["pred_level"] = np.exp(pred_t)
                    frame["actual_level"] = test["actual_adr_mxn"].to_numpy()
                else:
                    frame["pred_level"] = inv_logit(pred_t)
                    frame["actual_level"] = test["actual_occ"].to_numpy()
                frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _metrics_for(frame: pd.DataFrame, var: str) -> Dict[str, float]:
    """Compute level-scale MAE/RMSE (plus WAPE for ADR) and transformed-scale RMSE."""
    err = frame["pred_level"] - frame["actual_level"]
    metrics: Dict[str, float] = {
        "n": int(len(frame)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "rmse_transformed": float(np.sqrt(((frame["pred_t"] - frame["actual_t"]) ** 2).mean())),
    }
    if var == "adr":
        metrics["wape"] = float(err.abs().sum() / frame["actual_level"].abs().sum())
    return metrics


def summarize(oof: pd.DataFrame) -> Dict[str, object]:
    """Aggregate OOF metrics by `(var, horizon_bucket, model)` and skill vs. the best baseline."""
    table: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for var in ("adr", "occ"):
        table[var] = {}
        sub = oof[oof["var"] == var]
        for bucket, (h_lo, h_hi) in {**H_BUCKETS, "all": (1, MAX_H)}.items():
            bucket_df = sub[(sub["h"] >= h_lo) & (sub["h"] <= h_hi)]
            per_model = {
                model_name: _metrics_for(bucket_df[bucket_df["model"] == model_name], var)
                for model_name in bucket_df["model"].unique()
            }
            best_baseline_rmse = min(
                (per_model[m]["rmse_transformed"] for m in BASELINES if m in per_model),
                default=np.nan,
            )
            for model_metrics in per_model.values():
                # Positive skill = beats the strongest baseline on transformed-scale RMSE.
                model_metrics["skill_vs_best_baseline"] = float(
                    1.0 - model_metrics["rmse_transformed"] / best_baseline_rmse
                )
            table[var][bucket] = per_model

    # Diagnostic noise floor for synthetic data: MAE of a forecaster that knew `p_true`.
    noise_floor = None
    occ_rows = oof[(oof["var"] == "occ") & (oof["model"] == "seasonal_naive")]
    if len(occ_rows) and occ_rows["p_true"].notna().all():
        noise_floor = float((occ_rows["actual_level"] - occ_rows["p_true"]).abs().mean())

    return {"metrics": table, "occupancy_mae_noise_floor": noise_floor}


def choose_models(oof: pd.DataFrame) -> Dict[str, str]:
    """Select, per target (`adr`, `occ`), the model with the lowest overall transformed RMSE."""
    chosen: Dict[str, str] = {}
    for var in ("adr", "occ"):
        sub = oof[oof["var"] == var]
        scores = {
            model_name: float(np.sqrt(((grp["pred_t"] - grp["actual_t"]) ** 2).mean()))
            for model_name, grp in sub.groupby("model")
        }
        chosen[var] = min(scores, key=scores.get)
    return chosen


def oof_residuals(
    oof: pd.DataFrame,
    chosen: Dict[str, str],
    center: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Extract transformed-scale residuals (`actual - predicted`) for the chosen models.

    Centering removes the backtest-window mean residual so block-bootstrapping injects
    spread and dependence without importing a systematic bias; the removed bias is
    returned separately for transparent reporting.
    """
    parts = []
    for var in ("adr", "occ"):
        sub = oof[(oof["var"] == var) & (oof["model"] == chosen[var])]
        parts.append(
            sub.assign(res=sub["actual_t"] - sub["pred_t"])[["unit_id", "origin", "h", "res"]].rename(
                columns={"res": f"res_{var}"}
            )
        )
    residuals = parts[0].merge(parts[1], on=["unit_id", "origin", "h"])
    bias = {"adr": float(residuals["res_adr"].mean()), "occ": float(residuals["res_occ"].mean())}
    if center:
        residuals["res_adr"] -= bias["adr"]
        residuals["res_occ"] -= bias["occ"]
    return residuals, bias


def final_forecasts(
    panel: pd.DataFrame,
    pairs_all: pd.DataFrame,
    chosen: Dict[str, str],
    seed: int = 0,
    n_estimators: int = 150,
) -> pd.DataFrame:
    """Refit chosen models on all observed pairs and forecast `h = 1..12` past the final month."""
    last_obs = int(pairs_all.loc[pairs_all["y_adr"].notna(), "target"].max())
    train = pairs_all[pairs_all["y_adr"].notna()]
    future = pairs_all[(pairs_all["origin"] == last_obs) & pairs_all["y_adr"].isna()].copy()

    for var in ("adr", "occ"):
        future[f"pred_{var}_t"] = fit_predict(
            chosen[var], train, future, var, seed=seed, n_estimators=n_estimators
        )

    out = future[["unit_id", "target_month", "month_num", "h"]].rename(
        columns={"target_month": "year_month"}
    )
    out["pred_log_adr"] = future["pred_adr_t"].to_numpy()
    out["pred_logit_occ"] = future["pred_occ_t"].to_numpy()
    out["pred_adr_mxn"] = np.round(np.exp(out["pred_log_adr"]), 2)
    out["pred_occ_rate"] = np.round(inv_logit(out["pred_logit_occ"]), 4)
    return out.sort_values(["unit_id", "h"]).reset_index(drop=True)


def run_pipeline(
    panel: pd.DataFrame,
    out_dir: Optional[Path] = None,
    seed: int = 0,
    n_estimators: int = 150,
    first_test_origin: Optional[int] = None,
    models: Sequence[str] = ALL_MODELS,
    save_oof_predictions: bool = False,
) -> Dict[str, object]:
    """Run the complete walk-forward backtest, model selection, final forecast, and artifact export."""
    pairs_all = build_pairs(panel, include_future=True)
    labelled = pairs_all[pairs_all["y_adr"].notna()]
    n_months = int(labelled["target"].max()) + 1

    if first_test_origin is None:
        first_test_origin = max(MIN_ORIGIN + 12, n_months - 18)
    origins = list(range(first_test_origin, n_months - 1))

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
        if save_oof_predictions:  # ~2 MB; off by default to keep the repo lightweight
            oof.drop(columns=["p_true"]).to_csv(out_dir / "artifacts" / "oof_predictions.csv", index=False)

    return {"result": result, "forecasts": forecasts, "residuals": resid, "oof": oof}
