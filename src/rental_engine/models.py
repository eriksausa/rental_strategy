"""
Machine Learning Models for Nightly ADR (Model A) and Monthly Occupancy (Model B)
with strict chronological train/test splitting and residual uncertainty estimation.

Note on synthetic demo data:
  In this public demo, models are trained and evaluated on the synthetic monthly
  panel generated in `data.py` (train <= 2025-06, test > 2025-06) so that every
  backtest metric and residual standard deviation shown in the UI is genuinely
  computed by `.fit()` and `.predict()` rather than hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .assumptions import MONTH_NAMES_EN, MONTH_NAMES_ES, PORTFOLIO_UNITS

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


FEATURE_COLUMNS: List[str] = [
    "bedrooms",
    "bathrooms",
    "month_num",
    "month_sin",
    "month_cos",
    "is_high_season",
    "is_summer_dac_season",
    "has_pool",
    "has_rooftop",
    "market_median_adr_mxn",
    "market_median_occ",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical month encodings and Riviera Maya seasonal flags."""
    out = df.copy()
    out["month_sin"] = np.sin(2.0 * np.pi * out["month_num"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month_num"] / 12.0)
    out["is_high_season"] = out["month_num"].isin([12, 1, 2, 3, 4]).astype(int)
    out["is_summer_dac_season"] = out["month_num"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    return out


def chronological_train_test_split(
    df: pd.DataFrame, cutoff_year_month: str = "2025-06"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits monthly history strictly by `year_month` so future observations never
    leak into training features.
    """
    df_sorted = df.sort_values("year_month").reset_index(drop=True)
    train_df = df_sorted[df_sorted["year_month"] <= cutoff_year_month].copy()
    test_df = df_sorted[df_sorted["year_month"] > cutoff_year_month].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Chronological split at {cutoff_year_month} produced an empty partition.")
    return train_df, test_df


class SeasonalMedianBaseline:
    """
    Per-unit, per-month median (or fallback unit/global median) baseline estimator
    implementing `.fit(X, y)` and `.predict(X)` for leak-free comparison.
    """

    def __init__(self) -> None:
        self.unit_month_medians_: Dict[Tuple[str, int], float] = {}
        self.month_medians_: Dict[int, float] = {}
        self.global_median_: float = 0.0

    def fit(self, df_features: pd.DataFrame, y: np.ndarray | pd.Series) -> "SeasonalMedianBaseline":
        work = df_features.copy()
        work["_target"] = np.asarray(y, dtype=float)
        self.global_median_ = float(work["_target"].median())
        self.month_medians_ = (
            work.groupby("month_num")["_target"].median().astype(float).to_dict()
        )
        if "unit_id" in work.columns:
            self.unit_month_medians_ = (
                work.groupby(["unit_id", "month_num"])["_target"].median().astype(float).to_dict()
            )
        return self

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        preds: List[float] = []
        has_unit = "unit_id" in df_features.columns
        for _, row in df_features.iterrows():
            m = int(row["month_num"])
            if has_unit:
                key = (str(row["unit_id"]), m)
                if key in self.unit_month_medians_:
                    preds.append(self.unit_month_medians_[key])
                    continue
            preds.append(self.month_medians_.get(m, self.global_median_))
        return np.asarray(preds, dtype=float)


@dataclass
class TrainedModelBundle:
    """Holds fitted ADR and Occupancy models, backtest metrics, and residual stds."""
    adr_model: Any
    occ_model: Any
    adr_baseline: SeasonalMedianBaseline
    occ_baseline: SeasonalMedianBaseline
    active_model_name: str
    cutoff_year_month: str
    train_rows: int
    test_rows: int
    adr_residual_std_mxn: float
    occ_residual_std: float
    backtest_metrics_df: pd.DataFrame
    unit_feature_lookup: pd.DataFrame


def train_and_evaluate_models(
    df: pd.DataFrame,
    cutoff_year_month: str = "2025-06",
    random_state: int = 42,
) -> TrainedModelBundle:
    """
    Fits three model families on the chronological train partition (`year_month <= cutoff_year_month`)
    and evaluates out-of-time MAE, RMSE, and residual standard deviation on `year_month > cutoff_year_month`:
      1. Seasonal Median Baseline
      2. Random Forest Regressor (`scikit-learn`)
      3. XGBoost Regressor (`xgboost`, when installed)
    """
    feat_df = add_temporal_features(df)
    train_df, test_df = chronological_train_test_split(feat_df, cutoff_year_month=cutoff_year_month)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_adr_train = train_df["adr_mxn"].to_numpy(dtype=float)
    y_adr_test = test_df["adr_mxn"].to_numpy(dtype=float)
    y_occ_train = train_df["occupancy_rate"].to_numpy(dtype=float)
    y_occ_test = test_df["occupancy_rate"].to_numpy(dtype=float)

    # 1. Seasonal Median Baseline
    adr_base = SeasonalMedianBaseline().fit(train_df, y_adr_train)
    occ_base = SeasonalMedianBaseline().fit(train_df, y_occ_train)
    adr_base_pred = adr_base.predict(test_df)
    occ_base_pred = np.clip(occ_base.predict(test_df), 0.05, 0.98)

    # 2. Random Forest Regressor
    rf_adr = RandomForestRegressor(
        n_estimators=150, max_depth=6, min_samples_leaf=2, random_state=random_state
    )
    rf_occ = RandomForestRegressor(
        n_estimators=150, max_depth=6, min_samples_leaf=2, random_state=random_state
    )
    rf_adr.fit(X_train, y_adr_train)
    rf_occ.fit(X_train, y_occ_train)
    rf_adr_pred = rf_adr.predict(X_test)
    rf_occ_pred = np.clip(rf_occ.predict(X_test), 0.05, 0.98)

    rows: List[Dict[str, object]] = [
        {
            "Model": "Seasonal Median Baseline",
            "Target": "Model A: Nightly ADR (MXN)",
            "MAE": round(float(mean_absolute_error(y_adr_test, adr_base_pred)), 2),
            "RMSE": round(float(root_mean_squared_error(y_adr_test, adr_base_pred)), 2),
            "Residual_Std": round(float(np.std(y_adr_test - adr_base_pred, ddof=1)), 2),
            "Units": "MXN",
        },
        {
            "Model": "Random Forest Regressor",
            "Target": "Model A: Nightly ADR (MXN)",
            "MAE": round(float(mean_absolute_error(y_adr_test, rf_adr_pred)), 2),
            "RMSE": round(float(root_mean_squared_error(y_adr_test, rf_adr_pred)), 2),
            "Residual_Std": round(float(np.std(y_adr_test - rf_adr_pred, ddof=1)), 2),
            "Units": "MXN",
        },
    ]

    best_adr_model: Any = rf_adr
    best_occ_model: Any = rf_occ
    best_adr_pred = rf_adr_pred
    best_occ_pred = rf_occ_pred
    active_name = "RandomForestRegressor"

    # 3. XGBoost Regressor (if available)
    if HAS_XGBOOST:
        xgb_adr = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbosity=0,
        )
        xgb_occ = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbosity=0,
        )
        xgb_adr.fit(X_train, y_adr_train)
        xgb_occ.fit(X_train, y_occ_train)
        xgb_adr_pred = xgb_adr.predict(X_test)
        xgb_occ_pred = np.clip(xgb_occ.predict(X_test), 0.05, 0.98)

        rows.append(
            {
                "Model": "XGBoost Regressor",
                "Target": "Model A: Nightly ADR (MXN)",
                "MAE": round(float(mean_absolute_error(y_adr_test, xgb_adr_pred)), 2),
                "RMSE": round(float(root_mean_squared_error(y_adr_test, xgb_adr_pred)), 2),
                "Residual_Std": round(float(np.std(y_adr_test - xgb_adr_pred, ddof=1)), 2),
                "Units": "MXN",
            }
        )
        if root_mean_squared_error(y_adr_test, xgb_adr_pred) <= root_mean_squared_error(y_adr_test, rf_adr_pred):
            best_adr_model = xgb_adr
            best_adr_pred = xgb_adr_pred
            active_name = "XGBRegressor"
        if root_mean_squared_error(y_occ_test, xgb_occ_pred) <= root_mean_squared_error(y_occ_test, rf_occ_pred):
            best_occ_model = xgb_occ
            best_occ_pred = xgb_occ_pred

    rows.extend(
        [
            {
                "Model": "Seasonal Median Baseline",
                "Target": "Model B: Occupancy (%)",
                "MAE": round(float(mean_absolute_error(y_occ_test, occ_base_pred) * 100.0), 2),
                "RMSE": round(float(root_mean_squared_error(y_occ_test, occ_base_pred) * 100.0), 2),
                "Residual_Std": round(float(np.std(y_occ_test - occ_base_pred, ddof=1) * 100.0), 2),
                "Units": "pct_pts",
            },
            {
                "Model": "Random Forest Regressor",
                "Target": "Model B: Occupancy (%)",
                "MAE": round(float(mean_absolute_error(y_occ_test, rf_occ_pred) * 100.0), 2),
                "RMSE": round(float(root_mean_squared_error(y_occ_test, rf_occ_pred) * 100.0), 2),
                "Residual_Std": round(float(np.std(y_occ_test - rf_occ_pred, ddof=1) * 100.0), 2),
                "Units": "pct_pts",
            },
        ]
    )

    if HAS_XGBOOST:
        rows.append(
            {
                "Model": "XGBoost Regressor",
                "Target": "Model B: Occupancy (%)",
                "MAE": round(float(mean_absolute_error(y_occ_test, xgb_occ_pred) * 100.0), 2),
                "RMSE": round(float(root_mean_squared_error(y_occ_test, xgb_occ_pred) * 100.0), 2),
                "Residual_Std": round(float(np.std(y_occ_test - xgb_occ_pred, ddof=1) * 100.0), 2),
                "Units": "pct_pts",
            }
        )

    adr_residual_std = float(np.std(y_adr_test - best_adr_pred, ddof=1))
    occ_residual_std = float(np.std(y_occ_test - best_occ_pred, ddof=1))

    # Refit selected model architectures on the full historical dataset for forward forecasting,
    # while preserving the out-of-time backtest residual standard deviations for Monte Carlo.
    X_full = feat_df[FEATURE_COLUMNS]
    best_adr_model.fit(X_full, feat_df["adr_mxn"].to_numpy(dtype=float))
    best_occ_model.fit(X_full, feat_df["occupancy_rate"].to_numpy(dtype=float))

    unit_feature_lookup = (
        feat_df[
            [
                "unit_id",
                "month_num",
                "bedrooms",
                "bathrooms",
                "month_sin",
                "month_cos",
                "is_high_season",
                "is_summer_dac_season",
                "has_pool",
                "has_rooftop",
                "market_median_adr_mxn",
                "market_median_occ",
            ]
        ]
        .drop_duplicates(subset=["unit_id", "month_num"])
        .sort_values(["unit_id", "month_num"])
        .reset_index(drop=True)
    )

    return TrainedModelBundle(
        adr_model=best_adr_model,
        occ_model=best_occ_model,
        adr_baseline=adr_base,
        occ_baseline=occ_base,
        active_model_name=active_name,
        cutoff_year_month=cutoff_year_month,
        train_rows=len(train_df),
        test_rows=len(test_df),
        adr_residual_std_mxn=round(max(adr_residual_std, 25.0), 2),
        occ_residual_std=round(max(occ_residual_std, 0.02), 4),
        backtest_metrics_df=pd.DataFrame(rows),
        unit_feature_lookup=unit_feature_lookup,
    )


def predict_unit_monthly_forecasts(
    bundle: TrainedModelBundle,
    unit_id: str,
    months_window: List[int],
    lang: str = "en",
) -> List[Dict[str, float | int | str]]:
    """
    Generates monthly ADR and Occupancy forecasts for `unit_id` over `months_window`
    using the trained ML models in `bundle`, pairing each point forecast with the
    empirical out-of-time backtest residual standard deviation (`adr_residual_std_mxn`
    and `occ_residual_std`).
    """
    month_map = MONTH_NAMES_ES if lang == "es" else MONTH_NAMES_EN
    unit_rows = bundle.unit_feature_lookup[bundle.unit_feature_lookup["unit_id"] == unit_id]
    if unit_rows.empty:
        raise KeyError(f"Unknown unit_id '{unit_id}' in trained model lookup.")

    forecasts: List[Dict[str, float | int | str]] = []
    for m in months_window:
        m_int = int(m)
        row_df = unit_rows[unit_rows["month_num"] == m_int][FEATURE_COLUMNS]
        pred_adr = float(bundle.adr_model.predict(row_df)[0])
        pred_occ = float(np.clip(bundle.occ_model.predict(row_df)[0], 0.05, 0.98))
        forecasts.append(
            {
                "month_num": m_int,
                "month_name": month_map[m_int],
                "pred_adr_mxn": round(pred_adr, 2),
                "pred_occ_rate": round(pred_occ, 4),
                "adr_std_mxn": bundle.adr_residual_std_mxn,
                "occ_std": bundle.occ_residual_std,
            }
        )
    return forecasts
