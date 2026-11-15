"""
Phase 2 CLI & Re-export Wrapper: ADR (Model A) and Occupancy (Model B) Training & Backtesting.
Delegates all logic to `src.rental_engine.models`.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rental_engine.data import load_training_dataframe  # noqa: E402
from src.rental_engine.models import (  # noqa: E402
    FEATURE_COLUMNS,
    HAS_XGBOOST,
    SeasonalMedianBaseline,
    TrainedModelBundle,
    add_temporal_features,
    chronological_train_test_split,
    predict_unit_monthly_forecasts,
    train_and_evaluate_models,
)

__all__ = [
    "FEATURE_COLUMNS",
    "HAS_XGBOOST",
    "SeasonalMedianBaseline",
    "TrainedModelBundle",
    "add_temporal_features",
    "chronological_train_test_split",
    "predict_unit_monthly_forecasts",
    "train_and_evaluate_models",
]


if __name__ == "__main__":
    df = load_training_dataframe()
    bundle = train_and_evaluate_models(df, cutoff_year_month="2025-06")
    print(f"Active model family: {bundle.active_model_name} (train={bundle.train_rows}, test={bundle.test_rows})")
    print(f"Backtest residual stds -> ADR: {bundle.adr_residual_std_mxn:.2f} MXN | Occ: {bundle.occ_residual_std * 100:.2f} pts")
    print(bundle.backtest_metrics_df.to_string(index=False))
