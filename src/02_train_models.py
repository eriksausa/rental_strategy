"""
Phase 2: Machine Learning Models for ADR (Model A) and Occupancy (Model B)
with strict chronological train/test splitting (zero future data leakage).
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from xgboost import XGBRegressor
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False


FEATURE_COLUMNS = [
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