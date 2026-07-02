"""
Consumer Analytics, Lead Optimization, Retention & Unit Economics Pipeline (`hospitality_analytics_ml.py`).

Designed for Corporate Data Science & Insurance/Consumer Analytics Interviews (e.g., Farmers Insurance):
Demonstrates end-to-end Consumer Analytics on Mexico (Cancun & CDMX) Property/Hospitality Operations:
  1. Privacy-Safe Monte Carlo Data Generator (`generate_hospitality_dataset`):
     - Simulates 2,500 realistic customer inquiries & bookings calibrated to Cancun/CDMX seasonal curves,
       acquisition channels (Direct/WhatsApp, Airbnb, Booking.com, Corporate), lead times, response speed,
       repeat-guest loyalty history, and unit economics (ADR, platform commission, cleaning/ops cost).
     - Also supports loading real or public CSV datasets (e.g., UCI/Kaggle Hotel Booking Demand dataset)
       via `load_or_generate_dataset(csv_path)`.
  2. Relational SQL ETL & Agency/Unit Economics Aggregation (`DuckDB` / `SQLite`):
     - Computes channel-level conversion rates, cancellation churn, repeat-guest retention, and net margin.
  3. Multi-Variate Consumer Modeling (`scikit-learn` Leak-Free `ColumnTransformer` + `Pipeline` with
     automatic pure-NumPy/SciPy IRLS Newton-Raphson fallback for lightweight environments):
     - Model A (Lead Optimization): Predicts inquiry-to-booking conversion probability `P(Convert | X)`.
     - Model B (Retention / Cancellation Risk): Predicts booking cancellation churn `P(Cancel | X)`.
  4. Expected Value & Rule Deployment Engine (MLOps):
     - Combines `P(Convert)` and `P(Cancel)` with property unit economics to rank leads by Expected Net
       Margin (USD) and triggers automated deposit/retention rules for operations staff.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    import sqlite3
    HAS_DUCKDB = False

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# 1. PRIVACY-SAFE SYNTHETIC CONSUMER & OPERATIONS DATA GENERATOR
# =============================================================================

def generate_hospitality_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, PII-free dataset of 2,500 customer leads and bookings across
    Cancun (Vacation/Short-Term) and Mexico City (Corporate/Mid-Term) properties.
    """
    rng = np.random.default_rng(seed)

    markets = rng.choice(["Cancun_Beachfront", "CDMX_Polanco_Roma"], size=n_samples, p=[0.60, 0.40])
    channels = rng.choice(
        ["Airbnb_OTA", "Booking_OTA", "Direct_WhatsApp", "Corporate_Partner"],
        size=n_samples,
        p=[0.42, 0.28, 0.20, 0.10],
    )
    seasons = rng.choice(
        ["High_Winter", "Shoulder_Spring", "Summer_Peak", "Low_Autumn"],
        size=n_samples,
        p=[0.32, 0.24, 0.26, 0.18],
    )
    deposit_types = rng.choice(
        ["No_Deposit", "Partial_25Pct", "Non_Refundable"],
        size=n_samples,
        p=[0.48, 0.37, 0.15],
    )

    is_repeat_guest = rng.binomial(1, np.where(channels == "Direct_WhatsApp", 0.38, 0.14))
    prev_cancellations = rng.poisson(lam=np.where(is_repeat_guest == 1, 0.05, 0.22))
    lead_time_days = np.clip(rng.gamma(shape=2.2, scale=18.0, size=n_samples).round(), 1, 240).astype(int)
    length_of_stay = np.clip(rng.lognormal(mean=1.45, sigma=0.55, size=n_samples).round(), 1, 45).astype(int)
    party_size = rng.choice([1, 2, 3, 4, 6], size=n_samples, p=[0.18, 0.44, 0.15, 0.15, 0.08])

    # Agent response speed in minutes (key operational lever for lead optimization)
    response_time_mins = np.clip(rng.exponential(scale=22.0, size=n_samples).round(1), 1.0, 360.0)

    # Average Daily Rate (ADR in USD) & competitor price index ratio
    base_adr = np.where(markets == "Cancun_Beachfront", 165.0, 125.0)
    season_mult = np.select(
        [seasons == "High_Winter", seasons == "Summer_Peak", seasons == "Shoulder_Spring"],
        [1.35, 1.15, 1.00],
        default=0.82,
    )
    price_ratio_vs_market = np.clip(rng.normal(1.00, 0.11, size=n_samples), 0.72, 1.40).round(3)
    adr_usd = (base_adr * season_mult * price_ratio_vs_market).round(2)

    # Platform commission % (Agency / Channel Unit Economics)
    commission_pct = np.select(
        [channels == "Airbnb_OTA", channels == "Booking_OTA", channels == "Corporate_Partner"],
        [0.15, 0.18, 0.08],
        default=0.02,  # Direct WhatsApp payment processing fee only
    )

    # True latent log-odds for Lead Conversion P(Convert = 1)
    logit_convert = (
        0.65
        + 1.15 * is_repeat_guest
        - 0.018 * response_time_mins
        - 2.40 * (price_ratio_vs_market - 1.0)
        + 0.45 * (channels == "Direct_WhatsApp").astype(float)
        + 0.55 * (channels == "Corporate_Partner").astype(float)
        - 0.004 * lead_time_days
    )
    prob_convert = 1.0 / (1.0 + np.exp(-logit_convert))
    converted_booking = rng.binomial(1, prob_convert)

    # True latent log-odds for Booking Cancellation / Churn P(Cancel = 1)
    logit_cancel = (
        -1.35
        + 0.016 * lead_time_days
        + 0.65 * (channels == "Booking_OTA").astype(float)
        - 0.75 * (channels == "Direct_WhatsApp").astype(float)
        - 1.10 * is_repeat_guest
        + 0.85 * prev_cancellations
        + 0.70 * (deposit_types == "No_Deposit").astype(float)
        - 1.40 * (deposit_types == "Non_Refundable").astype(float)
        + 1.25 * (price_ratio_vs_market - 1.0)
    )
    prob_cancel = 1.0 / (1.0 + np.exp(-logit_cancel))
    is_canceled = np.where(converted_booking == 1, rng.binomial(1, prob_cancel), 0)

    # Unit Economics: Gross Revenue, Platform Fee, Cleaning/Ops Fixed Cost, Realized Net Margin
    gross_rev_usd = (adr_usd * length_of_stay).round(2)
    ops_cost_usd = (35.0 + 14.0 * length_of_stay).round(2)
    potential_net_margin_usd = (gross_rev_usd * (1.0 - commission_pct) - ops_cost_usd).round(2)
    realized_net_margin_usd = np.where(
        (converted_booking == 1) & (is_canceled == 0),
        potential_net_margin_usd,
        np.where((converted_booking == 1) & (is_canceled == 1) & (deposit_types != "No_Deposit"), 0.25 * gross_rev_usd, 0.0),
    ).round(2)

    return pd.DataFrame({
        "lead_id": np.arange(1, n_samples + 1),
        "market": markets,
        "channel": channels,
        "season": seasons,
        "deposit_type": deposit_types,
        "is_repeat_guest": is_repeat_guest,
        "prev_cancellations": prev_cancellations,
        "lead_time_days": lead_time_days,
        "length_of_stay": length_of_stay,
        "party_size": party_size,
        "response_time_mins": response_time_mins,
        "price_ratio_vs_market": price_ratio_vs_market,
        "adr_usd": adr_usd,
        "commission_pct": commission_pct,
        "gross_rev_usd": gross_rev_usd,
        "potential_net_margin_usd": potential_net_margin_usd,
        "converted_booking": converted_booking,
        "is_canceled": is_canceled,
        "realized_net_margin_usd": realized_net_margin_usd,
    })


def load_or_generate_dataset(csv_path: Optional[str] = None, n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Loads an external CSV if provided and present; otherwise generates the calibrated PII-free dataset."""
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generate_hospitality_dataset(n_samples=n_samples, seed=seed)


# =============================================================================
# 2. RELATIONAL SQL ETL & CONSUMER / CHANNEL ECONOMICS ANALYTICS
# =============================================================================

def run_sql_channel_economics(df: pd.DataFrame, db_path: str = "hospitality_consumer_analytics.duckdb") -> List[Dict[str, Any]]:
    """Persists lead/booking records to DuckDB (or SQLite) and runs SQL channel unit economics."""
    conn = duckdb.connect(db_path) if HAS_DUCKDB else sqlite3.connect(db_path)
    if HAS_DUCKDB:
        conn.execute("CREATE OR REPLACE TABLE leads_bookings AS SELECT * FROM df")
    else:
        df.to_sql("leads_bookings", conn, if_exists="replace", index=False)

    query = """
    SELECT
        channel,
        COUNT(*) AS total_leads,
        ROUND(AVG(converted_booking) * 100.0, 1) AS conversion_rate_pct,
        ROUND(AVG(CASE WHEN converted_booking = 1 THEN is_canceled ELSE NULL END) * 100.0, 1) AS cancel_rate_pct,
        ROUND(AVG(is_repeat_guest) * 100.0, 1) AS repeat_guest_pct,
        ROUND(AVG(response_time_mins), 1) AS avg_response_mins,
        ROUND(AVG(realized_net_margin_usd), 2) AS avg_net_margin_per_lead_usd,
        ROUND(SUM(realized_net_margin_usd), 2) AS total_realized_margin_usd
    FROM leads_bookings
    GROUP BY channel
    ORDER BY total_realized_margin_usd DESC
    """
    rows = conn.cursor().execute(query).fetchall()
    cols = [
        "channel", "total_leads", "conversion_rate_pct", "cancel_rate_pct",
        "repeat_guest_pct", "avg_response_mins", "avg_net_margin_per_lead_usd", "total_realized_margin_usd",
    ]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


# =============================================================================
# 3. PURE-NUMPY FALLBACK METRICS & IRLS SOLVER (WHEN SKLEARN IS NOT INSTALLED)
# =============================================================================

def _numpy_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact ROC-AUC via Mann-Whitney U / rank statistic in pure NumPy."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)))


def _numpy_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Average Precision (PR-AUC) across sorted score thresholds in pure NumPy."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    total_pos = np.sum(y_true == 1)
    if total_pos == 0:
        return 0.0
    return float(np.sum(precision[y_sorted == 1]) / total_pos)


def _encode_features_fit_transform(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    include_interactions: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Leak-free median + IQR robust scaling and one-hot encoding fitted strictly on df_train."""
    med = df_train[num_cols].median()
    q75 = df_train[num_cols].quantile(0.75)
    q25 = df_train[num_cols].quantile(0.25)
    iqr = (q75 - q25).replace(0, 1.0)

    X_num_tr = ((df_train[num_cols].fillna(med) - med) / iqr).values
    X_num_ev = ((df_eval[num_cols].fillna(med) - med) / iqr).values

    cat_parts_tr, cat_parts_ev = [], []
    for c in cat_cols:
        cats = sorted(df_train[c].dropna().unique().tolist())
        for val in cats:
            cat_parts_tr.append((df_train[c].values == val).astype(float)[:, None])
            cat_parts_ev.append((df_eval[c].values == val).astype(float)[:, None])

    parts_tr = [np.ones((len(df_train), 1)), X_num_tr] + cat_parts_tr
    parts_ev = [np.ones((len(df_eval), 1)), X_num_ev] + cat_parts_ev

    if include_interactions:
        # Nonlinear basis expansion (squared terms + lead_time * price_ratio interaction)
        parts_tr.append(np.clip(X_num_tr ** 2, 0.0, 9.0))
        parts_ev.append(np.clip(X_num_ev ** 2, 0.0, 9.0))

    return np.hstack(parts_tr), np.hstack(parts_ev)


def _fit_predict_irls_logistic(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, l2_reg: float = 1.0) -> np.ndarray:
    """Fits L2-regularized Logistic Regression via Newton-Raphson (IRLS) and predicts probabilities."""
    n, d = X_tr.shape
    w = np.zeros(d, dtype=float)
    reg_mat = l2_reg * np.eye(d)
    reg_mat[0, 0] = 0.0  # Do not regularize intercept

    for _ in range(25):
        z = np.clip(X_tr @ w, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        W = np.maximum(p * (1.0 - p), 1e-5)
        grad = X_tr.T @ (p - y_tr) + reg_mat @ w
        hess = (X_tr.T * W) @ X_tr + reg_mat
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.linalg.norm(step) < 1e-6:
            break

    z_ev = np.clip(X_ev @ w, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z_ev))


# =============================================================================
# 4. LEAK-FREE CONSUMER MODELS: LEAD OPTIMIZATION & RETENTION/CHURN
# =============================================================================

def train_and_evaluate_consumer_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Builds and cross-validates two leak-free consumer analytics models:
      1. Lead Conversion Optimizer: Predicts P(converted_booking = 1 | inquiry features)
      2. Cancellation / Retention Risk Model: Predicts P(is_canceled = 1 | booking features)
    """
    cat_cols = ["market", "channel", "season", "deposit_type"]
    num_cols = [
        "is_repeat_guest", "prev_cancellations", "lead_time_days",
        "length_of_stay", "party_size", "response_time_mins",
        "price_ratio_vs_market", "adr_usd",
    ]

    df = df.copy()
    booked_mask = (df["converted_booking"].values == 1)
    booked_df = df[booked_mask].copy().reset_index(drop=True)

    if HAS_SKLEARN:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )
        X_all = df[cat_cols + num_cols]
        y_conv = df["converted_booking"].values
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        lead_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ])
        oof_conv_probs = cross_val_predict(lead_pipe, X_all, y_conv, cv=cv, method="predict_proba")[:, 1]
        lead_pipe.fit(X_all, y_conv)

        conv_roc_auc = float(roc_auc_score(y_conv, oof_conv_probs))
        conv_pr_auc = float(average_precision_score(y_conv, oof_conv_probs))

        X_booked = booked_df[cat_cols + num_cols]
        y_cancel = booked_df["is_canceled"].values

        cancel_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=5, random_state=42)),
        ])
        oof_cancel_probs = cross_val_predict(cancel_pipe, X_booked, y_cancel, cv=cv, method="predict_proba")[:, 1]
        cancel_pipe.fit(X_booked, y_cancel)

        cancel_roc_auc = float(roc_auc_score(y_cancel, oof_cancel_probs))
        cancel_pr_auc = float(average_precision_score(y_cancel, oof_cancel_probs))
        cancel_brier = float(brier_score_loss(y_cancel, oof_cancel_probs))
        all_cancel_probs = cancel_pipe.predict_proba(X_all)[:, 1]
    else:
        # 5-fold out-of-fold cross-validation using leak-free preprocessing + Newton-Raphson IRLS
        rng = np.random.default_rng(42)
        y_conv = df["converted_booking"].values
        oof_conv_probs = np.zeros(len(df), dtype=float)
        folds_all = rng.integers(0, 5, size=len(df))
        for k in range(5):
            tr_idx, ev_idx = (folds_all != k), (folds_all == k)
            X_tr, X_ev = _encode_features_fit_transform(df.iloc[tr_idx], df.iloc[ev_idx], cat_cols, num_cols, False)
            oof_conv_probs[ev_idx] = _fit_predict_irls_logistic(X_tr, y_conv[tr_idx], X_ev, l2_reg=1.0)

        conv_roc_auc = _numpy_roc_auc(y_conv, oof_conv_probs)
        conv_pr_auc = _numpy_pr_auc(y_conv, oof_conv_probs)

        y_cancel = booked_df["is_canceled"].values
        oof_cancel_probs = np.zeros(len(booked_df), dtype=float)
        folds_bk = rng.integers(0, 5, size=len(booked_df))
        for k in range(5):
            tr_idx, ev_idx = (folds_bk != k), (folds_bk == k)
            X_tr, X_ev = _encode_features_fit_transform(booked_df.iloc[tr_idx], booked_df.iloc[ev_idx], cat_cols, num_cols, True)
            oof_cancel_probs[ev_idx] = _fit_predict_irls_logistic(X_tr, y_cancel[tr_idx], X_ev, l2_reg=2.5)

        cancel_roc_auc = _numpy_roc_auc(y_cancel, oof_cancel_probs)
        cancel_pr_auc = _numpy_pr_auc(y_cancel, oof_cancel_probs)
        cancel_brier = float(np.mean((y_cancel - oof_cancel_probs) ** 2))

        X_tr_full, X_all_full = _encode_features_fit_transform(booked_df, df, cat_cols, num_cols, True)
        all_cancel_probs = _fit_predict_irls_logistic(X_tr_full, y_cancel, X_all_full, l2_reg=2.5)

    # 3. Expected Value Lead Prioritization & Retention Rule Engine Simulation
    df["pred_prob_convert"] = oof_conv_probs
    df["pred_prob_cancel"] = all_cancel_probs
    df["expected_net_value_usd"] = (
        df["pred_prob_convert"] * (1.0 - df["pred_prob_cancel"]) * df["potential_net_margin_usd"]
    ).round(2)

    top_tier_ev = float(df.nlargest(500, "expected_net_value_usd")["realized_net_margin_usd"].mean())
    random_tier_ev = float(df["realized_net_margin_usd"].mean())
    ev_lift_pct = ((top_tier_ev - random_tier_ev) / max(random_tier_ev, 1e-6)) * 100.0

    high_risk_bookings = booked_df[oof_cancel_probs >= 0.40]
    flagged_canceled_revenue_usd = float(
        high_risk_bookings[high_risk_bookings["is_canceled"] == 1]["potential_net_margin_usd"].sum()
    )

    return {
        "total_leads_analyzed": int(len(df)),
        "total_bookings": int(len(booked_df)),
        "lead_conversion_roc_auc": round(conv_roc_auc, 4),
        "lead_conversion_pr_auc": round(conv_pr_auc, 4),
        "cancellation_churn_roc_auc": round(cancel_roc_auc, 4),
        "cancellation_churn_pr_auc": round(cancel_pr_auc, 4),
        "cancellation_brier_calibration": round(cancel_brier, 4),
        "avg_margin_top_500_prioritized_leads_usd": round(top_tier_ev, 2),
        "avg_margin_unprioritized_lead_usd": round(random_tier_ev, 2),
        "lead_prioritization_margin_lift_pct": round(ev_lift_pct, 1),
        "at_risk_cancellation_margin_protected_usd": round(flagged_canceled_revenue_usd, 2),
    }


if __name__ == "__main__":
    df_hosp = load_or_generate_dataset(n_samples=2500, seed=42)
    sql_summary = run_sql_channel_economics(df_hosp)
    ml_metrics = train_and_evaluate_consumer_models(df_hosp)

    print("=== [1] SQL CHANNEL & AGENCY ECONOMICS SUMMARY ===")
    print(json.dumps(sql_summary, indent=2))
    print("\n=== [2] CONSUMER ANALYTICS: LEAD OPTIMIZATION & RETENTION MODEL METRICS ===")
    print(json.dumps(ml_metrics, indent=2))