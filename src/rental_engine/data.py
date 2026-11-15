"""
SQLite Synthetic Demo Database Builder & Optional Cached AirROI Market Adapter.

- Generates a deterministic synthetic SQLite database (`data/rental_data.db`) with
  `source = 'synthetic_demo'` for the 4 illustrative portfolio units + 1 CDMX benchmark.
- Keeps the cached AirROI adapter (`fetch_airroi_market_cached`) disabled by default
  (`use_live_api=False`) so private API keys or cached JSON files can be plugged in
  later without changing the pipeline.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from .assumptions import BASE_SEASONAL_OCC_CURVE, PORTFOLIO_UNITS


ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "data" / "rental_data.db"
RAW_CACHE_DIR = ROOT_DIR / "data" / "raw"
SQL_SCHEMA_PATH = ROOT_DIR / "sql" / "schema_and_eda.sql"

PORTFOLIO_UNIT_ROWS = [
    ("UNIT_A_PLAYA", "Unit A Playa", "Playa del Carmen", "Not disclosed A", 1, 1, 1, 1, 1),
    ("UNIT_B_PLAYA", "Unit B Playa", "Playa del Carmen", "Not disclosed B", 1, 1, 1, 1, 1),
    ("UNIT_C_PLAYA", "Unit C Playa", "Playa del Carmen", "Not disclosed C", 2, 2, 1, 1, 1),
    ("UNIT_TULUM", "Unit Tulum", "Tulum", "Not disclosed", 1, 1, 1, 1, 1),
    ("MKT_CDMX_1BR", "CDMX Roma Norte 1BR Benchmark", "CDMX", "Roma Norte", 1, 1, 0, 1, 0),
]


def fetch_airroi_market_cached(
    market_payload: Dict, cache_filename: str, use_live_api: bool = False
) -> Dict:
    """
    Returns cached AirROI JSON from `data/raw/<cache_filename>` ($0.00 cost).
    Only calls `POST https://api.airroi.com/markets/metrics/all` if `use_live_api=True`
    and the cache file does not yet exist. Kept for optional private market pulls.
    """
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_CACHE_DIR / cache_filename
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if not use_live_api:
        return {"status": "offline_benchmark_mode", "results": []}

    import requests

    api_key = os.getenv("AIRROI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("AIRROI_API_KEY is not set in environment.")

    resp = requests.post(
        "https://api.airroi.com/markets/metrics/all",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json=market_payload,
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def build_starter_database(db_path: Path | str | None = DB_PATH) -> sqlite3.Connection:
    """
    Creates SQLite tables and populates illustrative synthetic monthly history
    (2024-01 through 2025-12) for all 4 portfolio units and the CDMX benchmark.
    If `db_path` is None or `":memory:"`, builds an in-memory SQLite database.
    """
    if db_path is None or str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        target_path = Path(db_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target_path)

    schema_sql = SQL_SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)

    conn.executemany(
        """
        INSERT OR REPLACE INTO properties
        (unit_id, unit_name, city, neighborhood, bedrooms, bathrooms, has_pool, has_rooftop, is_portfolio_unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        PORTFOLIO_UNIT_ROWS,
    )

    rng = np.random.default_rng(42)

    unit_base_profiles = {
        uid: (float(info["base_adr_mxn"]), float(info["occ_multiplier"]))
        for uid, info in PORTFOLIO_UNITS.items()
    }

    bench_rows: List[tuple] = []
    hist_rows: List[tuple] = []

    for unit_id, _, city, neighborhood, bedrooms, bathrooms, has_pool, has_rooftop, is_portfolio in PORTFOLIO_UNIT_ROWS:
        base_adr, occ_mult = unit_base_profiles[unit_id]
        for m in range(1, 13):
            if city == "CDMX":
                season_factor = 1.15 if m in (10, 11, 3, 4) else 0.98
                mkt_occ = float(np.clip(0.64 * (1.08 if m in (10, 11) else 0.98), 0.45, 0.85))
            else:
                season_factor = 1.25 if m in (12, 1, 2, 3) else (0.82 if m in (5, 6, 8, 9, 10) else 1.0)
                mkt_occ = float(np.clip(BASE_SEASONAL_OCC_CURVE[m] * occ_mult, 0.22, 0.92))
            mkt_adr = round(base_adr * season_factor, 2)
            bench_rows.append((city, neighborhood, bedrooms, m, mkt_adr, round(mkt_occ, 4)))

            for year in (2024, 2025):
                ym = f"{year}-{m:02d}"
                realized_adr = round(mkt_adr + rng.normal(0, 75), 2)
                realized_occ = float(np.clip(mkt_occ + rng.normal(0, 0.04), 0.18, 0.95))
                src = "synthetic_demo" if is_portfolio else "benchmark"
                hist_rows.append((unit_id, ym, m, realized_adr, round(realized_occ, 4), 4.5, 0.60, src))

    conn.executemany(
        """
        INSERT OR REPLACE INTO market_benchmarks
        (city, neighborhood, bedrooms, month_num, market_median_adr_mxn, market_median_occ)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        bench_rows,
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO monthly_history
        (unit_id, year_month, month_num, adr_mxn, occupancy_rate, avg_stay_nights, facebook_share, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        hist_rows,
    )
    conn.commit()
    return conn


def load_training_dataframe(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """
    Joins `monthly_history`, `properties`, and `market_benchmarks` into a single
    modeling DataFrame ready for feature engineering and chronological splitting.
    """
    close_after = False
    if conn is None:
        conn = build_starter_database(":memory:")
        close_after = True

    query = """
        SELECT
            h.unit_id,
            h.year_month,
            h.month_num,
            h.adr_mxn,
            h.occupancy_rate,
            h.avg_stay_nights,
            h.facebook_share,
            h.source,
            p.city,
            p.neighborhood,
            p.bedrooms,
            p.bathrooms,
            p.has_pool,
            p.has_rooftop,
            p.is_portfolio_unit,
            b.market_median_adr_mxn,
            b.market_median_occ
        FROM monthly_history h
        JOIN properties p
          ON h.unit_id = p.unit_id
        JOIN market_benchmarks b
          ON p.city = b.city
         AND p.neighborhood = b.neighborhood
         AND p.bedrooms = b.bedrooms
         AND h.month_num = b.month_num
        ORDER BY h.year_month, h.unit_id
    """
    df = pd.read_sql_query(query, conn)
    if close_after:
        conn.close()
    return df
