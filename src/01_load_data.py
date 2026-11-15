"""
Phase 1 CLI & Re-export Wrapper: SQLite Synthetic Demo Loader & Optional AirROI Cache Adapter.
Delegates all logic to `src.rental_engine.data`.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rental_engine.data import (  # noqa: E402
    DB_PATH,
    PORTFOLIO_UNIT_ROWS as PORTFOLIO_UNITS,
    RAW_CACHE_DIR,
    SQL_SCHEMA_PATH,
    build_starter_database,
    fetch_airroi_market_cached,
    load_training_dataframe,
)

__all__ = [
    "DB_PATH",
    "PORTFOLIO_UNITS",
    "RAW_CACHE_DIR",
    "SQL_SCHEMA_PATH",
    "build_starter_database",
    "fetch_airroi_market_cached",
    "load_training_dataframe",
]


if __name__ == "__main__":
    connection = build_starter_database()
    count = connection.execute("SELECT COUNT(*) FROM monthly_history").fetchone()[0]
    print(f"Initialized SQLite database at {DB_PATH} with {count} synthetic_demo/benchmark monthly records.")
