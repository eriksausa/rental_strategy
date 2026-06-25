-- ============================================================================
-- Casa Bella Rental Strategy Optimizer: SQLite / DuckDB Schema & EDA Queries
-- Portfolio Units:
--   * Playa del Carmen (Centro / Gonzalo Guerrero): 2x 1BR/1BA, 1x 2BR/2BA
--   * Tulum (La Veleta): 1x 1BR/1BA
--   * CDMX (Roma Norte): 1x 1BR/1BA Market Comparison Benchmark
-- ============================================================================

CREATE TABLE IF NOT EXISTS properties (
    unit_id TEXT PRIMARY KEY,
    unit_name TEXT NOT NULL,
    city TEXT NOT NULL,
    neighborhood TEXT NOT NULL,
    bedrooms INTEGER NOT NULL,
    bathrooms INTEGER NOT NULL,
    has_pool INTEGER DEFAULT 1,
    has_rooftop INTEGER DEFAULT 1,
    is_casa_bella INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS casa_bella_reservations (
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id TEXT NOT NULL,
    check_in TEXT NOT NULL,
    check_out TEXT NOT NULL,
    nights INTEGER NOT NULL,
    net_payout_mxn REAL NOT NULL,
    channel TEXT NOT NULL,          -- 'facebook', 'airbnb', 'repeat_direct', 'none'
    stay_type TEXT NOT NULL,        -- 'guest' or 'owner_block'
    FOREIGN KEY (unit_id) REFERENCES properties(unit_id)
);

CREATE TABLE IF NOT EXISTS monthly_history (
    unit_id TEXT NOT NULL,
    year_month TEXT NOT NULL,       -- Format: 'YYYY-MM'
    month_num INTEGER NOT NULL,     -- 1 to 12
    adr_mxn REAL NOT NULL,
    occupancy_rate REAL NOT NULL,   -- 0.0 to 1.0
    avg_stay_nights REAL DEFAULT 4.5,
    facebook_share REAL DEFAULT 0.60,
    source TEXT NOT NULL,           -- 'casa_bella_actual', 'airroi_api', 'benchmark'
    PRIMARY KEY (unit_id, year_month)
);

CREATE TABLE IF NOT EXISTS market_benchmarks (
    city TEXT NOT NULL,
    neighborhood TEXT NOT NULL,
    bedrooms INTEGER NOT NULL,
    month_num INTEGER NOT NULL,
    market_median_adr_mxn REAL NOT NULL,
    market_median_occ REAL NOT NULL,
    PRIMARY KEY (city, neighborhood, bedrooms, month_num)
);

-- ============================================================================
-- Analytical Query 1: Roll up raw Casa Bella reservation list into monthly ADR
-- and true available occupancy (excluding family/owner maintenance blocks).
-- ============================================================================
-- SELECT
--     unit_id,
--     strftime('%Y-%m', check_in) AS year_month,
--     SUM(CASE WHEN stay_type = 'guest' THEN net_payout_mxn ELSE 0 END) /
--         NULLIF(SUM(CASE WHEN stay_type = 'guest' THEN nights ELSE 0 END), 0) AS realized_adr_mxn,
--     CAST(SUM(CASE WHEN stay_type = 'guest' THEN nights ELSE 0 END) AS REAL) /
--         NULLIF(30 - SUM(CASE WHEN stay_type = 'owner_block' THEN nights ELSE 0 END), 0) AS adjusted_occupancy,
--     AVG(CASE WHEN stay_type = 'guest' THEN nights END) AS avg_stay_nights,
--     SUM(CASE WHEN channel = 'facebook' AND stay_type = 'guest' THEN nights ELSE 0 END) * 1.0 /
--         NULLIF(SUM(CASE WHEN stay_type = 'guest' THEN nights ELSE 0 END), 0) AS facebook_channel_share
-- FROM casa_bella_reservations
-- GROUP BY unit_id, strftime('%Y-%m', check_in);