# Rental Strategy Optimizer — Riviera Maya & CDMX

An end-to-end decision engine (**SQLite $\rightarrow$ Chronological ML Backtesting $\rightarrow$ 5,000-Run Monte Carlo Simulation $\rightarrow$ Bilingual Streamlit UI**) that evaluates whether a property owner in **Playa del Carmen** or **Tulum** should **accept a guaranteed 3-month or 6-month lease**, **negotiate a higher monthly rate**, or **keep the calendar open** for short-term stays (Airbnb + 0%-fee Facebook Group direct bookings).

🔗 **Live Interactive Demo (English / Español):** [https://casabella-eriksausa.streamlit.app/](https://casabella-eriksausa.streamlit.app/)

---

## 1. Synthetic Demo Data & Privacy Note

To protect business and guest privacy, no real booking records, guest PII, or proprietary unit databases are stored in this repository:
- `src/rental_engine/data.py` generates a deterministic synthetic monthly history (`source = 'synthetic_demo'`) for four anonymized Riviera Maya apartments (**Unit A Playa**, **Unit B Playa**, **Unit C Playa**, and **Unit Tulum**) plus an urban benchmark (**CDMX Roma Norte**).
- Market-level anchors—such as **Tulum's ~46% annual mean occupancy ($125 ADR)** and **Playa del Carmen's ~58% annual mean occupancy ($103 ADR)**—are cited from public **AirDNA MarketMinder (2026)** summaries.
- `fetch_airroi_market_cached()` in `src/rental_engine/data.py` preserves an offline-first adapter for cached AirROI JSON responses (`data/raw/`) so private market pulls can be integrated locally without API credit waste (`use_live_api=False` by default).

---

## 2. Domain Mechanics & Official CFE DAC 12-Month Rolling Rule

1. **CFE Domestic High-Consumption (`Tarifa DAC`) 12-Month Rolling Average Rule**:
   - Under official CFE rules ([CFE Tarifas Domésticas & DAC](https://app.cfe.mx/Aplicaciones/CCFE/Tarifas/TarifasCRECasa/Casa.aspx), accessed 2026-11-15), a domestic meter is **not** reclassified into DAC from a single month in isolation. Instead, CFE evaluates the **trailing 12-month rolling average** consumption (last 365 billed days / 6 bimonthly bills).
   - When the 12-month rolling average exceeds the locality's tariff threshold (**400 kWh/month** for **Tariff 1B**, or 850 kWh/month for Tariff 1C), the meter loses government subsidy and is billed at the unsubsidized **CFE DAC rate** (`~6.08 MXN/kWh + 16% IVA`).
   - **Winter Persistence & Lease Carryover**: Once reclassified into DAC, *every* month (including winter) is billed at DAC until the 12-month rolling average drops back at or below `400 kWh/month`. On a 3M/6M lease, the tenant pays CFE directly via meter readings—saving the owner `$33,000–$39,000 MXN` over a 6-month summer window—though heavy tenant A/C kWh still counts toward the meter's 12-month rolling average.
2. **Blended Booking Channel Economics (Facebook Groups vs. Airbnb)**:
   - Direct bookings sourced from Riviera Maya Facebook Groups carry **0% platform fee**, whereas Airbnb bookings incur `~16.5%–22%` combined platform and Quintana Roo lodging tax drag.
3. **Exact Calendar Month Lengths**:
   - `simulate_lease_vs_str()` and `compute_break_even_occupancy()` use exact calendar days per month (`calendar.monthrange`, e.g., 28 days in Feb, 31 in May/Jul/Aug) rather than a fixed 30-day approximation.

---

## 3. Explicit Assumptions Table (`src/rental_engine/assumptions.py`)

All modeling constants live in the frozen `Assumptions` dataclass in [`src/rental_engine/assumptions.py`](src/rental_engine/assumptions.py):

| Constant / Field | Default Value | Source / Rationale |
|---|---|---|
| `cfe_tariff` / `cfe_dac_threshold_kwh` | `"1B"` / `400.0 kWh/mo` | CFE Tarifa 1B 12-month rolling average limit (accessed 2026-11-15); Tariffs `1`..`1F` (`250`–`2,500 kWh/mo`) also mapped |
| `cfe_dac_rate_mxn_per_kwh` | `6.08 MXN/kWh` (+16% IVA) | CFE Peninsular unsubsidized DAC energy rate before IVA |
| `cfe_dac_fixed_charge_mxn` | `145.79 MXN/mo` (+16% IVA) | CFE Peninsular monthly fixed charge before IVA (optional in kWh trajectory mode) |
| `cfe_subsidized_rate_mxn_per_kwh` | `1.85 MXN/kWh` (+16% IVA) | Blended subsidized Tariff 1B rate below the DAC rolling threshold |
| `adr_std_mxn` / `occ_std` | Computed from ML backtest residuals | Out-of-time test residuals (`year_month > 2025-06`); fallbacks `180.0 MXN` and `0.08` |
| `min_sampled_adr_mxn` | `300.0 MXN` | Lower truncation bound on simulated nightly ADR draws |
| `min_sampled_occ_rate` / `max_sampled_occ_rate` | `0.05` (5%) / `0.98` (98%) | Physical occupancy bounds applied to Monte Carlo normal draws |
| `accept_lease_max_str_win_prob` | `0.45` (45%) | Recommend `ACCEPT CONTRACT` when `Lease Net >= P50 STR` and `P(STR > Lease) < 0.45` |
| `keep_open_min_str_win_prob` | `0.65` (65%) | Recommend `KEEP CALENDAR OPEN` when `P(STR > Lease) > 0.65` |
| `counter_offer_risk_premium` | `1.05` (+5%) | Risk-premium markup over indifference monthly rent for counter-offers |

---

## 4. Machine Learning Pipeline & Limitations

### Chronological Training & Residual Backtesting (`src/rental_engine/models.py`)
- **Strict Temporal Split**: `chronological_train_test_split()` partitions the panel at `cutoff_year_month = "2025-06"` (`90` training unit-months through June 2025, `30` out-of-time test unit-months from July to December 2025).
- **Models Fitted**: Compares a leak-free `SeasonalMedianBaseline` against `RandomForestRegressor` (`scikit-learn`) and `XGBRegressor` (`xgboost`) for both **Model A (Nightly ADR in MXN)** and **Model B (Monthly Occupancy Rate)**.
- **Data-Driven Monte Carlo Spread**: Out-of-time residual standard deviations ($\sigma_{\text{ADR}}$ and $\sigma_{\text{Occ}}$) from the test partition are passed directly into `simulate_lease_vs_str()`.

### Known Limitations
1. **Synthetic Demo Panel Circularity**: Because the public repo ships with synthetic monthly history generated around seasonal curves (`src/rental_engine/data.py`), the out-of-time backtest metrics in the demo verify pipeline correctness and leak-free temporal splitting rather than real-world predictive accuracy. Plugging in private historical reservation rollups via `sql/schema_and_eda.sql` recalibrates both the point forecasts and residual uncertainty automatically.
2. **Independent Monthly Normal Draws**: Monte Carlo simulations draw monthly ADR and occupancy perturbations independently across months within a lease window; multi-month macroeconomic shocks (e.g., hurricane closures or sargassum waves) can create positive serial correlation across adjacent months.
3. **Simplified Tiered Subsidized CFE Billing**: Below the 400 kWh/month 12-month rolling DAC threshold, CFE Tariff 1B uses stepped basic/intermediate/surplus kWh tiers that vary between summer and non-summer months; `cfe.py` approximates non-DAC billing with a single blended subsidized rate (`1.85 MXN/kWh + 16% IVA`).
4. **Scraped Calendar Block Bias**: Third-party Airbnb datasets (AirROI / InsideAirbnb) cannot distinguish paid guest stays from owner maintenance blocks; `sql/schema_and_eda.sql` separates `stay_type = 'guest'` from `stay_type = 'owner_block'` when rolling up private reservations.

---

## 5. Repository Structure & Quickstart

```text
streamlit_app.py                # Single Streamlit entry point (bilingual EN/ES UI)
src/rental_engine/
  ├── assumptions.py            # Explicit Assumptions dataclass + cited AirDNA benchmarks
  ├── cfe.py                    # 12-month rolling-average CFE DAC classifier & bill calculator
  ├── data.py                   # Synthetic SQLite builder + cached AirROI adapter
  ├── models.py                 # Chronological split, Baseline/RF/XGB training & residual stds
  ├── engine.py                 # Break-even occupancy & Monte Carlo lease-vs-STR simulator
  └── fb_parser.py              # Safe copy-paste Facebook Group post parser
sql/schema_and_eda.sql          # SQLite schema & owner-block-adjusted rollup query
tests/test_decision_engine.py   # Unit & integration test suite
.github/workflows/tests.yml     # GitHub Actions CI pipeline
```

```bash
pip install -r requirements.txt
python src/01_load_data.py
python src/02_train_models.py
pytest -v
streamlit run streamlit_app.py
```
