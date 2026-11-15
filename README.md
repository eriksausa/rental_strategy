# Rental Strategy Optimizer (Riviera Maya & CDMX)

A Data Science & Decision Engine portfolio project built for a family-run rental business to evaluate guaranteed 3-month and 6-month mid-term lease offers against short-term rentals (Airbnb + Facebook Group direct bookings) across **Playa del Carmen**, **Tulum**, and **Mexico City (CDMX)**.

> **Demo version:** all property data in this repository is synthetic. Real unit details, databases, and booking records are intentionally excluded.

## 1. Portfolio Units & Business Problem

The portfolio consists of four apartments in the Riviera Maya:
- **Playa del Carmen**: **Unit A Playa** and **Unit B Playa** (**1BR/1BA**, `UNIT_A_PLAYA`, `UNIT_B_PLAYA`) and **Unit C Playa** (**2BR/2BA**, `UNIT_C_PLAYA`).
- **Tulum**: **Unit Tulum** (**1BR/1BA**, `UNIT_TULUM`, calibrated to the Tulum market-wide average occupancy).

When a prospective tenant offers a fixed 3-month or 6-month lease (e.g., `$25,000 MXN/month`), the owner must decide whether to **accept the contract**, **negotiate a higher monthly rate**, or **keep the calendar open** for short-term stays.

### Key Domain Mechanics Modeled
1. **The CFE DAC Summer Electricity Trap (May–Oct)**: On short-term stays, the owner pays 100% of the CFE electricity bill. In May–Oct, guest 24/7 A/C usage pushes consumption above `400 kWh/month` into the unsubsidized **CFE DAC tariff** (`~6.08 MXN/kWh + 16% IVA`, or `$5,500–$6,500 MXN/month`). On a 3M/6M lease, the **tenant pays CFE directly** via check-in/check-out meter readings.
2. **Facebook Group Direct Bookings vs. Airbnb Fees**: Direct bookings sourced from Riviera Maya Facebook Groups carry **0% platform fee**, whereas Airbnb bookings incur `~16.5%–22%` combined platform and Quintana Roo lodging tax drag.
3. **100% User-Controlled Cost Inputs**: HOA fees, per-stay cleaning costs, Wi-Fi/water, channel mix, and winter/summer CFE bills are entered directly by the user rather than assumed.

## 2. Methodological Caveats & Data Hygiene

- **Synthetic Demo Data**: `src/01_load_data.py` generates seeded synthetic monthly history (`source = 'synthetic_demo'`). The `reservations` table schema is included but ships empty.
- **Scraped Calendar Block Bias**: Third-party Airbnb datasets (such as the AirROI API and InsideAirbnb) cannot distinguish between a paid guest booking and an owner maintenance block. Our SQL schema (`sql/schema_and_eda.sql`) separates `stay_type = 'guest'` from `stay_type = 'owner_block'` in `reservations`.
- **Zero API Credit Waste**: `src/01_load_data.py` caches all external JSON responses inside `data/raw/` and stores historical tables in `data/rental_data.db` so repeated runs cost `$0.00`.

## 3. Quickstart

```bash
pip install -r requirements.txt
python src/01_load_data.py
pytest -v
streamlit run app.py
```
