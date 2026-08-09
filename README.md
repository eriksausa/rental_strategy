# Casa Bella â€” Rental Strategy Optimizer (Riviera Maya & CDMX)

A Data Science & Decision Engine portfolio project built for **Casa Bella** to evaluate guaranteed 3-month and 6-month mid-term lease offers against short-term rentals (Airbnb + Facebook Group direct bookings) across **Playa del Carmen**, **Tulum**, and **Mexico City (CDMX)**.

## 1. Portfolio Units & Business Problem

**Casa Bella** operates four apartments in the Riviera Maya:
- **Playa del Carmen (Centro & Gonzalo Guerrero)**: Two **1BR/1BA** apartments (`CB_PDC_1BR_A`, `CB_PDC_1BR_B`) and one **2BR/2BA** apartment (`CB_PDC_2BR`).
- **Tulum (La Veleta)**: One **1BR/1BA** apartment (`CB_TUL_1BR`, calibrated around ~46% annual mean occupancy).

When a prospective tenant offers a fixed 3-month or 6-month lease (e.g., `$26,000â€“$29,000 MXN/month`), the owner must decide whether to **accept the contract**, **negotiate a higher monthly rate**, or **keep the calendar open** for short-term stays.

### Key Domain Mechanics Modeled
1. **The CFE DAC Summer Electricity Trap (Mayâ€“Oct)**: On short-term stays, the owner pays 100% of the CFE electricity bill. In Mayâ€“Oct, guest 24/7 A/C usage pushes consumption above `400 kWh/month` into the unsubsidized **CFE DAC tariff** (`~6.08 MXN/kWh + 16% IVA`, or `$5,500â€“$6,500 MXN/month`). On a 3M/6M lease, the **tenant pays CFE directly** via check-in/check-out meter readings.
2. **Facebook Group Direct Bookings vs. Airbnb Fees**: Direct bookings sourced from Riviera Maya Facebook Groups carry **0% platform fee**, whereas Airbnb bookings incur `~16.5%â€“22%` combined platform and Quintana Roo lodging tax drag.
3. **100% User-Controlled Cost Inputs**: HOA fees, per-stay cleaning costs, Wi-Fi/water, channel mix, and winter/summer CFE bills are entered directly by the user rather than assumed.

## 2. Methodological Caveats & Data Hygiene

- **Scraped Calendar Block Bias**: Third-party Airbnb datasets (such as the AirROI API and InsideAirbnb) cannot distinguish between a paid guest booking and an owner maintenance block. Our SQL schema (`sql/schema_and_eda.sql`) separates `stay_type = 'guest'` from `stay_type = 'owner_block'` in `casa_bella_reservations`.
- **Zero API Credit Waste**: `src/01_load_data.py` caches all external JSON responses inside `data/raw/` and stores historical tables in `data/rental_data.db` so repeated runs cost `$0.00`.

## 3. Quickstart

```bash
pip install -r requirements.txt
python src/01_load_data.py
pytest -v
streamlit run app.py
```