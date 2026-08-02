"""
Phase 4: Interactive Streamlit Decision Dashboard for Casa Bella.
Every expense parameter (HOA, cleaning, Wi-Fi, OTA %, Facebook Direct share, and
winter/summer CFE electricity) is entered directly by the user.
"""
import importlib.util
from pathlib import Path
import pandas as pd
import streamlit as st


def _load(name: str, rel_path: str):
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(name, root / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loader = _load("load_data", "src/01_load_data.py")
engine = _load("decision_engine", "src/03_decision_engine.py")

st.set_page_config(page_title="Casa Bella Rental Strategy Optimizer", layout="wide")
st.title("Casa Bella â€” Rental Strategy Optimizer (Playa del Carmen, Tulum & CDMX)")
st.caption("Compare guaranteed 3M/6M lease offers against Short-Term / Facebook Direct stays with CFE DAC simulation.")

conn = loader.build_starter_database()
units_df = pd.read_sql_query("SELECT * FROM properties WHERE is_casa_bella = 1", conn)

with st.sidebar:
    st.header("1. Select Apartment & Lease Offer")
    unit_label = st.selectbox("Casa Bella Unit", units_df["unit_name"].tolist())
    selected_unit = units_df[units_df["unit_name"] == unit_label].iloc[0]

    start_month = st.selectbox("Contract Start Month", list(range(1, 13)), index=4)
    contract_months = st.radio("Lease Duration (Months)", [3, 6], index=1, horizontal=True)
    lease_offer_mxn = st.number_input("Tenant Monthly Offer (MXN)", value=26000.0, step=500.0)

    st.header("2. Your Unit Expenses (No Assumptions)")
    hoa_mxn = st.number_input("Monthly HOA / Maintenance (MXN)", value=2800.0, step=100.0)
    wifi_mxn = st.number_input("Monthly Wi-Fi, Water & Gas (MXN)", value=1100.0, step=100.0)
    cleaning_mxn = st.number_input("Unrecovered Cleaning Cost per Stay (MXN)", value=650.0, step=50.0)
    avg_stay = st.number_input("Average Stay Length (Nights)", value=4.5, min_value=1.0, step=0.5)

    st.header("3. Booking Channel & CFE Electricity")
    fb_share = st.slider("Facebook Group Direct Booking Share (%)", 0, 100, 60) / 100.0
    airbnb_fee = st.slider("Airbnb Platform + Lodging Tax (%)", 10.0, 25.0, 18.0, step=0.5) / 100.0
    winter_cfe = st.number_input("Winter Monthly CFE Bill on STR (Novâ€“Apr, MXN)", value=1600.0, step=200.0)
    summer_cfe = st.number_input("Summer Monthly CFE DAC Bill on STR (Mayâ€“Oct, MXN)", value=5800.0, step=200.0)

months_in_window = [((start_month - 1 + i) % 12) + 1 for i in range(contract_months)]
bench_df = pd.read_sql_query(
    "SELECT month_num, market_median_adr_mxn, market_median_occ FROM market_benchmarks WHERE unit_id IS NULL OR 1=1",
    conn,
)
unit_hist = pd.read_sql_query(
    "SELECT month_num, AVG(adr_mxn) AS pred_adr_mxn, AVG(occupancy_rate) AS pred_occ_rate "
    "FROM monthly_history WHERE unit_id = ? GROUP BY month_num",
    conn,
    params=(selected_unit["unit_id"],),
)

forecasts = []
for m in months_in_window:
    row = unit_hist[unit_hist["month_num"] == m].iloc[0]
    forecasts.append(
        {
            "month_num": int(m),
            "pred_adr_mxn": float(row["pred_adr_mxn"]),
            "pred_occ_rate": float(row["pred_occ_rate"]),
            "adr_std_mxn": 180.0,
            "occ_std": 0.08,
        }
    )

cost_inputs = engine.UserCostInputs(
    unit_id=str(selected_unit["unit_id"]),
    contract_months=int(contract_months),
    monthly_lease_offer_mxn=float(lease_offer_mxn),
    hoa_monthly_mxn=float(hoa_mxn),
    wifi_gas_water_monthly_mxn=float(wifi_mxn),
    cleaning_cost_per_stay_mxn=float(cleaning_mxn),
    avg_stay_nights=float(avg_stay),
    airbnb_fee_pct=float(airbnb_fee),
    facebook_direct_share=float(fb_share),
    winter_cfe_monthly_mxn=float(winter_cfe),
    summer_cfe_monthly_mxn=float(summer_cfe),
)

sim = engine.simulate_lease_vs_str(forecasts, cost_inputs)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Recommendation", sim["recommendation"])
col2.metric(f"{contract_months}M Lease Net Income", f"${sim['lease_total_net_mxn']:,.0f} MXN")
col3.metric(f"{contract_months}M Expected STR Net (P50)", f"${sim['str_p50_net_mxn']:,.0f} MXN")
col4.metric("Indifference Monthly Rent", f"${sim['indifference_monthly_rent_mxn']:,.0f} MXN/mo")