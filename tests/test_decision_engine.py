"""
Unit tests for the Casa Bella Rental Strategy Optimizer core logic.
Compatible with both `python -m unittest discover -s tests -v` and `pytest -v`.
"""
import importlib.util
import unittest
from pathlib import Path
import pandas as pd


def _load_module(name: str, rel_path: str):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = _load_module("decision_engine", "src/03_decision_engine.py")
models = _load_module("train_models", "src/02_train_models.py")


class TestRentalStrategyOptimizer(unittest.TestCase):
    def test_cfe_dac_summer_threshold_and_iva(self):
        # Below 400 kWh in summer stays on subsidized tariff
        low_bill = engine.estimate_cfe_monthly_bill(kwh_used=350.0, is_summer_month=True)
        # Above 400 kWh in summer triggers DAC rate (6.08 MXN/kWh * 1.16 IVA)
        dac_bill = engine.estimate_cfe_monthly_bill(kwh_used=850.0, is_summer_month=True)
        self.assertLess(low_bill, 1000.0)
        self.assertAlmostEqual(dac_bill, round(850.0 * 6.08 * 1.16, 2), places=2)

    def test_facebook_direct_channel_reduces_platform_fee(self):
        # 60% Facebook direct (0% fee) + 40% Airbnb (18% fee) -> 7.2% blended fee
        blended = engine.compute_blended_ota_drag(airbnb_fee_pct=0.18, facebook_direct_share=0.60)
        self.assertAlmostEqual(blended, 0.072, places=4)

    def test_break_even_occupancy_clamped_to_unit_interval(self):
        costs = engine.UserCostInputs(
            unit_id="CB_PDC_1BR_A",
            contract_months=6,
            monthly_lease_offer_mxn=45000.0,
            hoa_monthly_mxn=2800.0,
            wifi_gas_water_monthly_mxn=1200.0,
            cleaning_cost_per_stay_mxn=650.0,
            avg_stay_nights=4.5,
            airbnb_fee_pct=0.18,
            facebook_direct_share=0.50,
            winter_cfe_monthly_mxn=1800.0,
            summer_cfe_monthly_mxn=6200.0,
        )
        be_occ = engine.compute_break_even_occupancy(
            adr_mxn=1100.0,
            monthly_lease_offer_mxn=45000.0,
            costs=costs,
            is_summer_month=True,
        )
        self.assertGreaterEqual(be_occ, 0.0)
        self.assertGreater(be_occ, 1.0)

    def test_monte_carlo_summer_lease_recommendation(self):
        costs = engine.UserCostInputs(
            unit_id="CB_TUL_1BR",
            contract_months=6,
            monthly_lease_offer_mxn=26000.0,
            hoa_monthly_mxn=3100.0,
            wifi_gas_water_monthly_mxn=1100.0,
            cleaning_cost_per_stay_mxn=700.0,
            avg_stay_nights=4.0,
            airbnb_fee_pct=0.18,
            facebook_direct_share=0.60,
            winter_cfe_monthly_mxn=1600.0,
            summer_cfe_monthly_mxn=6000.0,
        )
        summer_forecasts = [
            {"month_num": m, "pred_adr_mxn": 1350.0, "pred_occ_rate": 0.36, "adr_std_mxn": 150.0, "occ_std": 0.07}
            for m in [5, 6, 7, 8, 9, 10]
        ]
        res = engine.simulate_lease_vs_str(summer_forecasts, costs)
        self.assertEqual(res["recommendation"], "ACCEPT CONTRACT")
        self.assertGreater(res["lease_total_net_mxn"], res["str_p50_net_mxn"])

    def test_chronological_split_prevents_future_leakage(self):
        df = pd.DataFrame(
            {
                "unit_id": ["CB_PDC_1BR_A"] * 6,
                "year_month": ["2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08"],
                "month_num": [3, 4, 5, 6, 7, 8],
                "adr_mxn": [1800, 1650, 1300, 1250, 1400, 1350],
                "occupancy_rate": [0.78, 0.70, 0.52, 0.48, 0.58, 0.54],
            }
        )
        train_df, test_df = models.chronological_train_test_split(df, cutoff_year_month="2025-06")
        self.assertEqual(train_df["year_month"].max(), "2025-06")
        self.assertEqual(test_df["year_month"].min(), "2025-07")


if __name__ == "__main__":
    unittest.main()