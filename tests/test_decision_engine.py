"""
Unit tests for the Rental Strategy Optimizer core logic (`src/rental_engine`).
Compatible with both `python -m unittest discover -s tests -v` and `pytest -v`.
"""
import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rental_engine import (  # noqa: E402
    DEFAULT_ASSUMPTIONS,
    UserCostInputs,
    compute_blended_ota_drag,
    compute_break_even_occupancy,
    compute_rolling_12m_avg_kwh,
    days_in_month,
    estimate_cfe_monthly_bill,
    evaluate_12m_dac_trajectory,
    is_dac_reclassified,
    parse_facebook_rental_post,
    simulate_lease_vs_str,
)
from src.rental_engine.data import load_training_dataframe  # noqa: E402
from src.rental_engine.models import (  # noqa: E402
    chronological_train_test_split,
    predict_unit_monthly_forecasts,
    train_and_evaluate_models,
)


class TestRentalStrategyOptimizer(unittest.TestCase):
    def test_cfe_dac_summer_threshold_and_iva(self):
        # Below 400 kWh in summer stays on subsidized tariff
        low_bill = estimate_cfe_monthly_bill(kwh_used=350.0, is_summer_month=True)
        # Above 400 kWh in summer triggers DAC rate (6.08 MXN/kWh * 1.16 IVA)
        dac_bill = estimate_cfe_monthly_bill(kwh_used=850.0, is_summer_month=True)
        self.assertLess(low_bill, 1000.0)
        self.assertAlmostEqual(dac_bill, round(850.0 * 6.08 * 1.16, 2), places=2)

    def test_cfe_dac_12m_rolling_average_and_winter_persistence(self):
        # 6 winter months at 250 kWh + 6 summer months at 600 kWh -> mean = 425 kWh > 400 (Tariff 1B DAC)
        history_into_dac = [250.0] * 6 + [600.0] * 6
        self.assertAlmostEqual(compute_rolling_12m_avg_kwh(history_into_dac), 425.0, places=2)
        self.assertTrue(is_dac_reclassified(history_into_dac, tariff="1B"))

        # Even in a winter month (is_summer_month=False) with 260 kWh usage, a meter whose
        # 12-month rolling average is > 400 kWh stays billed at the unsubsidized DAC rate:
        winter_dac_bill = estimate_cfe_monthly_bill(
            kwh_used=260.0,
            is_summer_month=False,
            trailing_12m_kwh=history_into_dac,
            tariff="1B",
        )
        self.assertAlmostEqual(winter_dac_bill, round(260.0 * 6.08 * 1.16, 2), places=2)

        # Once 12-month trailing average drops back to <= 400 kWh/mo, meter exits DAC:
        history_below_dac = [320.0] * 12
        self.assertFalse(is_dac_reclassified(history_below_dac, tariff="1B"))
        subsidized_bill = estimate_cfe_monthly_bill(
            kwh_used=260.0,
            is_summer_month=False,
            trailing_12m_kwh=history_below_dac,
            tariff="1B",
        )
        self.assertAlmostEqual(subsidized_bill, round(260.0 * 1.85 * 1.16, 2), places=2)

    def test_evaluate_12m_dac_trajectory_crosses_threshold(self):
        # Prior 11 months at 370 kWh; month 1 at 650 kWh leaves 12M mean = 393.33 (<=400, not DAC),
        # month 2 at 700 kWh pushes 12M mean to 420.83 (>400, enters DAC)
        traj = evaluate_12m_dac_trajectory(
            prior_11m_kwh=[370.0] * 11,
            window_monthly_kwh=[650.0, 700.0],
            tariff="1B",
        )
        self.assertFalse(traj[0]["in_dac"])
        self.assertTrue(traj[1]["in_dac"])

    def test_facebook_direct_channel_reduces_platform_fee(self):
        # 60% Facebook direct (0% fee) + 40% Airbnb (18% fee) -> 7.2% blended fee
        blended = compute_blended_ota_drag(airbnb_fee_pct=0.18, facebook_direct_share=0.60)
        self.assertAlmostEqual(blended, 0.072, places=4)

    def test_break_even_occupancy_clamped_to_unit_interval(self):
        costs = UserCostInputs(
            unit_id="UNIT_A_PLAYA",
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
        be_occ = compute_break_even_occupancy(
            adr_mxn=1100.0,
            monthly_lease_offer_mxn=45000.0,
            costs=costs,
            is_summer_month=True,
        )
        self.assertGreaterEqual(be_occ, 0.0)
        self.assertEqual(be_occ, 1.0)

    def test_exact_days_in_month_and_monte_carlo_summer_lease(self):
        self.assertEqual(days_in_month(2026, 2), 28)
        self.assertEqual(days_in_month(2026, 5), 31)

        costs = UserCostInputs(
            unit_id="UNIT_TULUM",
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
        res = simulate_lease_vs_str(summer_forecasts, costs, reference_year=2026)
        self.assertEqual(res["recommendation"], "ACCEPT CONTRACT")
        self.assertGreater(res["lease_total_net_mxn"], res["str_p50_net_mxn"])
        # Verify May is labeled with 31 days and June with 30 days
        months_col = res["monthly_df"]["Month"].tolist()
        self.assertIn("May (31d)", months_col[0])
        self.assertIn("Jun (30d)", months_col[1])

    def test_chronological_split_prevents_future_leakage(self):
        df = pd.DataFrame(
            {
                "unit_id": ["UNIT_A_PLAYA"] * 6,
                "year_month": ["2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08"],
                "month_num": [3, 4, 5, 6, 7, 8],
                "adr_mxn": [1800, 1650, 1300, 1250, 1400, 1350],
                "occupancy_rate": [0.78, 0.70, 0.52, 0.48, 0.58, 0.54],
            }
        )
        train_df, test_df = chronological_train_test_split(df, cutoff_year_month="2025-06")
        self.assertEqual(train_df["year_month"].max(), "2025-06")
        self.assertEqual(test_df["year_month"].min(), "2025-07")

    def test_models_actually_trained_and_produce_backtest_residuals(self):
        df = load_training_dataframe()
        bundle = train_and_evaluate_models(df, cutoff_year_month="2025-06")
        self.assertEqual(bundle.train_rows, 90)
        self.assertEqual(bundle.test_rows, 30)
        self.assertGreater(bundle.adr_residual_std_mxn, 10.0)
        self.assertGreater(bundle.occ_residual_std, 0.01)
        self.assertGreaterEqual(len(bundle.backtest_metrics_df), 4)

        forecasts = predict_unit_monthly_forecasts(bundle, "UNIT_TULUM", [5, 6, 7])
        self.assertEqual(len(forecasts), 3)
        self.assertEqual(forecasts[0]["adr_std_mxn"], bundle.adr_residual_std_mxn)
        self.assertEqual(forecasts[0]["occ_std"], bundle.occ_residual_std)

    def test_facebook_post_parser_extracts_fields(self):
        post = "Available May 1st! 2BR apartment in Aldea Zama. $25,000 MXN/mes + CFE electric."
        parsed = parse_facebook_rental_post(post)
        self.assertEqual(parsed["bedrooms"], 2)
        self.assertEqual(parsed["neighborhood"], "Aldea Zama")
        self.assertEqual(parsed["asking_monthly_mxn"], 25000.0)
        self.assertTrue(parsed["tenant_pays_cfe"])


if __name__ == "__main__":
    unittest.main()
