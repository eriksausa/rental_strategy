"""
Tests for the walk-forward forecasting pipeline, the elasticity estimator, the residual-block
bootstrap, and its integration with the Monte Carlo engine (all on SYNTHETIC data).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rental_engine import (  # noqa: E402
    UserCostInputs,
    elasticity,
    forecast_service,
    forecasting,
    simulate_lease_vs_str,
    synthetic_panel,
)

FEATURE_COLS = forecasting.FEATURES + ["um_adr", "um_occ"]


def _costs() -> UserCostInputs:
    return UserCostInputs(
        unit_id="UNIT_A_PLAYA", contract_months=3, monthly_lease_offer_mxn=26000.0, hoa_monthly_mxn=3100.0,
        wifi_gas_water_monthly_mxn=1100.0, cleaning_cost_per_stay_mxn=700.0, avg_stay_nights=4.0,
        airbnb_fee_pct=0.18, facebook_direct_share=0.60, winter_cfe_monthly_mxn=1600.0, summer_cfe_monthly_mxn=6000.0,
    )


def _forecasts(n=3):
    return [
        {"month_num": m, "pred_adr_mxn": 1800.0, "pred_occ_rate": 0.55, "adr_std_mxn": 250.0, "occ_std": 0.10}
        for m in range(1, n + 1)
    ]


class TestSyntheticPanel(unittest.TestCase):
    def test_deterministic_balanced_and_bounded(self):
        a = synthetic_panel.generate_panel(seed=3, n_months=24)
        b = synthetic_panel.generate_panel(seed=3, n_months=24)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), a["unit_id"].nunique() * 24)
        self.assertTrue(a["occupancy_rate"].between(0.0, 1.0).all())
        self.assertTrue((a["adr_mxn"] > 0).all())
        self.assertTrue(a.attrs["synthetic"])
        self.assertEqual(a.attrs["true_params"]["elasticity"], synthetic_panel.TrueParams().elasticity)


class TestNoLookAhead(unittest.TestCase):
    def test_features_and_predictions_ignore_data_after_the_origin(self):
        origin = 24
        panel = synthetic_panel.generate_panel(seed=5, n_months=36)
        tampered = panel.copy()
        future = tampered["t"] > origin
        tampered.loc[future, "adr_mxn"] = tampered.loc[future, "adr_mxn"] * 10.0
        tampered.loc[future, "occupancy_rate"] = 0.99

        p1, p2 = forecasting.build_pairs(panel), forecasting.build_pairs(tampered)
        r1 = p1[p1["origin"] == origin].reset_index(drop=True)
        r2 = p2[p2["origin"] == origin].reset_index(drop=True)
        pd.testing.assert_frame_equal(r1[FEATURE_COLS], r2[FEATURE_COLS])

        models = ("seasonal_naive", "unit_month_mean", "gradient_boosting")
        o1 = forecasting.walk_forward(p1[p1["y_adr"].notna()], [origin], models=models, n_estimators=20)
        o2 = forecasting.walk_forward(p2[p2["y_adr"].notna()], [origin], models=models, n_estimators=20)
        np.testing.assert_allclose(o1["pred_t"].to_numpy(), o2["pred_t"].to_numpy())
        # ...while the realised targets do differ, so the check is not vacuous.
        self.assertFalse(np.allclose(o1["actual_t"].to_numpy(), o2["actual_t"].to_numpy()))


class TestElasticity(unittest.TestCase):
    def test_fixed_effects_recover_truth_and_naive_is_biased(self):
        panel = synthetic_panel.generate_panel(seed=7, n_months=48)
        est = elasticity.estimate_elasticity(panel, n_boot=120, seed=1)
        truth = est["true_value"]
        lo, hi = est["headline_ci95_bootstrap"]
        self.assertLess(abs(est["headline_estimate"] - truth), 0.35)
        self.assertLessEqual(lo, truth)
        self.assertGreaterEqual(hi, truth)
        self.assertGreater(est["naive_pooled_ols"], 0.0)  # wrong sign: seasonality and shocks confound price


class TestResidualBlocksAndService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        panel = synthetic_panel.generate_panel(seed=11, n_months=30)
        cls.out = forecasting.run_pipeline(panel, n_estimators=15, first_test_origin=24)
        cls.art = forecast_service.ForecastArtifacts(
            forecasts=cls.out["forecasts"], residuals=cls.out["residuals"],
            meta={"last_observed_month": cls.out["result"]["last_observed_month"]},
        )

    def test_pipeline_outputs(self):
        f = self.out["forecasts"]
        self.assertEqual(set(f["h"]), set(range(1, 13)))
        self.assertEqual(len(f), f["unit_id"].nunique() * 12)
        self.assertIn(self.out["result"]["chosen_models"]["adr"], forecasting.ALL_MODELS)
        self.assertAlmostEqual(self.out["residuals"]["res_adr"].mean(), 0.0, places=6)  # centred

    def test_window_mapping(self):
        self.assertEqual(forecast_service.window_start_month("2025-12", [5, 6, 7, 8, 9, 10]), "2026-05")
        self.assertIsNone(forecast_service.window_start_month("2025-12", [11, 12, 1, 2, 3, 4]))

    def test_forecast_window_returns_blocks(self):
        last = self.art.meta["last_observed_month"]
        start = str(pd.Period(last, freq="M") + 2)
        unit = self.art.forecasts["unit_id"].iloc[0]
        monthly, blocks = forecast_service.forecast_window(self.art, unit, start, 3)
        self.assertEqual([m["h"] for m in monthly], [2, 3, 4])
        self.assertIsNotNone(blocks)
        self.assertEqual(blocks.shape[1:], (3, 2))
        self.assertGreater(blocks.shape[0], 10)
        with self.assertRaises(ValueError):
            forecast_service.forecast_window(self.art, unit, str(pd.Period(last, freq="M") + 11), 6)


class TestEngineIntegration(unittest.TestCase):
    def test_default_path_is_independent_normal(self):
        res = simulate_lease_vs_str(_forecasts(), _costs(), n_simulations=2000)
        self.assertEqual(res["uncertainty_method"], "independent_normal")

    def test_persistent_errors_widen_the_distribution(self):
        rng = np.random.default_rng(0)
        n = 4000
        z = rng.normal(size=(n, 1))
        persistent = np.stack([np.repeat(z * 0.15, 3, axis=1), np.repeat(z * 0.4, 3, axis=1)], axis=2)
        independent = np.stack([rng.normal(size=(n, 3)) * 0.15, rng.normal(size=(n, 3)) * 0.4], axis=2)
        a = simulate_lease_vs_str(_forecasts(), _costs(), n_simulations=n, residual_blocks=persistent)
        b = simulate_lease_vs_str(_forecasts(), _costs(), n_simulations=n, residual_blocks=independent)
        self.assertEqual(a["uncertainty_method"], "block_bootstrap")
        self.assertGreater(np.std(a["sim_str_totals"]), 1.3 * np.std(b["sim_str_totals"]))

    def test_bad_block_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_lease_vs_str(_forecasts(), _costs(), n_simulations=100, residual_blocks=np.zeros((5, 2, 2)))


if __name__ == "__main__":
    unittest.main()
