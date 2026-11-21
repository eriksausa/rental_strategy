"""
Generate the synthetic panel, run the walk-forward backtest, estimate price elasticity,
and write reproducible artifacts and a Markdown report.

Usage:
    python scripts/run_forecast_pipeline.py            # ~1 minute on a laptop
    python scripts/run_forecast_pipeline.py --quick    # smaller, faster smoke test

Outputs (relative to `--out-dir`, defaulting to the repo root):
    artifacts/forecasts.csv         Point forecasts, horizons 1..12 past `last_observed_month`
    artifacts/oof_residuals.csv     Centered OOF residuals of the chosen models (block-bootstrap input)
    reports/forecast_metrics.json   Backtest metrics, chosen models, elasticity, metadata
    reports/forecast_report.md      Human-readable Markdown summary report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rental_engine import elasticity, forecasting, reporting, synthetic_panel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the walk-forward forecasting & elasticity pipeline.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the synthetic panel.")
    parser.add_argument("--months", type=int, default=48, help="Number of monthly periods to simulate.")
    parser.add_argument("--n-estimators", type=int, default=150, help="Trees for RF / Gradient Boosting.")
    parser.add_argument("--quick", action="store_true", help="Fewer trees and test origins for fast smoke tests.")
    parser.add_argument("--out-dir", type=Path, default=ROOT, help="Directory receiving artifacts/ and reports/.")
    args = parser.parse_args()

    n_estimators = 40 if args.quick else args.n_estimators
    t0 = time.time()

    panel = synthetic_panel.generate_panel(seed=args.seed, n_months=args.months)
    first_origin = (args.months - 6) if args.quick else None
    out = forecasting.run_pipeline(
        panel,
        out_dir=args.out_dir,
        seed=0,
        n_estimators=n_estimators,
        first_test_origin=first_origin,
    )
    result = out["result"]

    el = elasticity.estimate_elasticity(panel, n_boot=200 if args.quick else 300, seed=0)
    result["elasticity"] = el
    result["runtime_seconds"] = round(time.time() - t0, 1)

    reports_dir = Path(args.out_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "forecast_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    (reports_dir / "forecast_report.md").write_text(reporting.build_report(result, el), encoding="utf-8")

    print(f"Done in {result['runtime_seconds']}s. Wrote artifacts/ and reports/ under {args.out_dir}.")
    print(f"Chosen models: {result['chosen_models']}")
    print(f"Elasticity (headline): {el['headline_estimate']} vs true {el['true_value']}")


if __name__ == "__main__":
    main()
