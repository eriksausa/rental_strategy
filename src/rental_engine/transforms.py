"""
Shared mathematical transforms for occupancy rates and log-odds scales.

Why these live in one place:
  The synthetic panel generator, walk-forward forecasting pipeline, price-elasticity
  estimator, and block-bootstrap Monte Carlo simulator all operate on occupancy in
  log-odds space (`logit(p)`) so additive shocks stay strictly inside (0, 1).
  Centralizing the transforms keeps clipping bounds and continuity smoothing identical
  across every stage of the pipeline.
"""
from __future__ import annotations

import numpy as np

# Small epsilon prevents log(0) or division by zero when transforming boundary rates.
_PROB_EPS = 1e-6


def smooth_occupancy(occ: np.ndarray | float, days: np.ndarray | float) -> np.ndarray | float:
    """Apply Laplace continuity smoothing so 0% and 100% months stay finite in log-odds space.

    Equivalent to adding 0.5 occupied nights and 0.5 vacant nights to each month's count:
        p_smooth = (occ * days + 0.5) / (days + 1.0)
    """
    return (occ * days + 0.5) / (days + 1.0)


def logit(prob: np.ndarray | float) -> np.ndarray | float:
    """Convert probabilities in (0, 1) to log-odds `log(p / (1 - p))` with safe clipping."""
    clipped = np.clip(prob, _PROB_EPS, 1.0 - _PROB_EPS)
    return np.log(clipped / (1.0 - clipped))


def inv_logit(log_odds: np.ndarray | float) -> np.ndarray | float:
    """Convert log-odds back to probabilities in (0, 1) via the logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-log_odds))
