from __future__ import annotations

import numpy as np
from scipy.stats import norm


class CalibrationModel:
    def __init__(self):
        self.residual_std: float = 10.0  # default

    def fit(self, residuals: np.ndarray):
        self.residual_std = float(np.std(residuals))

    def win_probability(self, predicted_margin: float) -> float:
        """P(actual_margin > 0) given predicted_margin."""
        return float(norm.cdf(predicted_margin / self.residual_std))
