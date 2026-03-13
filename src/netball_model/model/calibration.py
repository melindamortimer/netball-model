from __future__ import annotations

import numpy as np
from scipy.stats import norm


class CalibrationModel:
    def __init__(self):
        self.residual_std: float = 10.0  # default
        self.total_residual_std: float = 10.0  # default

    def fit(self, residuals: np.ndarray, total_residuals: np.ndarray | None = None):
        self.residual_std = float(np.std(residuals))
        if total_residuals is not None:
            self.total_residual_std = float(np.std(total_residuals))

    def win_probability(self, predicted_margin: float) -> float:
        """P(actual_margin > 0) given predicted_margin."""
        return float(norm.cdf(predicted_margin / self.residual_std))
