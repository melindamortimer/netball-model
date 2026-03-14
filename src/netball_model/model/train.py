from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from netball_model.model.calibration import CalibrationModel

NON_FEATURE_COLUMNS = {
    "match_id", "home_team", "away_team", "margin", "total_goals", "_sample_weight",
}


class NetballModel:
    def __init__(self, alpha: float = 1.0):
        self.margin_model = Ridge(alpha=alpha)
        self.total_model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.calibration = CalibrationModel()
        self.feature_columns: list[str] = []

    def train(self, df: pd.DataFrame):
        self.feature_columns = [
            c for c in df.columns if c not in NON_FEATURE_COLUMNS
        ]

        X = df[self.feature_columns].values.astype(float)
        y_margin = df["margin"].values.astype(float)
        y_total = df["total_goals"].values.astype(float)

        # Extract sample weights if present
        sample_weight = None
        if "_sample_weight" in df.columns:
            sample_weight = df["_sample_weight"].values.astype(float)

        X_scaled = self.scaler.fit_transform(X)

        self.margin_model.fit(X_scaled, y_margin, sample_weight=sample_weight)
        self.total_model.fit(X_scaled, y_total, sample_weight=sample_weight)

        # Calibrate on training residuals
        margin_preds = self.margin_model.predict(X_scaled)
        margin_residuals = y_margin - margin_preds
        total_preds = self.total_model.predict(X_scaled)
        total_residuals = y_total - total_preds
        self.calibration.fit(margin_residuals, total_residuals=total_residuals)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill missing feature columns with 0.0 (matches without player stats)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_columns].values.astype(float)
        X_scaled = self.scaler.transform(X)

        margins = self.margin_model.predict(X_scaled)
        totals = self.total_model.predict(X_scaled)
        win_probs = [self.calibration.win_probability(m) for m in margins]

        result = df[["match_id", "home_team", "away_team"]].copy()
        result["predicted_margin"] = np.round(margins, 1)
        result["predicted_total"] = np.round(totals, 1)
        result["win_probability"] = np.round(win_probs, 4)
        return result

    @property
    def residual_std(self) -> float:
        return self.calibration.residual_std

    @property
    def total_residual_std(self) -> float:
        return self.calibration.total_residual_std

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> NetballModel:
        with open(path, "rb") as f:
            return pickle.load(f)
