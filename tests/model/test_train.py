import numpy as np
import pandas as pd

from netball_model.model.train import NetballModel
from netball_model.model.calibration import CalibrationModel


def _make_dummy_df(n=100):
    rng = np.random.default_rng(42)
    elo_diff = rng.normal(0, 100, n)
    noise = rng.normal(0, 8, n)
    margin = 0.05 * elo_diff + noise
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)],
        "home_team": ["A"] * n,
        "away_team": ["B"] * n,
        "elo_diff": elo_diff,
        "home_elo": 1500 + elo_diff / 2,
        "away_elo": 1500 - elo_diff / 2,
        "home_elo_rd": [200.0] * n,
        "away_elo_rd": [200.0] * n,
        "elo_win_prob": [0.5] * n,
        "home_rest_days": rng.integers(5, 10, n),
        "away_rest_days": rng.integers(5, 10, n),
        "rest_diff": [0] * n,
        "home_form_win_rate": rng.uniform(0.3, 0.7, n),
        "away_form_win_rate": rng.uniform(0.3, 0.7, n),
        "home_form_avg_margin": rng.normal(0, 5, n),
        "away_form_avg_margin": rng.normal(0, 5, n),
        "h2h_home_win_rate": [0.5] * n,
        "home_travel_km": [0.0] * n,
        "away_travel_km": rng.uniform(0, 3000, n),
        "travel_diff": rng.uniform(0, 3000, n),
        "margin": margin,
        "total_goals": rng.integers(90, 130, n).astype(float),
    })


def test_train_and_predict():
    df = _make_dummy_df(100)
    model = NetballModel()
    model.train(df)

    pred = model.predict(df.iloc[[0]])
    assert "predicted_margin" in pred.columns
    assert "predicted_total" in pred.columns
    assert len(pred) == 1


def test_feature_columns_excludes_targets():
    model = NetballModel()
    df = _make_dummy_df(50)
    model.train(df)
    assert "margin" not in model.feature_columns
    assert "total_goals" not in model.feature_columns
    assert "match_id" not in model.feature_columns
    assert "home_team" not in model.feature_columns


def test_calibration():
    residuals = np.random.normal(0, 10, 200)
    cal = CalibrationModel()
    cal.fit(residuals)

    # Predicted margin of +5 with std 10 should give >50% win prob
    prob = cal.win_probability(predicted_margin=5.0)
    assert prob > 0.5
    assert prob < 1.0

    # Predicted margin of 0 should give ~50%
    prob_even = cal.win_probability(predicted_margin=0.0)
    assert abs(prob_even - 0.5) < 0.05
