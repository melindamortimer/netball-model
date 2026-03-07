import numpy as np

from netball_model.model.train import NetballModel
from netball_model.model.calibration import CalibrationModel


def test_train_and_predict(dummy_feature_df):
    df = dummy_feature_df(100)
    model = NetballModel()
    model.train(df)

    pred = model.predict(df.iloc[[0]])
    assert "predicted_margin" in pred.columns
    assert "predicted_total" in pred.columns
    assert len(pred) == 1


def test_feature_columns_excludes_targets(dummy_feature_df):
    model = NetballModel()
    df = dummy_feature_df(50)
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
