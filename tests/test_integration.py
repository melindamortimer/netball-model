"""End-to-end test using mock data: ingest -> train -> predict."""
import pandas as pd

from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel
from netball_model.value.detector import ValueDetector


def test_end_to_end(tmp_db, seed_matches):
    # 1. Setup DB + seed data
    db = Database(tmp_db)
    db.initialize()
    matches = seed_matches(db, n=50)

    # 2. Build features
    builder = FeatureBuilder(matches)
    df = builder.build_matrix(start_index=1)
    assert len(df) == 49

    # 3. Train model
    model = NetballModel()
    model.train(df)

    # 4. Predict on last match
    last_row = builder.build_row(len(matches) - 1)
    pred = model.predict(pd.DataFrame([last_row]))
    assert "predicted_margin" in pred.columns
    assert "win_probability" in pred.columns

    # 5. Value detection
    detector = ValueDetector(min_edge=0.05)
    prediction = {
        "margin": float(pred["predicted_margin"].iloc[0]),
        "total_goals": float(pred["predicted_total"].iloc[0]),
        "win_prob": float(pred["win_probability"].iloc[0]),
        "residual_std": model.calibration.residual_std,
        "total_residual_std": model.calibration.total_residual_std,
    }
    odds_dict = {"home_odds": 1.80}
    result = detector.evaluate(prediction, odds_dict)
    assert isinstance(result, list)
    assert len(result) > 0
    assert "market" in result[0]
    assert "edge" in result[0]

    # 6. Save/load model roundtrip
    model_path = tmp_db.parent / "model.pkl"
    model.save(model_path)
    loaded = NetballModel.load(model_path)
    pred2 = loaded.predict(pd.DataFrame([last_row]))
    assert float(pred2["predicted_margin"].iloc[0]) == float(pred["predicted_margin"].iloc[0])
