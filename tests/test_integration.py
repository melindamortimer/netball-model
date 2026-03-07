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
    result = detector.evaluate(
        home_team=matches[-1]["home_team"],
        away_team=matches[-1]["away_team"],
        model_win_prob=float(pred["win_probability"].iloc[0]),
        betfair_home_back=1.80,
    )
    assert "is_value" in result
    assert isinstance(result["is_value"], bool)

    # 6. Save/load model roundtrip
    model_path = tmp_db.parent / "model.pkl"
    model.save(model_path)
    loaded = NetballModel.load(model_path)
    pred2 = loaded.predict(pd.DataFrame([last_row]))
    assert float(pred2["predicted_margin"].iloc[0]) == float(pred["predicted_margin"].iloc[0])
