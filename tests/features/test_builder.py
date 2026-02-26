import pandas as pd

from netball_model.features.builder import FeatureBuilder


def test_build_feature_row():
    matches = [
        {
            "match_id": "1", "date": "2024-03-23", "round_num": 1,
            "home_team": "Firebirds", "away_team": "Swifts",
            "home_score": 60, "away_score": 55, "venue": "Brisbane Entertainment Centre",
        },
        {
            "match_id": "2", "date": "2024-03-30", "round_num": 2,
            "home_team": "Swifts", "away_team": "Firebirds",
            "home_score": 58, "away_score": 52, "venue": "Ken Rosewall Arena",
        },
    ]
    builder = FeatureBuilder(matches)
    row = builder.build_row(match_index=1)

    assert "home_elo" in row
    assert "away_elo" in row
    assert "home_rest_days" in row
    assert "away_rest_days" in row
    assert "home_form_win_rate" in row
    assert "home_form_avg_margin" in row
    assert "h2h_home_win_rate" in row
    assert "elo_diff" in row
    assert "margin" in row
    assert row["margin"] == 58 - 52


def test_build_matrix():
    matches = [
        {
            "match_id": str(i), "date": f"2024-03-{20+i:02d}", "round_num": i,
            "home_team": "A", "away_team": "B",
            "home_score": 55 + i, "away_score": 50, "venue": "Brisbane Entertainment Centre",
        }
        for i in range(1, 6)
    ]
    builder = FeatureBuilder(matches)
    df = builder.build_matrix(start_index=1)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "margin" in df.columns
    assert "home_elo" in df.columns
