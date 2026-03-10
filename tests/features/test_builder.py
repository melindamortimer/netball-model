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


def test_build_row_with_player_stats():
    """FeatureBuilder should include matchup features when player_stats is provided."""
    matches = [
        {
            "match_id": "m_0", "date": "2024-03-23T12:00:00+11:00", "round_num": 1,
            "home_team": "A", "away_team": "B",
            "home_score": 60, "away_score": 55, "venue": "Brisbane Entertainment Centre",
        },
        {
            "match_id": "m_1", "date": "2024-03-30T12:00:00+11:00", "round_num": 2,
            "home_team": "B", "away_team": "A",
            "home_score": 58, "away_score": 52, "venue": "Ken Rosewall Arena",
        },
    ]

    positions = ["GS", "GA", "WA", "C", "WD", "GD", "GK"]
    player_stats = {}
    for mid in ["m_0", "m_1"]:
        starters = []
        for team_idx, team in enumerate(["A", "B"]):
            for pos_idx, pos in enumerate(positions):
                starters.append({
                    "player_id": team_idx * 100 + pos_idx, "team": team,
                    "position": pos, "goals": 30 if pos in ("GS", "GA") else 0,
                    "attempts": 35 if pos in ("GS", "GA") else 0,
                    "assists": 5, "rebounds": 3, "feeds": 10, "turnovers": 2,
                    "gains": 3, "intercepts": 2, "deflections": 1,
                    "penalties": 2, "centre_pass_receives": 4,
                })
        player_stats[mid] = starters

    builder = FeatureBuilder(matches, player_stats=player_stats)
    row = builder.build_row(match_index=1)

    # Should have matchup features alongside existing features
    assert "home_elo" in row
    assert "gs_vs_gk_shooting_pressure" in row
    assert "ga_vs_gd_shooting_pressure" in row
    assert "c_vs_c_distribution_battle" in row
    assert "attack_matchup" in row


def test_build_row_without_player_stats():
    """FeatureBuilder should work as before when player_stats is None."""
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
    assert "gs_vs_gk_shooting_pressure" not in row
