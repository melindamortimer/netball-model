from netball_model.features.contextual import ContextualFeatures


def test_rest_days():
    matches = [
        {"date": "2024-03-30", "home_team": "Firebirds", "away_team": "Swifts"},
        {"date": "2024-04-06", "home_team": "Firebirds", "away_team": "Vixens"},
    ]
    cf = ContextualFeatures(matches)
    rest = cf.rest_days("Firebirds", match_index=1)
    assert rest == 7


def test_rest_days_first_match():
    matches = [
        {"date": "2024-03-30", "home_team": "Firebirds", "away_team": "Swifts"},
    ]
    cf = ContextualFeatures(matches)
    rest = cf.rest_days("Firebirds", match_index=0)
    assert rest is None


def test_recent_form():
    matches = [
        {"home_team": "A", "away_team": "B", "home_score": 60, "away_score": 50},
        {"home_team": "C", "away_team": "A", "home_score": 45, "away_score": 55},
        {"home_team": "A", "away_team": "D", "home_score": 40, "away_score": 50},
        {"home_team": "E", "away_team": "A", "home_score": 48, "away_score": 52},
        {"home_team": "A", "away_team": "F", "home_score": 58, "away_score": 55},
        {"home_team": "A", "away_team": "G", "home_score": 60, "away_score": 50},
    ]
    cf = ContextualFeatures(matches)
    win_rate, avg_margin = cf.recent_form("A", match_index=5, window=5)
    # Matches 0-4 for team A: W(+10), W(+10), L(-10), W(+4), W(+3)
    assert win_rate == 4 / 5
    assert avg_margin == (10 + 10 - 10 + 4 + 3) / 5


def test_head_to_head():
    matches = [
        {"home_team": "A", "away_team": "B", "home_score": 60, "away_score": 50},
        {"home_team": "B", "away_team": "A", "home_score": 55, "away_score": 45},
        {"home_team": "A", "away_team": "B", "home_score": 70, "away_score": 60},
        {"home_team": "A", "away_team": "B", "home_score": 50, "away_score": 50},
    ]
    cf = ContextualFeatures(matches)
    h2h_win_rate = cf.head_to_head("A", "B", match_index=3)
    # Matches 0-2: A wins match 0 and 2, loses match 1 -> 2/3
    assert h2h_win_rate == 2 / 3


def test_travel_distance():
    cf = ContextualFeatures([])
    dist = cf.travel_distance("Brisbane", "Perth")
    assert dist > 3000  # roughly 3600km


def test_travel_distance_same_city():
    cf = ContextualFeatures([])
    dist = cf.travel_distance("Melbourne", "Melbourne")
    assert dist == 0.0


def test_is_home():
    cf = ContextualFeatures([])
    match = {"home_team": "Firebirds", "away_team": "Swifts"}
    assert cf.is_home("Firebirds", match) is True
    assert cf.is_home("Swifts", match) is False


def test_round_features():
    matches = [
        {"home_team": "A", "away_team": "B", "season": 2024, "round_num": 3,
         "date": "2024-04-01", "home_score": 60, "away_score": 55, "venue": ""},
    ]
    ctx = ContextualFeatures(matches)
    result = ctx.round_features(matches[0])
    assert result["round_number"] == 3
    assert abs(result["season_progress"] - 3 / 14) < 0.01
