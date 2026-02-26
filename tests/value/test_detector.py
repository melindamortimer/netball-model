from netball_model.value.detector import ValueDetector


def test_detect_value_home():
    detector = ValueDetector(min_edge=0.05)

    result = detector.evaluate(
        home_team="Firebirds",
        away_team="Swifts",
        model_win_prob=0.65,
        betfair_home_back=1.80,  # implied prob = 55.6%
    )

    assert result["model_prob"] == 0.65
    assert abs(result["implied_prob"] - 1 / 1.80) < 0.01
    assert result["edge"] > 0.05
    assert result["is_value"] is True
    assert result["bet_side"] == "home"


def test_no_value():
    detector = ValueDetector(min_edge=0.05)

    result = detector.evaluate(
        home_team="Firebirds",
        away_team="Swifts",
        model_win_prob=0.55,
        betfair_home_back=1.75,  # implied prob = 57.1%
    )

    assert result["is_value"] is False


def test_value_on_away():
    detector = ValueDetector(min_edge=0.05)

    result = detector.evaluate(
        home_team="Firebirds",
        away_team="Swifts",
        model_win_prob=0.35,
        betfair_home_back=1.60,  # implied home prob = 62.5%
        betfair_away_back=2.50,  # implied away prob = 40%
    )

    # Model says away win prob = 0.65, implied away = 40%
    assert result["bet_side"] == "away"
    assert result["is_value"] is True
