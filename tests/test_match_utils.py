from netball_model.match_utils import determine_winner


def test_home_win():
    assert determine_winner(65, 55) == "home"


def test_away_win():
    assert determine_winner(50, 60) == "away"


def test_draw():
    assert determine_winner(55, 55) == "draw"


def test_single_goal_margin():
    assert determine_winner(56, 55) == "home"
    assert determine_winner(55, 56) == "away"
