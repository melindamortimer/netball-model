from netball_model.features.elo import GlickoSystem


def test_initial_ratings():
    system = GlickoSystem()
    rating = system.get_rating("Queensland Firebirds")
    assert rating["rating"] == 1500.0
    assert rating["rd"] == 350.0


def test_update_after_match():
    system = GlickoSystem()
    system.update("Queensland Firebirds", "NSW Swifts", winner="away", margin=5)

    fb = system.get_rating("Queensland Firebirds")
    sw = system.get_rating("NSW Swifts")

    assert fb["rating"] < 1500.0
    assert sw["rating"] > 1500.0
    assert fb["rd"] < 350.0
    assert sw["rd"] < 350.0


def test_ratings_converge_over_multiple_matches():
    system = GlickoSystem()
    for _ in range(10):
        system.update("Team A", "Team B", winner="home", margin=10)

    a = system.get_rating("Team A")
    b = system.get_rating("Team B")

    assert a["rating"] > 1600
    assert b["rating"] < 1400


def test_separate_pools():
    system = GlickoSystem()
    system.update("Australia", "New Zealand", winner="home", margin=5, pool="international")
    system.update("Firebirds", "Swifts", winner="home", margin=5, pool="ssn")

    assert system.get_rating("Australia", pool="ssn")["rating"] == 1500.0
    assert system.get_rating("Australia", pool="international")["rating"] > 1500.0


def test_predicted_win_probability():
    system = GlickoSystem()
    for _ in range(5):
        system.update("Strong Team", "Weak Team", winner="home", margin=15)

    prob = system.predict_win_prob("Strong Team", "Weak Team")
    assert prob > 0.5
    assert prob < 1.0
