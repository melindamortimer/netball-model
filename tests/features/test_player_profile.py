from netball_model.features.player_profile import PlayerProfiler

# Positions where each derived ratio applies
SHOOTER_POSITIONS = {"GS", "GA"}
DEFENDER_POSITIONS = {"GD", "GK", "WD"}
FEEDER_POSITIONS = {"WA", "C", "GA"}


def _make_stat_rows(n=5, goals=30, attempts=35, feeds=10, turnovers=2,
                    gains=3, intercepts=2, position="GS"):
    """Create n stat rows for testing."""
    return [
        {
            "player_id": 100, "position": position, "team": "A",
            "goals": goals, "attempts": attempts, "assists": 5,
            "rebounds": 3, "feeds": feeds, "turnovers": turnovers,
            "gains": gains, "intercepts": intercepts, "deflections": 1,
            "penalties": 2, "centre_pass_receives": 4,
        }
        for _ in range(n)
    ]


def test_compute_profile_basic():
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=5, goals=30, attempts=35)
    profile = profiler.compute_profile(rows, position="GS")

    assert profile["goals"] == 30.0
    assert profile["attempts"] == 35.0
    assert profile["assists"] == 5.0
    assert profile["matches_used"] == 5


def test_compute_profile_shooting_pct():
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=3, goals=30, attempts=40, position="GS")
    profile = profiler.compute_profile(rows, position="GS")

    assert abs(profile["shooting_pct"] - 0.75) < 0.001


def test_compute_profile_clean_steal_rate():
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=3, gains=4, intercepts=2, position="GD")
    profile = profiler.compute_profile(rows, position="GD")

    assert abs(profile["clean_steal_rate"] - 0.5) < 0.001


def test_compute_profile_delivery_efficiency():
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=3, feeds=20, turnovers=4, position="WA")
    profile = profiler.compute_profile(rows, position="WA")

    assert abs(profile["delivery_efficiency"] - 5.0) < 0.001


def test_compute_profile_zero_divisor():
    """Derived ratios should be 0 when denominator is 0."""
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=3, goals=0, attempts=0, position="GS")
    profile = profiler.compute_profile(rows, position="GS")

    assert profile["shooting_pct"] == 0.0


def test_compute_profile_empty():
    profiler = PlayerProfiler()
    profile = profiler.compute_profile([], position="GS")

    assert profile is None
