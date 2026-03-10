import pytest

from netball_model.features.matchups import MatchupFeatures


def _make_profile(position, goals=30, attempts=35, assists=5, feeds=10,
                  turnovers=2, gains=3, intercepts=2, deflections=1,
                  penalties=2, rebounds=3, centre_pass_receives=4,
                  shooting_pct=None, clean_steal_rate=None,
                  delivery_efficiency=None):
    """Build a fake player profile dict."""
    p = {
        "goals": goals, "attempts": attempts, "assists": assists,
        "feeds": feeds, "turnovers": turnovers, "gains": gains,
        "intercepts": intercepts, "deflections": deflections,
        "penalties": penalties, "rebounds": rebounds,
        "centre_pass_receives": centre_pass_receives,
        "matches_used": 5,
    }
    if position in ("GS", "GA"):
        p["shooting_pct"] = shooting_pct if shooting_pct is not None else (
            goals / attempts if attempts > 0 else 0.0
        )
    if position in ("GD", "GK", "WD"):
        p["clean_steal_rate"] = clean_steal_rate if clean_steal_rate is not None else (
            intercepts / gains if gains > 0 else 0.0
        )
    if position in ("WA", "C", "GA"):
        p["delivery_efficiency"] = delivery_efficiency if delivery_efficiency is not None else (
            feeds / turnovers if turnovers > 0 else 0.0
        )
    return p


# --- GS vs GK ---

def test_gs_vs_gk_shooting_pressure():
    mf = MatchupFeatures()
    home = {"GS": _make_profile("GS", goals=35, attempts=40, shooting_pct=0.875)}
    away = {"GK": _make_profile("GK", deflections=5)}

    features = mf.compute_features(home, away)

    # shooting_pressure = GS shooting_pct - (GK deflections / GS attempts)
    # = 0.875 - (5 / 40) = 0.875 - 0.125 = 0.75
    assert features["gs_vs_gk_shooting_pressure"] == pytest.approx(0.75)


def test_gs_vs_gk_rebounding_battle():
    mf = MatchupFeatures()
    home = {"GS": _make_profile("GS", rebounds=5)}
    away = {"GK": _make_profile("GK", rebounds=4)}

    features = mf.compute_features(home, away)

    assert features["gs_vs_gk_rebounding_battle"] == pytest.approx(1.0)


def test_gs_vs_gk_shot_volume_vs_disruption():
    mf = MatchupFeatures()
    home = {"GS": _make_profile("GS", attempts=45)}
    away = {"GK": _make_profile("GK", intercepts=3, deflections=6)}

    features = mf.compute_features(home, away)

    # = GS attempts - (GK intercepts + GK deflections) = 45 - (3 + 6) = 36
    assert features["gs_vs_gk_shot_volume_vs_disruption"] == pytest.approx(36.0)


# --- GA vs GD ---

def test_ga_vs_gd_shooting_pressure():
    mf = MatchupFeatures()
    home = {"GA": _make_profile("GA", goals=18, attempts=22, shooting_pct=18/22)}
    away = {"GD": _make_profile("GD", deflections=4)}

    features = mf.compute_features(home, away)

    # = GA shooting_pct - (GD deflections / GA attempts)
    # = 18/22 - 4/22 = 14/22
    assert features["ga_vs_gd_shooting_pressure"] == pytest.approx(14 / 22)


def test_ga_vs_gd_feed_vs_intercept():
    mf = MatchupFeatures()
    home = {"GA": _make_profile("GA", feeds=20)}
    away = {"GD": _make_profile("GD", intercepts=3)}

    features = mf.compute_features(home, away)

    assert features["ga_vs_gd_feed_vs_intercept"] == pytest.approx(17.0)


def test_ga_vs_gd_creativity_vs_discipline():
    mf = MatchupFeatures()
    home = {"GA": _make_profile("GA", assists=12)}
    away = {"GD": _make_profile("GD", gains=4)}

    features = mf.compute_features(home, away)

    assert features["ga_vs_gd_creativity_vs_discipline"] == pytest.approx(8.0)


# --- WA vs WD ---

def test_wa_vs_wd_delivery_vs_disruption():
    mf = MatchupFeatures()
    home = {"WA": _make_profile("WA", feeds=30)}
    away = {"WD": _make_profile("WD", intercepts=2, deflections=3)}

    features = mf.compute_features(home, away)

    # = WA feeds - (WD intercepts + WD deflections) = 30 - 5 = 25
    assert features["wa_vs_wd_delivery_vs_disruption"] == pytest.approx(25.0)


def test_wa_vs_wd_turnover_vulnerability():
    mf = MatchupFeatures()
    home = {"WA": _make_profile("WA", turnovers=4)}
    away = {"WD": _make_profile("WD", gains=2)}

    features = mf.compute_features(home, away)

    assert features["wa_vs_wd_turnover_vulnerability"] == pytest.approx(2.0)


def test_wa_vs_wd_supply_line():
    mf = MatchupFeatures()
    home = {"WA": _make_profile("WA", centre_pass_receives=15)}
    away = {"WD": _make_profile("WD", deflections=3)}

    features = mf.compute_features(home, away)

    assert features["wa_vs_wd_supply_line"] == pytest.approx(12.0)


# --- C vs C ---

def test_c_vs_c_distribution_battle():
    mf = MatchupFeatures()
    home = {"C": _make_profile("C", feeds=25, assists=14)}
    away = {"C": _make_profile("C", feeds=20, assists=10)}

    features = mf.compute_features(home, away)

    # = (25 + 14) - (20 + 10) = 39 - 30 = 9
    assert features["c_vs_c_distribution_battle"] == pytest.approx(9.0)


def test_c_vs_c_disruption_battle():
    mf = MatchupFeatures()
    home = {"C": _make_profile("C", gains=2, intercepts=1)}
    away = {"C": _make_profile("C", gains=1, intercepts=0)}

    features = mf.compute_features(home, away)

    # = (2 + 1) - (1 + 0) = 2
    assert features["c_vs_c_disruption_battle"] == pytest.approx(2.0)


def test_c_vs_c_turnover_differential():
    mf = MatchupFeatures()
    home = {"C": _make_profile("C", turnovers=2)}
    away = {"C": _make_profile("C", turnovers=5)}

    features = mf.compute_features(home, away)

    # = away turnovers - home turnovers = 5 - 2 = 3 (positive = home advantage)
    assert features["c_vs_c_turnover_differential"] == pytest.approx(3.0)


# --- WD vs WA ---

def test_wd_vs_wa_pressure_effectiveness():
    mf = MatchupFeatures()
    home = {"WD": _make_profile("WD", gains=3, intercepts=2)}
    away = {"WA": _make_profile("WA", turnovers=4)}

    features = mf.compute_features(home, away)

    # = (WD gains + WD intercepts) - WA turnovers = 5 - 4 = 1
    assert features["wd_vs_wa_pressure_effectiveness"] == pytest.approx(1.0)


def test_wd_vs_wa_aerial_battle():
    mf = MatchupFeatures()
    home = {"WD": _make_profile("WD", deflections=4)}
    away = {"WA": _make_profile("WA", feeds=28)}

    features = mf.compute_features(home, away)

    # = WD deflections - WA feeds = 4 - 28 = -24
    assert features["wd_vs_wa_aerial_battle"] == pytest.approx(-24.0)


# --- Composite scores ---

def test_composite_attack_matchup():
    mf = MatchupFeatures()
    home = {
        "GS": _make_profile("GS", goals=35, attempts=40, shooting_pct=0.875),
        "GA": _make_profile("GA", goals=15, attempts=20, shooting_pct=0.75),
        "WA": _make_profile("WA", feeds=30),
    }
    away = {
        "GK": _make_profile("GK", deflections=4),
        "GD": _make_profile("GD", deflections=2),
        "WD": _make_profile("WD", intercepts=1, deflections=2),
    }

    features = mf.compute_features(home, away)

    gs_sp = 0.875 - (4 / 40)   # 0.775
    ga_sp = 0.75 - (2 / 20)    # 0.65
    wa_dd = 30 - (1 + 2)       # 27.0
    expected = (gs_sp + ga_sp + wa_dd) / 3
    assert features["attack_matchup"] == pytest.approx(expected)


def test_composite_defence_matchup():
    mf = MatchupFeatures()
    home = {"WD": _make_profile("WD", gains=3, intercepts=2, deflections=4)}
    away = {"WA": _make_profile("WA", turnovers=4, feeds=28)}

    features = mf.compute_features(home, away)

    pressure = (3 + 2) - 4      # 1.0
    aerial = 4 - 28             # -24.0
    expected = (pressure + aerial) / 2
    assert features["defence_matchup"] == pytest.approx(expected)


def test_composite_midcourt_matchup():
    mf = MatchupFeatures()
    home = {
        "C": _make_profile("C", feeds=25, assists=14, gains=2, intercepts=1, turnovers=2),
        "WA": _make_profile("WA", feeds=30, turnovers=3, centre_pass_receives=15),
    }
    away = {
        "C": _make_profile("C", feeds=20, assists=10, gains=1, intercepts=0, turnovers=4),
        "WD": _make_profile("WD", intercepts=1, deflections=2, gains=2),
    }

    features = mf.compute_features(home, away)

    c_dist = (25 + 14) - (20 + 10)       # 9.0
    c_disr = (2 + 1) - (1 + 0)           # 2.0
    c_to = 4 - 2                          # 2.0
    wa_dd = 30 - (1 + 2)                  # 27.0
    wa_tv = 3 - 2                         # 1.0
    wa_sl = 15 - 2                        # 13.0
    expected = (c_dist + c_disr + c_to + wa_dd + wa_tv + wa_sl) / 6
    assert features["midcourt_matchup"] == pytest.approx(expected)


# --- Edge cases ---

def test_all_pairs_produce_features():
    mf = MatchupFeatures()
    positions = ["GS", "GA", "WA", "C", "WD", "GD", "GK"]
    home = {p: _make_profile(p) for p in positions}
    away = {p: _make_profile(p) for p in positions}

    features = mf.compute_features(home, away)

    expected_keys = {
        "gs_vs_gk_shooting_pressure", "gs_vs_gk_rebounding_battle",
        "gs_vs_gk_shot_volume_vs_disruption",
        "ga_vs_gd_shooting_pressure", "ga_vs_gd_feed_vs_intercept",
        "ga_vs_gd_creativity_vs_discipline",
        "wa_vs_wd_delivery_vs_disruption", "wa_vs_wd_turnover_vulnerability",
        "wa_vs_wd_supply_line",
        "c_vs_c_distribution_battle", "c_vs_c_disruption_battle",
        "c_vs_c_turnover_differential",
        "wd_vs_wa_pressure_effectiveness", "wd_vs_wa_aerial_battle",
        "attack_matchup", "defence_matchup", "midcourt_matchup",
    }
    assert set(features.keys()) == expected_keys


def test_missing_position_returns_zeros():
    """Missing positions should produce 0.0 for all their features."""
    mf = MatchupFeatures()
    features = mf.compute_features({}, {})

    assert features["gs_vs_gk_shooting_pressure"] == 0.0
    assert features["ga_vs_gd_feed_vs_intercept"] == 0.0
    assert features["c_vs_c_distribution_battle"] == 0.0
    assert features["wd_vs_wa_pressure_effectiveness"] == 0.0
    assert features["attack_matchup"] == 0.0


def test_missing_one_side_defaults_to_zero():
    """Only home GS present, no away GK — away stats default to 0."""
    mf = MatchupFeatures()
    home = {"GS": _make_profile("GS", goals=35, attempts=40, shooting_pct=0.875, rebounds=5)}
    features = mf.compute_features(home, {})

    # No GK: deflections=0, so deflection_rate=0
    # shooting_pressure = 0.875 - 0 = 0.875
    assert features["gs_vs_gk_shooting_pressure"] == pytest.approx(0.875)
    assert features["gs_vs_gk_rebounding_battle"] == pytest.approx(5.0)
    assert features["gs_vs_gk_shot_volume_vs_disruption"] == pytest.approx(40.0)


def test_zero_attempts_avoids_division_by_zero():
    """GS with 0 attempts should not cause ZeroDivisionError."""
    mf = MatchupFeatures()
    home = {"GS": _make_profile("GS", goals=0, attempts=0, shooting_pct=0.0)}
    away = {"GK": _make_profile("GK", deflections=5)}

    features = mf.compute_features(home, away)

    # deflection_rate = 0 (guarded), shooting_pressure = 0 - 0 = 0
    assert features["gs_vs_gk_shooting_pressure"] == pytest.approx(0.0)
