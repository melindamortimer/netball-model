from netball_model.data.squads import get_squad, get_all_squads, SQUADS_2026

POSITIONS = {"GS", "GA", "WA", "C", "WD", "GD", "GK"}
TEAMS_2026 = [
    "Adelaide Thunderbirds", "GIANTS Netball", "Melbourne Mavericks",
    "Melbourne Vixens", "NSW Swifts", "Queensland Firebirds",
    "Sunshine Coast Lightning", "West Coast Fever",
]


def test_all_teams_present():
    squads = get_all_squads(2026)
    assert set(squads.keys()) == set(TEAMS_2026)


def test_each_team_has_seven_positions():
    for team in TEAMS_2026:
        squad = get_squad(team, 2026)
        assert set(squad.keys()) == POSITIONS, f"{team} missing positions"


def test_no_duplicate_players_across_teams():
    all_players = []
    for team, squad in get_all_squads(2026).items():
        all_players.extend(squad.values())
    assert len(all_players) == len(set(all_players)), "Duplicate player found"


def test_get_squad_specific_team():
    squad = get_squad("Melbourne Vixens", 2026)
    assert squad["GS"] == "Sophie Garbin"
    assert squad["C"] == "Kate Moloney"


def test_unknown_team_raises():
    import pytest
    with pytest.raises(KeyError):
        get_squad("Nonexistent Team", 2026)
