from netball_model.data.team_names import normalise_team, TEAM_NAME_MAP


def test_normalise_exact_match():
    assert normalise_team("Adelaide Thunderbirds") == "Adelaide Thunderbirds"
    assert normalise_team("GIANTS Netball") == "GIANTS Netball"


def test_normalise_case_insensitive():
    assert normalise_team("adelaide thunderbirds") == "Adelaide Thunderbirds"
    assert normalise_team("MELBOURNE VIXENS") == "Melbourne Vixens"


def test_normalise_abbreviations():
    assert normalise_team("Thunderbirds") == "Adelaide Thunderbirds"
    assert normalise_team("Swifts") == "NSW Swifts"
    assert normalise_team("Fever") == "West Coast Fever"


def test_normalise_suffix_match():
    assert normalise_team("SC Lightning") == "Sunshine Coast Lightning"


def test_normalise_unknown_returns_none():
    assert normalise_team("Unknown Team") is None


def test_map_has_all_teams():
    canonical = set(TEAM_NAME_MAP.values())
    assert len(canonical) == 9  # 9 SSN teams (including Mavericks)
