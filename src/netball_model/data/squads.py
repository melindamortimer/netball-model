"""2026 SSN squad data and lookup functions."""
from __future__ import annotations

SQUADS_2026 = {
    "Adelaide Thunderbirds": {
        "GS": "Elmere van der Berg",
        "GA": "Georgie Horjus",
        "WA": "Kayla Graham",
        "C": "Kate Heffernan",
        "WD": "Latanya Wilson",
        "GD": "Matilda Garrett",
        "GK": "Shamera Sterling-Humphrey",
    },
    "GIANTS Netball": {
        "GS": "Lucy Austin",
        "GA": "Sophie Dwyer",
        "WA": "Whitney Souness",
        "C": "Hope White",
        "WD": "Amy Sligar",
        "GD": "Erin O'Brien",
        "GK": "Jane Watson",
    },
    "Melbourne Mavericks": {
        "GS": "Shimona Nelson",
        "GA": "Reilley Batcheldor",
        "WA": "Sacha McDonald",
        "C": "Jamie-Lee Price",
        "WD": "Amy Parmenter",
        "GD": "Kim Brown",
        "GK": "Tara Hinchliffe",
    },
    "Melbourne Vixens": {
        "GS": "Sophie Garbin",
        "GA": "Kiera Austin",
        "WA": "Hannah Mundy",
        "C": "Kate Moloney",
        "WD": "Kate Eddy",
        "GD": "Jo Weston",
        "GK": "Rudi Ellis",
    },
    "NSW Swifts": {
        "GS": "Grace Nweke",
        "GA": "Helen Housby",
        "WA": "Gina Crampton",
        "C": "Maddy Proud",
        "WD": "Maddy Turner",
        "GD": "Teigan O'Shannassy",
        "GK": "Sarah Klau",
    },
    "Queensland Firebirds": {
        "GS": "Mary Cholhok",
        "GA": "Te Paea Selby-Rickit",
        "WA": "Macy Gardner",
        "C": "Maddy Gordon",
        "WD": "Lara Dunkley",
        "GD": "Ruby Bakewell-Doran",
        "GK": "Kelly Jackson",
    },
    "Sunshine Coast Lightning": {
        "GS": "Donnell Wallam",
        "GA": "Gabby Sinclair",
        "WA": "Leesa Mi Mi",
        "C": "Liz Watson",
        "WD": "Mahalia Cassidy",
        "GD": "Karin Burger",
        "GK": "Ash Ervin",
    },
    "West Coast Fever": {
        "GS": "Romelda Aiken-George",
        "GA": "Sasha Glasgow",
        "WA": "Alice Teague-Neeld",
        "C": "Jordan Cransberg",
        "WD": "Jess Anstiss",
        "GD": "Fran Williams",
        "GK": "Kadie-Ann Dehaney",
    },
}

_SQUADS_BY_SEASON = {
    2026: SQUADS_2026,
}


def get_squad(team: str, season: int = 2026) -> dict[str, str]:
    """Return {position: player_name} for a team in a season."""
    squads = _SQUADS_BY_SEASON.get(season)
    if squads is None:
        raise ValueError(f"No squad data for season {season}")
    return squads[team]  # KeyError if team not found


def get_all_squads(season: int = 2026) -> dict[str, dict[str, str]]:
    """Return all team squads for a season."""
    squads = _SQUADS_BY_SEASON.get(season)
    if squads is None:
        raise ValueError(f"No squad data for season {season}")
    return dict(squads)
