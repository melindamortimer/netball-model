"""Shared team name normalisation for Super Netball."""
from __future__ import annotations

# Keys are lowercased display names, values are canonical Champion Data names.
TEAM_NAME_MAP: dict[str, str] = {
    "adelaide thunderbirds": "Adelaide Thunderbirds",
    "thunderbirds": "Adelaide Thunderbirds",
    "collingwood magpies": "Collingwood Magpies",
    "magpies": "Collingwood Magpies",
    "giants netball": "GIANTS Netball",
    "gws giants": "GIANTS Netball",
    "giant": "GIANTS Netball",
    "giants": "GIANTS Netball",
    "melbourne vixens": "Melbourne Vixens",
    "vixens": "Melbourne Vixens",
    "melbourne mavericks": "Melbourne Mavericks",
    "mavericks": "Melbourne Mavericks",
    "nsw swifts": "NSW Swifts",
    "swifts": "NSW Swifts",
    "queensland firebirds": "Queensland Firebirds",
    "firebirds": "Queensland Firebirds",
    "sunshine coast lightning": "Sunshine Coast Lightning",
    "lightning": "Sunshine Coast Lightning",
    "west coast fever": "West Coast Fever",
    "fever": "West Coast Fever",
}


def normalise_team(name: str) -> str | None:
    """Map a display team name to the canonical Champion Data name."""
    key = name.strip().lower()
    if key in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[key]
    # Try matching by suffix (e.g. "SC Lightning" -> "Sunshine Coast Lightning")
    for mapped_key, canonical in TEAM_NAME_MAP.items():
        if key.endswith(mapped_key):
            return canonical
    return None
