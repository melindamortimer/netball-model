from __future__ import annotations


def determine_winner(home_score: int, away_score: int) -> str:
    """Return 'home', 'away', or 'draw' based on scores."""
    if home_score > away_score:
        return "home"
    elif away_score > home_score:
        return "away"
    return "draw"
