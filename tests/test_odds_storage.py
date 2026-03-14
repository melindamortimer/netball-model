import json
import datetime
from pathlib import Path


def test_save_odds_json_creates_new_file(tmp_path):
    """First save creates the file with correct structure."""
    from app import save_odds_json

    matches = [
        {
            "date": "2026-03-15",
            "home_team": "West Coast Fever",
            "away_team": "Sunshine Coast Lightning",
            "home_odds": 2.05,
            "away_odds": 1.75,
            "handicap_line": 1.5,
            "handicap_home_odds": 1.80,
            "handicap_away_odds": 2.00,
            "total_line": 125.5,
            "over_odds": 1.87,
            "under_odds": 1.87,
        }
    ]

    out_path = save_odds_json(matches, output_dir=tmp_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["source"] == "bet365_screenshot"
    assert data["session_date"] == datetime.date.today().isoformat()
    assert len(data["matches"]) == 1
    assert data["matches"][0]["home_team"] == "West Coast Fever"


def test_save_odds_json_appends_and_deduplicates(tmp_path):
    """Second save appends new matches and deduplicates existing ones."""
    from app import save_odds_json

    match_a = {
        "date": "2026-03-15",
        "home_team": "West Coast Fever",
        "away_team": "Sunshine Coast Lightning",
        "home_odds": 2.05,
        "away_odds": 1.75,
    }
    match_b = {
        "date": "2026-03-15",
        "home_team": "Melbourne Vixens",
        "away_team": "NSW Swifts",
        "home_odds": 1.50,
        "away_odds": 2.50,
    }

    # First save
    save_odds_json([match_a], output_dir=tmp_path)

    # Second save with match_a updated + new match_b
    match_a_updated = dict(match_a)
    match_a_updated["home_odds"] = 2.10  # odds changed
    save_odds_json([match_a_updated, match_b], output_dir=tmp_path)

    data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
    assert len(data["matches"]) == 2  # deduped, not 3
    # The updated odds should be present (replaced)
    fever_match = [m for m in data["matches"] if m["home_team"] == "West Coast Fever"][0]
    assert fever_match["home_odds"] == 2.10
