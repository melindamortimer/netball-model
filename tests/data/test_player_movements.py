import pytest
from netball_model.data.database import Database
from netball_model.data.player_movements import (
    get_roster_continuity,
    get_player_movements,
    get_team_continuity_all,
)


@pytest.fixture
def db_with_two_seasons(tmp_path):
    """DB with players across two seasons, some moving teams."""
    db = Database(tmp_path / "test.db")
    db.initialize()

    # Season 2024: Team A has players 1-7, Team B has players 8-14
    for i, pos in enumerate(["GS", "GA", "WA", "C", "WD", "GD", "GK"]):
        db.insert_player_stats({
            "match_id": "m2024_01", "player_id": i + 1,
            "player_name": f"Player{i+1}", "team": "Team A",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })
        db.insert_player_stats({
            "match_id": "m2024_01", "player_id": i + 8,
            "player_name": f"Player{i+8}", "team": "Team B",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })

    # Create match records
    for mid, season in [("m2024_01", 2024), ("m2025_01", 2025)]:
        db.upsert_match({
            "match_id": mid, "competition_id": 1, "season": season,
            "round_num": 1, "game_num": 1, "date": f"{season}-03-15",
            "venue": "Test", "home_team": "Team A", "away_team": "Team B",
            "home_score": 60, "away_score": 55,
            "home_q1": 15, "home_q2": 15, "home_q3": 15, "home_q4": 15,
            "away_q1": 14, "away_q2": 14, "away_q3": 14, "away_q4": 13,
        })

    # Season 2025: Players 1-5 stay on A, players 6-7 move to B
    for i, pos in enumerate(["GS", "GA", "WA", "C", "WD"]):
        db.insert_player_stats({
            "match_id": "m2025_01", "player_id": i + 1,
            "player_name": f"Player{i+1}", "team": "Team A",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })
    # Player 6 and 7 moved to Team B
    for i, pos in enumerate(["GD", "GK"]):
        db.insert_player_stats({
            "match_id": "m2025_01", "player_id": i + 6,
            "player_name": f"Player{i+6}", "team": "Team B",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })
    # New players 15-16 on Team A
    for i, pos in enumerate(["GD", "GK"]):
        db.insert_player_stats({
            "match_id": "m2025_01", "player_id": i + 15,
            "player_name": f"Player{i+15}", "team": "Team A",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })

    return db


def test_roster_continuity_full_retention(db_with_two_seasons):
    # Team A: 5 out of 7 starters stayed (players 1-5), 2 are new (15-16)
    continuity = get_roster_continuity("Team A", 2025, db_with_two_seasons)
    assert abs(continuity - 5 / 7) < 0.01


def test_continuity_all_teams(db_with_two_seasons):
    result = get_team_continuity_all(2025, db_with_two_seasons)
    assert "Team A" in result
    assert 0.0 <= result["Team A"] <= 1.0
