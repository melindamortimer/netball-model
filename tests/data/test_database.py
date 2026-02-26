import sqlite3

from netball_model.data.database import Database


def test_creates_tables(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    conn = sqlite3.connect(tmp_db)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "matches" in tables
    assert "player_stats" in tables
    assert "odds_history" in tables
    assert "elo_ratings" in tables


def test_upsert_match(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    match = {
        "match_id": "10393_01_01",
        "competition_id": 10393,
        "season": 2018,
        "round_num": 1,
        "game_num": 1,
        "date": "2018-04-28",
        "venue": "Brisbane Entertainment Centre",
        "home_team": "Queensland Firebirds",
        "away_team": "NSW Swifts",
        "home_score": 55,
        "away_score": 60,
        "home_q1": 14,
        "home_q2": 13,
        "home_q3": 15,
        "home_q4": 13,
        "away_q1": 16,
        "away_q2": 14,
        "away_q3": 15,
        "away_q4": 15,
    }

    db.upsert_match(match)
    rows = db.get_matches(season=2018)
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Queensland Firebirds"

    # Upsert again — should not duplicate
    db.upsert_match(match)
    rows = db.get_matches(season=2018)
    assert len(rows) == 1


def test_insert_player_stats(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    stats = {
        "match_id": "10393_01_01",
        "player_id": 12345,
        "player_name": "Gretel Bueta",
        "team": "Queensland Firebirds",
        "position": "GS",
        "goals": 30,
        "attempts": 33,
        "assists": 2,
        "rebounds": 4,
        "feeds": 5,
        "turnovers": 3,
        "gains": 0,
        "intercepts": 0,
        "deflections": 0,
        "penalties": 2,
        "centre_pass_receives": 0,
        "net_points": 78.5,
    }

    db.insert_player_stats(stats)
    rows = db.get_player_stats(match_id="10393_01_01")
    assert len(rows) == 1
    assert rows[0]["player_name"] == "Gretel Bueta"
