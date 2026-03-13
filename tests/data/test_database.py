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


def _make_match(match_id, season=2024, home="Team A", away="Team B", home_score=55, away_score=60):
    return {
        "match_id": match_id,
        "competition_id": 99999,
        "season": season,
        "round_num": 1,
        "game_num": 1,
        "date": "2024-04-01",
        "venue": "Test Arena",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_q1": 14, "home_q2": 13, "home_q3": 14, "home_q4": 14,
        "away_q1": 15, "away_q2": 15, "away_q3": 15, "away_q4": 15,
    }


def _make_player_stat(match_id, player_id, name="Test Player"):
    return {
        "match_id": match_id,
        "player_id": player_id,
        "player_name": name,
        "team": "Team A",
        "position": "GS",
        "goals": 10, "attempts": 12, "assists": 1, "rebounds": 2,
        "feeds": 3, "turnovers": 1, "gains": 0, "intercepts": 0,
        "deflections": 0, "penalties": 1, "centre_pass_receives": 0,
        "net_points": 25.0,
    }


def test_connection_context_manager(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    with db.connection() as conn:
        conn.execute(
            "INSERT INTO matches (match_id, competition_id, season, round_num, game_num, home_team, away_team) "
            "VALUES ('ctx_01', 1, 2024, 1, 1, 'A', 'B')"
        )

    rows = db.get_matches(season=2024)
    assert len(rows) == 1
    assert rows[0]["match_id"] == "ctx_01"


def test_upsert_matches_batch(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    matches = [_make_match(f"batch_{i:02d}") for i in range(10)]
    db.upsert_matches(matches)

    rows = db.get_matches(season=2024)
    assert len(rows) == 10

    # Upsert again — should not duplicate
    db.upsert_matches(matches)
    rows = db.get_matches(season=2024)
    assert len(rows) == 10


def test_insert_player_stats_batch(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    stats = [_make_player_stat("m_01", pid, f"Player {pid}") for pid in range(20)]
    db.insert_player_stats_batch(stats)

    rows = db.get_player_stats(match_id="m_01")
    assert len(rows) == 20

    # Replace — should not duplicate
    db.insert_player_stats_batch(stats)
    rows = db.get_player_stats(match_id="m_01")
    assert len(rows) == 20


def test_get_player_history(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    # Insert 3 matches on different dates
    for i, date in enumerate(["2024-03-01", "2024-03-08", "2024-03-15"]):
        db.upsert_match({
            "match_id": f"m_{i}", "competition_id": 1, "season": 2024,
            "round_num": i + 1, "game_num": 1, "date": date,
            "venue": "Test", "home_team": "A", "away_team": "B",
            "home_score": 55, "away_score": 50,
            "home_q1": 14, "home_q2": 14, "home_q3": 14, "home_q4": 13,
            "away_q1": 13, "away_q2": 13, "away_q3": 12, "away_q4": 12,
        })
        db.insert_player_stats({
            "match_id": f"m_{i}", "player_id": 100, "player_name": "",
            "team": "A", "position": "GS",
            "goals": 30 + i, "attempts": 35, "assists": 2, "rebounds": 3,
            "feeds": 4, "turnovers": 2, "gains": 0, "intercepts": 0,
            "deflections": 0, "penalties": 1, "centre_pass_receives": 0,
            "net_points": 0.0,
        })

    # Before the 3rd match, should get 2 rows (matches 0 and 1)
    history = db.get_player_history(player_id=100, before_date="2024-03-15", limit=5)
    assert len(history) == 2
    # Most recent first
    assert history[0]["goals"] == 31  # match 1 (2024-03-08)
    assert history[1]["goals"] == 30  # match 0 (2024-03-01)

    # With limit=1, should get only 1 row
    history = db.get_player_history(player_id=100, before_date="2024-03-15", limit=1)
    assert len(history) == 1
    assert history[0]["goals"] == 31


def test_get_starters_for_match(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    db.upsert_match({
        "match_id": "m_01", "competition_id": 1, "season": 2024,
        "round_num": 1, "game_num": 1, "date": "2024-03-01",
        "venue": "Test", "home_team": "A", "away_team": "B",
        "home_score": 55, "away_score": 50,
        "home_q1": 14, "home_q2": 14, "home_q3": 14, "home_q4": 13,
        "away_q1": 13, "away_q2": 13, "away_q3": 12, "away_q4": 12,
    })

    positions = ["GS", "GA", "WA", "C", "WD", "GD", "GK"]
    for team_idx, team in enumerate(["A", "B"]):
        for pos_idx, pos in enumerate(positions):
            db.insert_player_stats({
                "match_id": "m_01", "player_id": team_idx * 100 + pos_idx,
                "player_name": "", "team": team, "position": pos,
                "goals": 10, "attempts": 12, "assists": 1, "rebounds": 2,
                "feeds": 3, "turnovers": 1, "gains": 0, "intercepts": 0,
                "deflections": 0, "penalties": 1, "centre_pass_receives": 0,
                "net_points": 0.0,
            })
        # Add a substitute (position = "-")
        db.insert_player_stats({
            "match_id": "m_01", "player_id": team_idx * 100 + 50,
            "player_name": "", "team": team, "position": "-",
            "goals": 0, "attempts": 0, "assists": 0, "rebounds": 0,
            "feeds": 0, "turnovers": 0, "gains": 0, "intercepts": 0,
            "deflections": 0, "penalties": 0, "centre_pass_receives": 0,
            "net_points": 0.0,
        })

    starters = db.get_starters_for_match("m_01")
    assert len(starters) == 14  # 7 per team, no subs
    positions_found = {s["position"] for s in starters}
    assert "-" not in positions_found
    assert positions_found == set(positions)


def test_upsert_and_get_player_elo(tmp_db):
    db = Database(tmp_db)
    db.initialize()
    db.upsert_player_elo({
        "player_id": 1, "player_name": "Test Player", "position": "GS",
        "pool": "ssn", "match_id": "m1", "rating": 1600.0, "rd": 80.0, "vol": 0.06
    })
    result = db.get_latest_player_elo(1, "GS")
    assert result is not None
    assert result["rating"] == 1600.0


def test_get_all_player_elos(tmp_db):
    db = Database(tmp_db)
    db.initialize()
    for pid, rating in [(1, 1600), (2, 1400)]:
        db.upsert_player_elo({
            "player_id": pid, "player_name": f"P{pid}", "position": "GS",
            "pool": "ssn", "match_id": "m1", "rating": rating, "rd": 80, "vol": 0.06
        })
    results = db.get_all_player_elos()
    assert len(results) == 2
