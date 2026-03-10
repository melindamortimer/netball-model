import numpy as np
import pytest

from netball_model.data.database import Database
from netball_model.services import train_model, backtest_season


def _seed_db(db, n=50, season=2024):
    """Seed a database with synthetic matches."""
    teams = [
        "Queensland Firebirds", "NSW Swifts", "Melbourne Vixens",
        "West Coast Fever", "Adelaide Thunderbirds", "GIANTS Netball",
        "Collingwood Magpies", "Sunshine Coast Lightning",
    ]
    rng = np.random.default_rng(42)
    matches = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        home_score = int(rng.integers(45, 70))
        away_score = int(rng.integers(45, 70))
        matches.append({
            "match_id": f"svc_{season}_{i:03d}",
            "competition_id": 99999,
            "season": season,
            "round_num": (i // 4) + 1,
            "game_num": (i % 4) + 1,
            "date": f"{season}-{((i // 4) % 10) + 3:02d}-{(i % 28) + 1:02d}",
            "venue": "Brisbane Entertainment Centre",
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "home_q1": home_score // 4,
            "home_q2": home_score // 4,
            "home_q3": home_score // 4,
            "home_q4": home_score - 3 * (home_score // 4),
            "away_q1": away_score // 4,
            "away_q2": away_score // 4,
            "away_q3": away_score // 4,
            "away_q4": away_score - 3 * (away_score // 4),
        })
    db.upsert_matches(matches)
    return matches


def test_train_model(tmp_db):
    db = Database(tmp_db)
    db.initialize()
    _seed_db(db, n=50)

    output = str(tmp_db.parent / "model.pkl")
    model, mae = train_model(db, output)

    assert model is not None
    assert mae >= 0


def test_train_model_insufficient_data(tmp_db):
    db = Database(tmp_db)
    db.initialize()
    _seed_db(db, n=5)

    with pytest.raises(ValueError, match="Need at least 20"):
        train_model(db, str(tmp_db.parent / "model.pkl"))


def test_backtest_season(tmp_db):
    db = Database(tmp_db)
    db.initialize()
    _seed_db(db, n=50, season=2023)
    _seed_db(db, n=20, season=2024)

    results = backtest_season(db, (2023, 2023), 2024)

    assert results["test_season"] == 2024
    assert results["matches"] == 20
    assert 0.0 <= results["accuracy"] <= 1.0
    assert results["mae"] >= 0


def test_backtest_insufficient_data(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    with pytest.raises(ValueError, match="Insufficient data"):
        backtest_season(db, (2023, 2023), 2024)


def test_backtest_season_with_player_stats(tmp_db):
    """Backtest should work when player stats are available."""
    db = Database(tmp_db)
    db.initialize()
    matches_2023 = _seed_db(db, n=50, season=2023)
    matches_2024 = _seed_db(db, n=20, season=2024)

    # Add player stats for each match
    positions = ["GS", "GA", "WA", "C", "WD", "GD", "GK"]
    for m in matches_2023 + matches_2024:
        for team_idx, team in enumerate([m["home_team"], m["away_team"]]):
            for pos_idx, pos in enumerate(positions):
                db.insert_player_stats({
                    "match_id": m["match_id"],
                    "player_id": hash(team) % 10000 + pos_idx,
                    "player_name": "", "team": team, "position": pos,
                    "goals": 30 if pos in ("GS", "GA") else 0,
                    "attempts": 35 if pos in ("GS", "GA") else 0,
                    "assists": 5, "rebounds": 3, "feeds": 10, "turnovers": 2,
                    "gains": 3, "intercepts": 2, "deflections": 1,
                    "penalties": 2, "centre_pass_receives": 4,
                    "net_points": 0.0,
                })

    result = backtest_season(db, (2023, 2023), 2024, use_player_stats=True)

    assert result["matches"] > 0
    assert 0 <= result["accuracy"] <= 1
    assert result["mae"] >= 0
