"""Tests for get_odds_for_match DB method."""
import tempfile
from pathlib import Path

from netball_model.data.database import Database


def test_get_odds_for_match_returns_none_when_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        db.initialize()
        assert db.get_odds_for_match("nonexistent") is None


def test_get_odds_for_match_returns_latest():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        db.initialize()

        # Insert two odds records for the same match
        db.upsert_odds({
            "match_id": "m1",
            "source": "betsapi",
            "home_back_odds": 1.50,
            "home_lay_odds": None,
            "away_back_odds": 2.50,
            "away_lay_odds": None,
            "home_volume": 0,
            "away_volume": 0,
            "timestamp": "2026-01-01",
        })
        db.upsert_odds_extended({
            "match_id": "m1",
            "source": "bet365",
            "home_back_odds": 1.38,
            "home_lay_odds": None,
            "away_back_odds": 3.00,
            "away_lay_odds": None,
            "home_volume": 0,
            "away_volume": 0,
            "timestamp": "2026-01-02",
            "handicap_home_odds": 1.85,
            "handicap_line": -4.5,
            "handicap_away_odds": 1.95,
            "total_line": 125.5,
            "over_odds": 1.87,
            "under_odds": 1.87,
        })

        result = db.get_odds_for_match("m1")
        assert result is not None
        # Should return the latest (bet365) record
        assert result["source"] == "bet365"
        assert result["home_back_odds"] == 1.38
        assert result["handicap_line"] == -4.5
        assert result["total_line"] == 125.5
