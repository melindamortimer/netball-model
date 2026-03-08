from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from netball_model.data.database import Database


def _seed_db(tmp_path):
    """Create a DB with one match for testing."""
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.initialize()
    db.upsert_match({
        "match_id": "12438_01_01",
        "competition_id": 12438,
        "season": 2024,
        "round_num": 1,
        "game_num": 1,
        "date": "2024-03-30",
        "venue": "Test Arena",
        "home_team": "Adelaide Thunderbirds",
        "away_team": "NSW Swifts",
        "home_score": 55,
        "away_score": 60,
        "home_q1": 14, "home_q2": 13, "home_q3": 15, "home_q4": 13,
        "away_q1": 16, "away_q2": 14, "away_q3": 15, "away_q4": 15,
    })
    return db_path, db


def _mock_fetch_odds(events, **kwargs):
    """Simulate BetsApiClient.fetch_odds_for_events."""
    return [
        {**ev, "home_odds": 1.75, "away_odds": 2.10, "bookmaker": "bet365"}
        for ev in events
    ]


def _mock_fetch_odds_none(events, **kwargs):
    """Simulate no odds available."""
    return [
        {**ev, "home_odds": None, "away_odds": None, "bookmaker": None}
        for ev in events
    ]


def test_import_betsapi_odds(tmp_path):
    """Test import_betsapi_odds with mocked API client."""
    from netball_model.services import import_betsapi_odds

    db_path, db = _seed_db(tmp_path)

    test_events = [
        {"event_id": "10371196", "home_team": "Adelaide Thunderbirds", "away_team": "NSW Swifts", "date": "2024-03-30"},
    ]

    with patch("netball_model.data.betsapi.BetsApiClient") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.fetch_odds_for_events.side_effect = _mock_fetch_odds

        counts = import_betsapi_odds(db, "fake_token", test_events, season=2024)

    assert counts["matched"] == 1
    assert counts["total"] == 1

    odds = db.get_odds(source="betsapi")
    assert len(odds) == 1
    assert odds[0]["home_back_odds"] == 1.75
    assert odds[0]["away_back_odds"] == 2.10


def test_import_betsapi_odds_swapped(tmp_path):
    """Test swapped home/away is handled correctly."""
    from netball_model.services import import_betsapi_odds

    db_path, db = _seed_db(tmp_path)

    # BetsAPI lists teams in reverse order
    test_events = [
        {"event_id": "10371196", "home_team": "NSW Swifts", "away_team": "Adelaide Thunderbirds", "date": "2024-03-30"},
    ]

    with patch("netball_model.data.betsapi.BetsApiClient") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.fetch_odds_for_events.side_effect = _mock_fetch_odds

        counts = import_betsapi_odds(db, "fake_token", test_events, season=2024)

    assert counts["matched"] == 1

    odds = db.get_odds(source="betsapi")
    assert len(odds) == 1
    # Odds should be swapped to match DB orientation
    assert odds[0]["home_back_odds"] == 2.10  # Adelaide (DB home)
    assert odds[0]["away_back_odds"] == 1.75  # NSW Swifts (DB away)


def test_import_betsapi_odds_no_odds(tmp_path):
    """Test events with no odds are counted."""
    from netball_model.services import import_betsapi_odds

    db_path, db = _seed_db(tmp_path)

    test_events = [
        {"event_id": "10371196", "home_team": "Adelaide Thunderbirds", "away_team": "NSW Swifts"},
    ]

    with patch("netball_model.data.betsapi.BetsApiClient") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.fetch_odds_for_events.side_effect = _mock_fetch_odds_none

        counts = import_betsapi_odds(db, "fake_token", test_events, season=2024)

    assert counts["no_odds"] == 1
    assert counts["matched"] == 0
    assert len(db.get_odds(source="betsapi")) == 0


def test_import_betsapi_odds_unmatched(tmp_path):
    """Test events not in DB are counted as unmatched."""
    from netball_model.services import import_betsapi_odds

    db_path, db = _seed_db(tmp_path)

    test_events = [
        {"event_id": "99999", "home_team": "Melbourne Vixens", "away_team": "West Coast Fever"},
    ]

    with patch("netball_model.data.betsapi.BetsApiClient") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.fetch_odds_for_events.side_effect = _mock_fetch_odds

        counts = import_betsapi_odds(db, "fake_token", test_events, season=2024)

    assert counts["unmatched"] == 1
    assert counts["matched"] == 0


def test_import_betsapi_odds_no_matches_in_db(tmp_path):
    """Test raises ValueError when no matches in DB."""
    from netball_model.services import import_betsapi_odds
    import pytest

    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.initialize()

    with pytest.raises(ValueError, match="No matches"):
        import_betsapi_odds(db, "fake_token", [], season=2024)
