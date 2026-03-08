from unittest.mock import MagicMock, patch

from netball_model.data.betsapi import (
    BetsApiClient,
    normalise_team,
    _extract_best_closing_odds,
)


# ------------------------------------------------------------------
# normalise_team
# ------------------------------------------------------------------


def test_normalise_team_exact():
    assert normalise_team("Adelaide Thunderbirds") == "Adelaide Thunderbirds"
    assert normalise_team("nsw swifts") == "NSW Swifts"
    assert normalise_team("GIANTS Netball") == "GIANTS Netball"


def test_normalise_team_abbreviations():
    assert normalise_team("Thunderbirds") == "Adelaide Thunderbirds"
    assert normalise_team("Fever") == "West Coast Fever"
    assert normalise_team("Lightning") == "Sunshine Coast Lightning"


def test_normalise_team_unknown():
    assert normalise_team("Unknown Team XYZ") is None


# ------------------------------------------------------------------
# _extract_best_closing_odds
# ------------------------------------------------------------------


def test_extract_best_closing_odds_empty():
    assert _extract_best_closing_odds([]) is None
    assert _extract_best_closing_odds(None) is None


def test_extract_best_closing_odds_picks_latest():
    market = [
        {"home_od": "2.10", "away_od": "1.80", "add_time": "1000", "bookmaker_id": "bk1"},
        {"home_od": "2.00", "away_od": "1.85", "add_time": "2000", "bookmaker_id": "bk2"},
    ]
    result = _extract_best_closing_odds(market)
    assert result["home_odds"] == 2.00
    assert result["away_odds"] == 1.85


def test_extract_best_closing_odds_skips_invalid():
    market = [
        {"home_od": "0.5", "away_od": "1.80", "add_time": "1000"},  # home <= 1.0
        {"home_od": None, "away_od": "1.80", "add_time": "2000"},   # None
        {"home_od": "2.10", "away_od": "1.90", "add_time": "3000", "bookmaker_id": "bk1"},
    ]
    result = _extract_best_closing_odds(market)
    assert result["home_odds"] == 2.10
    assert result["away_odds"] == 1.90


def test_extract_best_closing_odds_skips_live_entries():
    """Entries with a score (ss) are in-play and should be ignored."""
    market = [
        {"home_od": "1.80", "away_od": "2.00", "add_time": "1000"},                      # pre-match
        {"home_od": "1.50", "away_od": "2.50", "add_time": "2000", "ss": "14-12"},        # live
        {"home_od": "1.01", "away_od": "15.0", "add_time": "3000", "ss": "50-40"},        # live
    ]
    result = _extract_best_closing_odds(market)
    assert result["home_odds"] == 1.80
    assert result["away_odds"] == 2.00


# ------------------------------------------------------------------
# BetsApiClient
# ------------------------------------------------------------------


@patch("netball_model.data.betsapi.httpx.Client")
def test_client_fetch_event_odds(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": 1,
        "results": {
            "odds": {
                "147_1": [
                    {"home_od": "1.75", "away_od": "2.10", "add_time": "1000", "bookmaker_id": "bet365"},
                ]
            }
        },
    }
    mock_client.get.return_value = mock_response

    client = BetsApiClient.__new__(BetsApiClient)
    client._client = mock_client

    odds = client.fetch_event_odds("10371196")
    assert odds is not None
    assert odds["home_odds"] == 1.75
    assert odds["away_odds"] == 2.10


@patch("netball_model.data.betsapi.httpx.Client")
def test_client_fetch_odds_for_events(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": 1,
        "results": {
            "odds": {
                "147_1": [
                    {"home_od": "1.75", "away_od": "2.10", "add_time": "1000", "bookmaker_id": "bet365"},
                ]
            }
        },
    }
    mock_client.get.return_value = mock_response

    client = BetsApiClient.__new__(BetsApiClient)
    client._client = mock_client

    events = [
        {"event_id": "10371196", "home_team": "Adelaide Thunderbirds", "away_team": "NSW Swifts"},
    ]
    results = client.fetch_odds_for_events(events, delay=0)
    assert len(results) == 1
    assert results[0]["home_team"] == "Adelaide Thunderbirds"
    assert results[0]["away_team"] == "NSW Swifts"
    assert results[0]["home_odds"] == 1.75
    assert results[0]["away_odds"] == 2.10


@patch("netball_model.data.betsapi.httpx.Client")
def test_client_fetch_event_odds_no_data(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": 1, "results": {"odds": {}}}
    mock_client.get.return_value = mock_response

    client = BetsApiClient.__new__(BetsApiClient)
    client._client = mock_client

    assert client.fetch_event_odds("99999") is None
