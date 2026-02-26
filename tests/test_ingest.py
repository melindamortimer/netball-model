from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from netball_model.cli import main


@patch("netball_model.cli.ChampionDataClient")
@patch("netball_model.cli.Database")
def test_ingest_command(mock_db_cls, mock_client_cls, tmp_path):
    mock_client = AsyncMock()
    mock_client.fetch_season.return_value = [
        (
            {
                "match_id": "12438_01_01",
                "competition_id": 12438,
                "season": 2024,
                "round_num": 1,
                "game_num": 1,
                "date": "2024-03-30",
                "venue": "Test Arena",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 55,
                "away_score": 60,
                "home_q1": 14, "home_q2": 13, "home_q3": 15, "home_q4": 13,
                "away_q1": 16, "away_q2": 14, "away_q3": 15, "away_q4": 15,
            },
            [],
        )
    ]
    mock_client.close = AsyncMock()
    mock_client_cls.return_value = mock_client

    mock_db = mock_db_cls.return_value

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "--season", "2024", "--db", str(tmp_path / "test.db")])
    assert result.exit_code == 0
    mock_db.initialize.assert_called_once()
    mock_db.upsert_match.assert_called_once()
