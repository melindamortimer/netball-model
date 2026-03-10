import json
from pathlib import Path

import httpx
import pytest
import respx

from netball_model.data.champion_data import ChampionDataClient, COMPETITION_IDS

FIXTURES = Path(__file__).parent / "fixtures"


def test_competition_ids_has_all_seasons():
    for year in range(2017, 2026):
        assert year in COMPETITION_IDS, f"Missing competition ID for {year}"
        assert "regular" in COMPETITION_IDS[year]
        assert "finals" in COMPETITION_IDS[year]


@respx.mock
@pytest.mark.asyncio
async def test_fetch_match():
    sample = json.loads((FIXTURES / "sample_match.json").read_text())
    comp_id = 12438
    url = f"https://mc.championdata.com/data/{comp_id}/{comp_id}0101.json"
    respx.get(url).respond(json=sample)

    client = ChampionDataClient()
    match_data = await client.fetch_match(comp_id, round_num=1, game_num=1)
    assert match_data is not None
    assert "matchStats" in match_data
    await client.close()


@respx.mock
@pytest.mark.asyncio
async def test_fetch_match_404():
    comp_id = 12438
    url = f"https://mc.championdata.com/data/{comp_id}/{comp_id}9901.json"
    respx.get(url).respond(status_code=404)

    client = ChampionDataClient()
    result = await client.fetch_match(comp_id, round_num=99, game_num=1)
    assert result is None
    await client.close()


def test_parse_match():
    sample = json.loads((FIXTURES / "sample_match.json").read_text())

    client = ChampionDataClient()
    match, players = client.parse_match(
        sample, competition_id=10393, season=2018, round_num=1, game_num=1
    )

    assert match["home_team"] == "Queensland Firebirds"
    assert match["away_team"] == "NSW Swifts"
    assert match["home_score"] == 55  # 14+13+15+13
    assert match["away_score"] == 60  # 16+14+15+15
    assert match["home_q1"] == 14
    assert match["away_q4"] == 15
    assert match["venue"] == "Brisbane Entertainment Centre"

    assert len(players) == 2
    assert players[0]["player_name"] == "Gretel Bueta"
    assert players[0]["goals"] == 30
    assert players[0]["net_points"] == 78.5
    assert players[0]["position"] == "GS"
    assert players[1]["player_name"] == "Sophie Wallace"
    assert players[1]["team"] == "NSW Swifts"

    # turnovers should come from generalPlayTurnovers, not turnovers
    assert players[0]["turnovers"] == 2  # generalPlayTurnovers value
    assert players[1]["turnovers"] == 3
