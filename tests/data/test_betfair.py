import json
import tarfile
import io

import pytest

from netball_model.data.betfair import BetfairParser


@pytest.fixture
def sample_betfair_market():
    return [
        {
            "op": "mcm",
            "clk": "1234",
            "pt": 1711800000000,
            "mc": [
                {
                    "id": "1.234567890",
                    "marketDefinition": {
                        "eventName": "Queensland Firebirds v NSW Swifts",
                        "marketType": "MATCH_ODDS",
                        "openDate": "2024-03-30T05:00:00.000Z",
                        "runners": [
                            {"id": 12345, "name": "Queensland Firebirds", "status": "ACTIVE"},
                            {"id": 67890, "name": "NSW Swifts", "status": "ACTIVE"},
                        ],
                    },
                    "rc": [
                        {
                            "id": 12345,
                            "batb": [[0, 2.10, 150.0]],
                            "batl": [[0, 2.14, 100.0]],
                            "tv": 250.0,
                        },
                        {
                            "id": 67890,
                            "batb": [[0, 1.85, 200.0]],
                            "batl": [[0, 1.88, 120.0]],
                            "tv": 320.0,
                        },
                    ],
                }
            ],
        }
    ]


def test_parse_market_file(sample_betfair_market):
    parser = BetfairParser()
    odds = parser.parse_market_data(sample_betfair_market)

    assert len(odds) >= 1
    record = odds[0]
    assert record["home_team"] == "Queensland Firebirds"
    assert record["away_team"] == "NSW Swifts"
    assert record["home_back_odds"] == 2.10
    assert record["away_back_odds"] == 1.85


def test_parse_tar_file(sample_betfair_market, tmp_path):
    json_bytes = "\n".join(json.dumps(line) for line in sample_betfair_market).encode()
    tar_path = tmp_path / "betfair_netball.tar"

    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name="1.234567890.json")
        info.size = len(json_bytes)
        tar.addfile(info, io.BytesIO(json_bytes))

    parser = BetfairParser()
    all_odds = parser.parse_tar(tar_path)
    assert len(all_odds) >= 1


def test_skips_non_match_odds(sample_betfair_market):
    # Change market type to something else
    sample_betfair_market[0]["mc"][0]["marketDefinition"]["marketType"] = "OVER_UNDER"
    parser = BetfairParser()
    odds = parser.parse_market_data(sample_betfair_market)
    assert len(odds) == 0
