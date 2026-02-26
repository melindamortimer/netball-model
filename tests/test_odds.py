import json
import tarfile
import io

from click.testing import CliRunner

from netball_model.cli import main


def test_odds_command(tmp_path):
    # Create a minimal Betfair TAR
    market_data = [
        {
            "op": "mcm",
            "clk": "1234",
            "pt": 1711800000000,
            "mc": [
                {
                    "id": "1.234567890",
                    "marketDefinition": {
                        "eventName": "Team A v Team B",
                        "marketType": "MATCH_ODDS",
                        "openDate": "2024-03-30T05:00:00.000Z",
                        "runners": [
                            {"id": 1, "name": "Team A", "status": "ACTIVE"},
                            {"id": 2, "name": "Team B", "status": "ACTIVE"},
                        ],
                    },
                    "rc": [
                        {"id": 1, "batb": [[0, 2.10, 150.0]], "batl": [[0, 2.14, 100.0]], "tv": 250.0},
                        {"id": 2, "batb": [[0, 1.85, 200.0]], "batl": [[0, 1.88, 120.0]], "tv": 320.0},
                    ],
                }
            ],
        }
    ]
    json_bytes = "\n".join(json.dumps(line) for line in market_data).encode()
    tar_path = tmp_path / "betfair.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name="1.234567890.json")
        info.size = len(json_bytes)
        tar.addfile(info, io.BytesIO(json_bytes))

    db_path = tmp_path / "test.db"
    runner = CliRunner()
    result = runner.invoke(main, ["odds", "--file", str(tar_path), "--db", str(db_path)])
    assert result.exit_code == 0
    assert "Imported" in result.output
