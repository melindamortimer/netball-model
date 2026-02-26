from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


class BetfairParser:
    """Parse Betfair historical TAR/JSON data files.

    Extracts MATCH_ODDS markets with back/lay prices and volume from
    Betfair's streaming data format (newline-delimited JSON).
    """

    def parse_market_data(self, lines: list[dict]) -> list[dict]:
        """Parse a list of Betfair market change messages.

        Each message follows Betfair's streaming API format with op=mcm.
        Only MATCH_ODDS markets are extracted; other market types are skipped.

        Args:
            lines: List of parsed JSON objects from a Betfair data file.

        Returns:
            List of dicts with home/away odds, volume, and timestamps.
        """
        results = []
        home_team = None
        away_team = None
        runner_map: dict[int, str] = {}
        market_type = None
        event_date = None

        for line in lines:
            if "mc" not in line:
                continue
            for mc in line["mc"]:
                if "marketDefinition" in mc:
                    md = mc["marketDefinition"]
                    market_type = md.get("marketType")
                    event_date = md.get("openDate", "")
                    event_name = md.get("eventName", "")

                    if market_type != "MATCH_ODDS":
                        continue

                    runners = md.get("runners", [])
                    if len(runners) >= 2:
                        runner_map = {r["id"]: r["name"] for r in runners}
                        if " v " in event_name:
                            parts = event_name.split(" v ", 1)
                            home_team = parts[0].strip()
                            away_team = parts[1].strip()
                        else:
                            names = list(runner_map.values())
                            home_team = names[0]
                            away_team = names[1]

                if market_type != "MATCH_ODDS" or not home_team:
                    continue

                if "rc" not in mc:
                    continue

                timestamp = datetime.fromtimestamp(
                    line.get("pt", 0) / 1000, tz=timezone.utc
                ).isoformat()

                home_back = None
                home_lay = None
                away_back = None
                away_lay = None
                home_vol = 0.0
                away_vol = 0.0

                for rc in mc["rc"]:
                    runner_name = runner_map.get(rc["id"], "")
                    back = rc.get("batb", [])
                    lay = rc.get("batl", [])
                    vol = rc.get("tv", 0.0)

                    best_back = back[0][1] if back else None
                    best_lay = lay[0][1] if lay else None

                    if runner_name == home_team:
                        home_back = best_back
                        home_lay = best_lay
                        home_vol = vol
                    elif runner_name == away_team:
                        away_back = best_back
                        away_lay = best_lay
                        away_vol = vol

                if home_back is not None or away_back is not None:
                    results.append(
                        {
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_back_odds": home_back,
                            "home_lay_odds": home_lay,
                            "away_back_odds": away_back,
                            "away_lay_odds": away_lay,
                            "home_volume": home_vol,
                            "away_volume": away_vol,
                            "timestamp": timestamp,
                            "event_date": event_date,
                        }
                    )

        return results

    def parse_tar(self, tar_path: str | Path) -> list[dict]:
        """Parse all MATCH_ODDS markets from a Betfair TAR archive.

        Betfair historical data is distributed as TAR files containing
        one JSON file per market. Each JSON file is newline-delimited
        with one streaming message per line.

        Args:
            tar_path: Path to the TAR (or compressed TAR) archive.

        Returns:
            List of odds records from all MATCH_ODDS markets in the archive.
        """
        all_odds = []
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".json"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read().decode("utf-8")
                lines = []
                for raw_line in content.strip().split("\n"):
                    if raw_line.strip():
                        lines.append(json.loads(raw_line))
                odds = self.parse_market_data(lines)
                all_odds.extend(odds)
        return all_odds
