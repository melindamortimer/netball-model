from __future__ import annotations

import httpx

COMPETITION_IDS: dict[int, dict[str, int | None]] = {
    2017: {"regular": 10083, "finals": 10084},
    2018: {"regular": 10393, "finals": 10394},
    2019: {"regular": 10724, "finals": 10725},
    2020: {"regular": 11108, "finals": 11109},
    2021: {"regular": 11391, "finals": 11392},
    2022: {"regular": 11665, "finals": 11666},
    2023: {"regular": 12045, "finals": 12046},
    2024: {"regular": 12438, "finals": 12439},
    2025: {"regular": 12715, "finals": 12716},
    2026: {"regular": 12949, "finals": None},
}

BASE_URL = "https://mc.championdata.com/data"


class ChampionDataClient:
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self._client.aclose()

    def _build_url(self, comp_id: int, round_num: int, game_num: int) -> str:
        return f"{BASE_URL}/{comp_id}/{comp_id}{round_num:02d}0{game_num}.json"

    async def fetch_match(
        self, comp_id: int, round_num: int, game_num: int
    ) -> dict | None:
        url = self._build_url(comp_id, round_num, game_num)
        resp = await self._client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def parse_match(
        self,
        data: dict,
        competition_id: int,
        season: int,
        round_num: int,
        game_num: int,
    ) -> tuple[dict, list[dict]]:
        ms = data["matchStats"]
        mi = ms["matchInfo"]
        teams = ms["teamInfo"]["team"]

        home_squad_id = mi["homeSquadId"]
        away_squad_id = mi["awaySquadId"]

        home_team = next(
            t["squadName"] for t in teams if t["squadId"] == home_squad_id
        )
        away_team = next(
            t["squadName"] for t in teams if t["squadId"] == away_squad_id
        )

        # Quarter scores from teamPeriodStats
        quarter_scores: dict[str, dict[int, int]] = {"home": {}, "away": {}}
        for row in ms["teamPeriodStats"]["team"]:
            sid = row["squadId"]
            period = int(row["period"])
            goals = int(row["goals"])
            if sid == home_squad_id:
                quarter_scores["home"][period] = goals
            elif sid == away_squad_id:
                quarter_scores["away"][period] = goals

        home_total = sum(quarter_scores["home"].values())
        away_total = sum(quarter_scores["away"].values())

        venue = mi.get("venueName", "")
        date = mi.get("localStartTime", "")

        match_id = f"{competition_id}_{round_num:02d}_{game_num:02d}"

        match = {
            "match_id": match_id,
            "competition_id": competition_id,
            "season": season,
            "round_num": round_num,
            "game_num": game_num,
            "date": date,
            "venue": venue,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_total,
            "away_score": away_total,
            "home_q1": quarter_scores["home"].get(1, 0),
            "home_q2": quarter_scores["home"].get(2, 0),
            "home_q3": quarter_scores["home"].get(3, 0),
            "home_q4": quarter_scores["home"].get(4, 0),
            "away_q1": quarter_scores["away"].get(1, 0),
            "away_q2": quarter_scores["away"].get(2, 0),
            "away_q3": quarter_scores["away"].get(3, 0),
            "away_q4": quarter_scores["away"].get(4, 0),
        }

        players = []
        if "playerStats" in ms and "player" in ms["playerStats"]:
            for p in ms["playerStats"]["player"]:
                sid = p["squadId"]
                team_name = home_team if sid == home_squad_id else away_team
                players.append(
                    {
                        "match_id": match_id,
                        "player_id": int(p["playerId"]),
                        "player_name": f"{p.get('firstname', '')} {p.get('surname', '')}".strip(),
                        "team": team_name,
                        "position": p.get("startingPositionCode", ""),
                        "goals": int(p.get("goals", 0)),
                        "attempts": int(p.get("goalAttempts", 0)),
                        "assists": int(p.get("goalAssists", 0)),
                        "rebounds": int(p.get("rebounds", 0)),
                        "feeds": int(p.get("feeds", 0)),
                        "turnovers": int(p.get("turnovers", 0)),
                        "gains": int(p.get("gain", 0)),
                        "intercepts": int(p.get("intercepts", 0)),
                        "deflections": int(p.get("deflectionWithGain", 0))
                        + int(p.get("deflectionWithNoGain", 0)),
                        "penalties": int(p.get("penalties", 0)),
                        "centre_pass_receives": int(
                            p.get("centrePassReceives", 0)
                        ),
                        "net_points": float(p.get("netPoints", 0)),
                    }
                )

        return match, players

    async def fetch_season(
        self, season: int, max_rounds: int = 17, max_games: int = 4
    ) -> list[tuple[dict, list[dict]]]:
        if season not in COMPETITION_IDS:
            raise ValueError(f"Unknown season: {season}")

        results = []
        for phase in ("regular", "finals"):
            comp_id = COMPETITION_IDS[season][phase]
            if comp_id is None:
                continue

            for round_num in range(1, max_rounds + 1):
                round_empty = True
                for game_num in range(1, max_games + 1):
                    data = await self.fetch_match(comp_id, round_num, game_num)
                    if data is None:
                        continue
                    round_empty = False
                    match, players = self.parse_match(
                        data, comp_id, season, round_num, game_num
                    )
                    results.append((match, players))
                if round_empty:
                    break

        return results
