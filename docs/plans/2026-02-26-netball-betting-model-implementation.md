# Netball Betting Model — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python CLI tool that predicts netball match outcomes using Glicko-2 ratings + Ridge regression and compares predictions against Betfair exchange odds to find value bets.

**Architecture:** Data flows from Champion Data API (match/player stats) and Betfair (historical odds) into SQLite. A Glicko-2 rating system tracks team strength over time. Contextual features (home/away, rest, form, etc.) feed into Ridge regression to predict margins. Calibrated probability estimates are compared against exchange odds to flag value.

**Tech Stack:** Python 3.11+, Poetry, httpx, pandas, scikit-learn, glicko2, click, rich, pytest, SQLite.

**Reference docs:**
- Design: `docs/plans/2026-02-26-netball-betting-model-design.md`
- Champion Data API: `docs/champion_data_api_reference.md`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/netball_model/__init__.py`
- Create: `src/netball_model/cli.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `.gitignore`

**Step 1: Initialize Poetry project**

Run:
```bash
cd /Users/mortimerme/Documents/GitHub-personal/netball-model
poetry init --name netball-model --python "^3.11" --no-interaction
```

**Step 2: Configure pyproject.toml**

Replace the generated `pyproject.toml` with:

```toml
[tool.poetry]
name = "netball-model"
version = "0.1.0"
description = "Netball betting model — Glicko-2 ratings + Ridge regression"
authors = []
packages = [{include = "netball_model", from = "src"}]

[tool.poetry.dependencies]
python = "^3.11"
httpx = "^0.27"
pandas = "^2.2"
scikit-learn = "^1.4"
glicko2 = "^2.0"
click = "^8.1"
rich = "^13.7"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.23"
respx = "^0.21"

[tool.poetry.scripts]
netball = "netball_model.cli:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 3: Create directory structure**

```bash
mkdir -p src/netball_model/data
mkdir -p src/netball_model/features
mkdir -p src/netball_model/model
mkdir -p src/netball_model/value
mkdir -p tests/data
mkdir -p tests/features
mkdir -p tests/model
mkdir -p tests/value
mkdir -p data
mkdir -p notebooks
```

**Step 4: Create __init__.py files**

`src/netball_model/__init__.py`:
```python
"""Netball betting model."""
```

Create empty `__init__.py` in each subpackage:
- `src/netball_model/data/__init__.py`
- `src/netball_model/features/__init__.py`
- `src/netball_model/model/__init__.py`
- `src/netball_model/value/__init__.py`
- `tests/__init__.py`
- `tests/data/__init__.py`
- `tests/features/__init__.py`
- `tests/model/__init__.py`
- `tests/value/__init__.py`

**Step 5: Create minimal CLI**

`src/netball_model/cli.py`:
```python
import click


@click.group()
def main():
    """Netball betting model CLI."""
    pass
```

**Step 6: Create conftest.py**

`tests/conftest.py`:
```python
import pathlib
import tempfile

import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path."""
    return tmp_path / "test.db"
```

**Step 7: Add data/ to .gitignore**

Append to `.gitignore`:
```
# Project data
data/*.db
data/*.tar
data/*.json
```

**Step 8: Install dependencies**

```bash
poetry install
```

**Step 9: Write a smoke test**

`tests/test_cli.py`:
```python
from click.testing import CliRunner

from netball_model.cli import main


def test_cli_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Netball betting model CLI" in result.output
```

**Step 10: Run test**

```bash
poetry run pytest tests/test_cli.py -v
```
Expected: PASS

**Step 11: Commit**

```
feat: scaffold project with Poetry, CLI, and test infrastructure
```

---

## Task 2: Database Schema

**Files:**
- Create: `src/netball_model/data/database.py`
- Create: `tests/data/test_database.py`

**Step 1: Write the failing test**

`tests/data/test_database.py`:
```python
import sqlite3

from netball_model.data.database import Database


def test_creates_tables(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    conn = sqlite3.connect(tmp_db)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "matches" in tables
    assert "player_stats" in tables
    assert "odds_history" in tables
    assert "elo_ratings" in tables


def test_upsert_match(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    match = {
        "match_id": "10393_01_01",
        "competition_id": 10393,
        "season": 2018,
        "round_num": 1,
        "game_num": 1,
        "date": "2018-04-28",
        "venue": "Brisbane Entertainment Centre",
        "home_team": "Queensland Firebirds",
        "away_team": "NSW Swifts",
        "home_score": 55,
        "away_score": 60,
        "home_q1": 14,
        "home_q2": 13,
        "home_q3": 15,
        "home_q4": 13,
        "away_q1": 16,
        "away_q2": 14,
        "away_q3": 15,
        "away_q4": 15,
    }

    db.upsert_match(match)
    rows = db.get_matches(season=2018)
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Queensland Firebirds"

    # Upsert again — should not duplicate
    db.upsert_match(match)
    rows = db.get_matches(season=2018)
    assert len(rows) == 1


def test_insert_player_stats(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    stats = {
        "match_id": "10393_01_01",
        "player_id": 12345,
        "player_name": "Gretel Bueta",
        "team": "Queensland Firebirds",
        "position": "GS",
        "goals": 30,
        "attempts": 33,
        "assists": 2,
        "rebounds": 4,
        "feeds": 5,
        "turnovers": 3,
        "gains": 0,
        "intercepts": 0,
        "deflections": 0,
        "penalties": 2,
        "centre_pass_receives": 0,
        "net_points": 78.5,
    }

    db.insert_player_stats(stats)
    rows = db.get_player_stats(match_id="10393_01_01")
    assert len(rows) == 1
    assert rows[0]["player_name"] == "Gretel Bueta"
```

**Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/data/test_database.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement Database class**

`src/netball_model/data/database.py`:
```python
from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self):
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                competition_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                round_num INTEGER NOT NULL,
                game_num INTEGER NOT NULL,
                date TEXT,
                venue TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                home_q1 INTEGER,
                home_q2 INTEGER,
                home_q3 INTEGER,
                home_q4 INTEGER,
                away_q1 INTEGER,
                away_q2 INTEGER,
                away_q3 INTEGER,
                away_q4 INTEGER
            );

            CREATE TABLE IF NOT EXISTS player_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT,
                goals INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                rebounds INTEGER DEFAULT 0,
                feeds INTEGER DEFAULT 0,
                turnovers INTEGER DEFAULT 0,
                gains INTEGER DEFAULT 0,
                intercepts INTEGER DEFAULT 0,
                deflections INTEGER DEFAULT 0,
                penalties INTEGER DEFAULT 0,
                centre_pass_receives INTEGER DEFAULT 0,
                net_points REAL DEFAULT 0,
                UNIQUE(match_id, player_id)
            );

            CREATE TABLE IF NOT EXISTS odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'betfair',
                home_back_odds REAL,
                home_lay_odds REAL,
                away_back_odds REAL,
                away_lay_odds REAL,
                home_volume REAL DEFAULT 0,
                away_volume REAL DEFAULT 0,
                timestamp TEXT,
                UNIQUE(match_id, source, timestamp)
            );

            CREATE TABLE IF NOT EXISTS elo_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT NOT NULL,
                pool TEXT NOT NULL DEFAULT 'ssn',
                match_id TEXT,
                rating REAL NOT NULL DEFAULT 1500.0,
                rd REAL NOT NULL DEFAULT 350.0,
                vol REAL NOT NULL DEFAULT 0.06,
                timestamp TEXT,
                UNIQUE(team, pool, match_id)
            );

            CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
            CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_stats(match_id);
            CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_history(match_id);
            CREATE INDEX IF NOT EXISTS idx_elo_team ON elo_ratings(team, pool);
            """
        )
        conn.commit()
        conn.close()

    def upsert_match(self, match: dict):
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO matches (
                match_id, competition_id, season, round_num, game_num,
                date, venue, home_team, away_team,
                home_score, away_score,
                home_q1, home_q2, home_q3, home_q4,
                away_q1, away_q2, away_q3, away_q4
            ) VALUES (
                :match_id, :competition_id, :season, :round_num, :game_num,
                :date, :venue, :home_team, :away_team,
                :home_score, :away_score,
                :home_q1, :home_q2, :home_q3, :home_q4,
                :away_q1, :away_q2, :away_q3, :away_q4
            )
            ON CONFLICT(match_id) DO UPDATE SET
                home_score=excluded.home_score,
                away_score=excluded.away_score,
                home_q1=excluded.home_q1, home_q2=excluded.home_q2,
                home_q3=excluded.home_q3, home_q4=excluded.home_q4,
                away_q1=excluded.away_q1, away_q2=excluded.away_q2,
                away_q3=excluded.away_q3, away_q4=excluded.away_q4
            """,
            match,
        )
        conn.commit()
        conn.close()

    def insert_player_stats(self, stats: dict):
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO player_stats (
                match_id, player_id, player_name, team, position,
                goals, attempts, assists, rebounds, feeds,
                turnovers, gains, intercepts, deflections,
                penalties, centre_pass_receives, net_points
            ) VALUES (
                :match_id, :player_id, :player_name, :team, :position,
                :goals, :attempts, :assists, :rebounds, :feeds,
                :turnovers, :gains, :intercepts, :deflections,
                :penalties, :centre_pass_receives, :net_points
            )
            """,
            stats,
        )
        conn.commit()
        conn.close()

    def get_matches(self, season: int | None = None) -> list[dict]:
        conn = self._connect()
        if season:
            cursor = conn.execute(
                "SELECT * FROM matches WHERE season = ? ORDER BY date", (season,)
            )
        else:
            cursor = conn.execute("SELECT * FROM matches ORDER BY date")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_player_stats(self, match_id: str) -> list[dict]:
        conn = self._connect()
        cursor = conn.execute(
            "SELECT * FROM player_stats WHERE match_id = ?", (match_id,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def upsert_odds(self, odds: dict):
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO odds_history (
                match_id, source, home_back_odds, home_lay_odds,
                away_back_odds, away_lay_odds,
                home_volume, away_volume, timestamp
            ) VALUES (
                :match_id, :source, :home_back_odds, :home_lay_odds,
                :away_back_odds, :away_lay_odds,
                :home_volume, :away_volume, :timestamp
            )
            """,
            odds,
        )
        conn.commit()
        conn.close()

    def upsert_elo(self, elo: dict):
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO elo_ratings (
                team, pool, match_id, rating, rd, vol, timestamp
            ) VALUES (
                :team, :pool, :match_id, :rating, :rd, :vol, :timestamp
            )
            """,
            elo,
        )
        conn.commit()
        conn.close()

    def get_latest_elo(self, team: str, pool: str = "ssn") -> dict | None:
        conn = self._connect()
        cursor = conn.execute(
            """
            SELECT * FROM elo_ratings
            WHERE team = ? AND pool = ?
            ORDER BY id DESC LIMIT 1
            """,
            (team, pool),
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
```

**Step 4: Run tests**

```bash
poetry run pytest tests/data/test_database.py -v
```
Expected: PASS

**Step 5: Commit**

```
feat: add SQLite database schema and CRUD operations
```

---

## Task 3: Champion Data API Client

**Files:**
- Create: `src/netball_model/data/champion_data.py`
- Create: `tests/data/test_champion_data.py`
- Create: `tests/data/fixtures/sample_match.json`

**Reference:** `docs/champion_data_api_reference.md`

**Step 1: Create test fixture**

Save a minimal sample JSON that mirrors the Champion Data response structure to `tests/data/fixtures/sample_match.json`:

```json
{
  "matchStats": {
    "matchInfo": {
      "matchId": [{"matchId": "1001"}],
      "matchType": [{"matchType": "H&A"}],
      "venue": [{"venueName": "Brisbane Entertainment Centre"}],
      "localStartTime": [{"localStartTime": "2024-03-30 15:00:00"}],
      "period": [
        {"period": "1", "periodCompleted": "1"},
        {"period": "2", "periodCompleted": "1"},
        {"period": "3", "periodCompleted": "1"},
        {"period": "4", "periodCompleted": "1"}
      ]
    },
    "teamInfo": {
      "team": [
        {"squadId": "801", "squadName": "Queensland Firebirds", "squadCode": "FIR"},
        {"squadId": "810", "squadName": "NSW Swifts", "squadCode": "SWI"}
      ]
    },
    "teamPeriodStats": {
      "team": [
        {"squadId": "801", "period": "1", "goals": "14", "goal1": "14", "goal2": "0", "attempt1": "16", "attempt2": "0", "goalMisses": "2"},
        {"squadId": "801", "period": "2", "goals": "13", "goal1": "13", "goal2": "0", "attempt1": "14", "attempt2": "0", "goalMisses": "1"},
        {"squadId": "801", "period": "3", "goals": "15", "goal1": "15", "goal2": "0", "attempt1": "17", "attempt2": "0", "goalMisses": "2"},
        {"squadId": "801", "period": "4", "goals": "13", "goal1": "13", "goal2": "0", "attempt1": "15", "attempt2": "0", "goalMisses": "2"},
        {"squadId": "810", "period": "1", "goals": "16", "goal1": "16", "goal2": "0", "attempt1": "17", "attempt2": "0", "goalMisses": "1"},
        {"squadId": "810", "period": "2", "goals": "14", "goal1": "14", "goal2": "0", "attempt1": "15", "attempt2": "0", "goalMisses": "1"},
        {"squadId": "810", "period": "3", "goals": "15", "goal1": "15", "goal2": "0", "attempt1": "16", "attempt2": "0", "goalMisses": "1"},
        {"squadId": "810", "period": "4", "goals": "15", "goal1": "15", "goal2": "0", "attempt1": "16", "attempt2": "0", "goalMisses": "1"}
      ]
    },
    "playerStats": {
      "player": [
        {
          "playerId": "12345", "displayName": "G. Bueta",
          "firstname": "Gretel", "surname": "Bueta",
          "squadId": "801", "startingPosition": "GS",
          "goals": "30", "goalAttempts": "33",
          "goalAssists": "2", "rebounds": "4",
          "feeds": "5", "feedWithAttempt": "3",
          "turnovers": "3", "generalPlayTurnovers": "2",
          "gains": "0", "intercepts": "0",
          "deflections": "0", "deflectionWithGain": "0",
          "deflectionWithNoGain": "0", "penalties": "2",
          "contactPenalties": "1", "obstructionPenalties": "1",
          "centrePassReceives": "0",
          "netPoints": "78.5"
        },
        {
          "playerId": "67890", "displayName": "S. Wallace",
          "firstname": "Sophie", "surname": "Wallace",
          "squadId": "810", "startingPosition": "GA",
          "goals": "25", "goalAttempts": "28",
          "goalAssists": "5", "rebounds": "2",
          "feeds": "8", "feedWithAttempt": "5",
          "turnovers": "4", "generalPlayTurnovers": "3",
          "gains": "1", "intercepts": "1",
          "deflections": "2", "deflectionWithGain": "1",
          "deflectionWithNoGain": "1", "penalties": "3",
          "contactPenalties": "2", "obstructionPenalties": "1",
          "centrePassReceives": "3",
          "netPoints": "65.0"
        }
      ]
    }
  }
}
```

**Step 2: Write the failing tests**

`tests/data/test_champion_data.py`:
```python
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


@respx.mock
@pytest.mark.asyncio
async def test_parse_match():
    sample = json.loads((FIXTURES / "sample_match.json").read_text())

    client = ChampionDataClient()
    match, players = client.parse_match(
        sample, competition_id=10393, season=2018, round_num=1, game_num=1
    )

    assert match["home_team"] == "Queensland Firebirds"
    assert match["away_team"] == "NSW Swifts"
    assert match["home_score"] == 55
    assert match["away_score"] == 60
    assert match["home_q1"] == 14
    assert match["away_q4"] == 15

    assert len(players) == 2
    assert players[0]["player_name"] == "Gretel Bueta"
    assert players[0]["goals"] == 30
    assert players[0]["net_points"] == 78.5
```

**Step 3: Run tests to verify they fail**

```bash
poetry run pytest tests/data/test_champion_data.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Implement ChampionDataClient**

`src/netball_model/data/champion_data.py`:
```python
from __future__ import annotations

import httpx

COMPETITION_IDS: dict[int, dict[str, int]] = {
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
        teams = ms["teamInfo"]["team"]
        home_team = teams[0]["squadName"]
        away_team = teams[1]["squadName"]
        home_id = teams[0]["squadId"]
        away_id = teams[1]["squadId"]

        # Quarter scores from teamPeriodStats
        quarter_scores = {"home": {}, "away": {}}
        for row in ms["teamPeriodStats"]["team"]:
            side = "home" if row["squadId"] == home_id else "away"
            period = int(row["period"])
            quarter_scores[side][period] = int(row["goals"])

        home_total = sum(quarter_scores["home"].values())
        away_total = sum(quarter_scores["away"].values())

        venue = ""
        if "venue" in ms["matchInfo"] and ms["matchInfo"]["venue"]:
            venue = ms["matchInfo"]["venue"][0].get("venueName", "")

        date = ""
        if "localStartTime" in ms["matchInfo"] and ms["matchInfo"]["localStartTime"]:
            date = ms["matchInfo"]["localStartTime"][0].get("localStartTime", "")

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
                team_name = home_team if p["squadId"] == home_id else away_team
                players.append(
                    {
                        "match_id": match_id,
                        "player_id": int(p["playerId"]),
                        "player_name": f"{p.get('firstname', '')} {p.get('surname', '')}".strip(),
                        "team": team_name,
                        "position": p.get("startingPosition", ""),
                        "goals": int(p.get("goals", 0)),
                        "attempts": int(p.get("goalAttempts", 0)),
                        "assists": int(p.get("goalAssists", 0)),
                        "rebounds": int(p.get("rebounds", 0)),
                        "feeds": int(p.get("feeds", 0)),
                        "turnovers": int(p.get("turnovers", 0)),
                        "gains": int(p.get("gains", 0)),
                        "intercepts": int(p.get("intercepts", 0)),
                        "deflections": int(p.get("deflections", 0)),
                        "penalties": int(p.get("penalties", 0)),
                        "centre_pass_receives": int(p.get("centrePassReceives", 0)),
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
```

**Step 5: Run tests**

```bash
poetry run pytest tests/data/test_champion_data.py -v
```
Expected: PASS

**Step 6: Commit**

```
feat: add Champion Data API client with match parsing
```

---

## Task 4: Ingest CLI Command

**Files:**
- Modify: `src/netball_model/cli.py`
- Create: `tests/test_ingest.py`

**Step 1: Write the failing test**

`tests/test_ingest.py`:
```python
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
```

**Step 2: Implement the ingest command**

Update `src/netball_model/cli.py`:
```python
import asyncio

import click

from netball_model.data.champion_data import ChampionDataClient
from netball_model.data.database import Database

DEFAULT_DB = "data/netball.db"


@click.group()
def main():
    """Netball betting model CLI."""
    pass


@main.command()
@click.option("--season", required=True, type=int, help="SSN season year (e.g. 2024)")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def ingest(season: int, db: str):
    """Pull match + player data from Champion Data."""
    asyncio.run(_ingest(season, db))


async def _ingest(season: int, db_path: str):
    db = Database(db_path)
    db.initialize()

    client = ChampionDataClient()
    try:
        click.echo(f"Fetching SSN {season} data from Champion Data...")
        results = await client.fetch_season(season)

        for match, players in results:
            db.upsert_match(match)
            for p in players:
                db.insert_player_stats(p)

        click.echo(f"Ingested {len(results)} matches for SSN {season}.")
    finally:
        await client.close()
```

**Step 3: Run tests**

```bash
poetry run pytest tests/test_ingest.py tests/test_cli.py -v
```
Expected: PASS

**Step 4: Manual smoke test (actually hits API)**

```bash
poetry run netball ingest --season 2024 --db data/netball.db
```
Expected: Downloads real data. Verify with:
```bash
python -c "
from netball_model.data.database import Database
db = Database('data/netball.db')
rows = db.get_matches(season=2024)
print(f'{len(rows)} matches ingested')
for r in rows[:3]:
    print(f'  {r[\"date\"]} {r[\"home_team\"]} {r[\"home_score\"]} - {r[\"away_score\"]} {r[\"away_team\"]}')
"
```

**Step 5: Commit**

```
feat: add ingest CLI command for Champion Data
```

---

## Task 5: Betfair Historical Odds Parser

**Files:**
- Create: `src/netball_model/data/betfair.py`
- Create: `tests/data/test_betfair.py`

**Context:** Betfair historical data comes as TAR files containing JSON. The user downloads these manually from `historicdata.betfair.com` (filter: Sport=Netball). Each JSON file represents a market (e.g., Match Odds for one match). The structure contains `runners` (selections) with `ex.availableToBack` and `ex.availableToLay` prices, and `totalMatched` volume.

**Step 1: Write the failing test**

`tests/data/test_betfair.py`:
```python
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
    # Create a fake TAR with one JSON
    json_bytes = "\n".join(json.dumps(line) for line in sample_betfair_market).encode()
    tar_path = tmp_path / "betfair_netball.tar"

    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name="1.234567890.json")
        info.size = len(json_bytes)
        tar.addfile(info, io.BytesIO(json_bytes))

    parser = BetfairParser()
    all_odds = parser.parse_tar(tar_path)
    assert len(all_odds) >= 1
```

**Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/data/test_betfair.py -v
```

**Step 3: Implement BetfairParser**

`src/netball_model/data/betfair.py`:
```python
from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


class BetfairParser:
    def parse_market_data(self, lines: list[dict]) -> list[dict]:
        """Parse a list of Betfair streaming JSON lines into odds records."""
        results = []
        home_team = None
        away_team = None
        runner_map = {}
        market_type = None
        event_date = None

        for line in lines:
            if "mc" not in line:
                continue
            for mc in line["mc"]:
                # Extract market definition if present
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
                        # Parse "Team A v Team B" from event name
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

                # Extract runner changes (odds)
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
        """Parse all MATCH_ODDS markets from a Betfair TAR file."""
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
```

**Step 4: Run tests**

```bash
poetry run pytest tests/data/test_betfair.py -v
```
Expected: PASS

**Step 5: Commit**

```
feat: add Betfair historical odds TAR/JSON parser
```

---

## Task 6: Odds CLI Command

**Files:**
- Modify: `src/netball_model/cli.py`

**Step 1: Add the odds command**

Add to `src/netball_model/cli.py`:
```python
from netball_model.data.betfair import BetfairParser

@main.command()
@click.option("--file", "tar_file", required=True, type=click.Path(exists=True), help="Betfair TAR file path")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def odds(tar_file: str, db: str):
    """Import Betfair historical odds from a TAR file."""
    db_conn = Database(db)
    db_conn.initialize()

    parser = BetfairParser()
    click.echo(f"Parsing {tar_file}...")
    all_odds = parser.parse_tar(tar_file)

    # We need to match odds to matches by team names + date.
    # For now, store with a generated match_id placeholder.
    imported = 0
    for record in all_odds:
        # Use last odds snapshot per match (closest to kick-off)
        odds_record = {
            "match_id": f"bf_{record['home_team']}_{record['event_date'][:10]}",
            "source": "betfair",
            "home_back_odds": record["home_back_odds"],
            "home_lay_odds": record["home_lay_odds"],
            "away_back_odds": record["away_back_odds"],
            "away_lay_odds": record["away_lay_odds"],
            "home_volume": record["home_volume"],
            "away_volume": record["away_volume"],
            "timestamp": record["timestamp"],
        }
        db_conn.upsert_odds(odds_record)
        imported += 1

    click.echo(f"Imported {imported} odds records.")
```

**Step 2: Test manually with a real Betfair file (once downloaded)**

```bash
poetry run netball odds --file data/betfair_netball_2024.tar --db data/netball.db
```

**Step 3: Commit**

```
feat: add odds CLI command for Betfair TAR import
```

---

## Task 7: Glicko-2 Rating System

**Files:**
- Create: `src/netball_model/features/elo.py`
- Create: `tests/features/test_elo.py`

**Step 1: Write the failing tests**

`tests/features/test_elo.py`:
```python
from netball_model.features.elo import GlickoSystem


def test_initial_ratings():
    system = GlickoSystem()
    rating = system.get_rating("Queensland Firebirds")
    assert rating["rating"] == 1500.0
    assert rating["rd"] == 350.0


def test_update_after_match():
    system = GlickoSystem()
    system.update("Queensland Firebirds", "NSW Swifts", winner="away", margin=5)

    fb = system.get_rating("Queensland Firebirds")
    sw = system.get_rating("NSW Swifts")

    # Loser rating should decrease, winner should increase
    assert fb["rating"] < 1500.0
    assert sw["rating"] > 1500.0
    # Rating deviation should decrease (more certainty)
    assert fb["rd"] < 350.0
    assert sw["rd"] < 350.0


def test_ratings_converge_over_multiple_matches():
    system = GlickoSystem()
    # Team A wins 10 straight against Team B
    for _ in range(10):
        system.update("Team A", "Team B", winner="home", margin=10)

    a = system.get_rating("Team A")
    b = system.get_rating("Team B")

    assert a["rating"] > 1600
    assert b["rating"] < 1400


def test_separate_pools():
    system = GlickoSystem()
    system.update("Australia", "New Zealand", winner="home", margin=5, pool="international")
    system.update("Firebirds", "Swifts", winner="home", margin=5, pool="ssn")

    # International pool should not affect SSN pool
    assert system.get_rating("Australia", pool="ssn")["rating"] == 1500.0
    assert system.get_rating("Australia", pool="international")["rating"] > 1500.0


def test_predicted_win_probability():
    system = GlickoSystem()
    # After establishing some rating difference
    for _ in range(5):
        system.update("Strong Team", "Weak Team", winner="home", margin=15)

    prob = system.predict_win_prob("Strong Team", "Weak Team")
    assert prob > 0.5
    assert prob < 1.0
```

**Step 2: Run tests to verify failure**

```bash
poetry run pytest tests/features/test_elo.py -v
```

**Step 3: Implement GlickoSystem**

`src/netball_model/features/elo.py`:
```python
from __future__ import annotations

import math
from dataclasses import dataclass, field

from glicko2 import Player


@dataclass
class TeamRating:
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06


class GlickoSystem:
    def __init__(self):
        self._ratings: dict[str, dict[str, TeamRating]] = {}

    def _key(self, team: str, pool: str) -> TeamRating:
        if pool not in self._ratings:
            self._ratings[pool] = {}
        if team not in self._ratings[pool]:
            self._ratings[pool][team] = TeamRating()
        return self._ratings[pool][team]

    def get_rating(self, team: str, pool: str = "ssn") -> dict:
        tr = self._key(team, pool)
        return {"rating": tr.rating, "rd": tr.rd, "vol": tr.vol}

    def update(
        self,
        home_team: str,
        away_team: str,
        winner: str,
        margin: int = 0,
        pool: str = "ssn",
    ):
        home_tr = self._key(home_team, pool)
        away_tr = self._key(away_team, pool)

        home_player = Player(rating=home_tr.rating, rd=home_tr.rd, vol=home_tr.vol)
        away_player = Player(rating=away_tr.rating, rd=away_tr.rd, vol=away_tr.vol)

        if winner == "home":
            home_score, away_score = 1.0, 0.0
        elif winner == "away":
            home_score, away_score = 0.0, 1.0
        else:
            home_score, away_score = 0.5, 0.5

        # Apply margin-of-victory multiplier to make blowouts more informative
        mov_mult = math.log(max(abs(margin), 1) + 1)

        home_player.update_player(
            [away_player.rating], [away_player.rd], [home_score]
        )
        away_player.update_player(
            [home_player.rating], [home_player.rd], [away_score]
        )

        # Scale the update by MOV (move rating further for big wins)
        home_delta = home_player.rating - home_tr.rating
        away_delta = away_player.rating - away_tr.rating

        home_tr.rating = home_tr.rating + home_delta * mov_mult
        home_tr.rd = home_player.rd
        home_tr.vol = home_player.vol

        away_tr.rating = away_tr.rating + away_delta * mov_mult
        away_tr.rd = away_player.rd
        away_tr.vol = away_player.vol

    def predict_win_prob(
        self, home_team: str, away_team: str, pool: str = "ssn"
    ) -> float:
        home = self._key(home_team, pool)
        away = self._key(away_team, pool)

        q = math.log(10) / 400
        combined_rd = math.sqrt(home.rd**2 + away.rd**2)
        g = 1 / math.sqrt(1 + 3 * q**2 * combined_rd**2 / math.pi**2)
        expected = 1 / (1 + 10 ** (-g * (home.rating - away.rating) / 400))
        return expected

    def get_all_ratings(self, pool: str = "ssn") -> dict[str, dict]:
        if pool not in self._ratings:
            return {}
        return {
            team: {"rating": tr.rating, "rd": tr.rd, "vol": tr.vol}
            for team, tr in self._ratings[pool].items()
        }
```

**Step 4: Run tests**

```bash
poetry run pytest tests/features/test_elo.py -v
```
Expected: PASS

**Step 5: Commit**

```
feat: add Glicko-2 rating system with margin-of-victory scaling
```

---

## Task 8: Contextual Features

**Files:**
- Create: `src/netball_model/features/contextual.py`
- Create: `tests/features/test_contextual.py`

**Step 1: Write the failing tests**

`tests/features/test_contextual.py`:
```python
from netball_model.features.contextual import ContextualFeatures


def test_rest_days():
    matches = [
        {"date": "2024-03-30", "home_team": "Firebirds", "away_team": "Swifts"},
        {"date": "2024-04-06", "home_team": "Firebirds", "away_team": "Vixens"},
    ]
    cf = ContextualFeatures(matches)
    rest = cf.rest_days("Firebirds", match_index=1)
    assert rest == 7


def test_rest_days_first_match():
    matches = [
        {"date": "2024-03-30", "home_team": "Firebirds", "away_team": "Swifts"},
    ]
    cf = ContextualFeatures(matches)
    rest = cf.rest_days("Firebirds", match_index=0)
    assert rest is None


def test_recent_form():
    matches = [
        {"home_team": "A", "away_team": "B", "home_score": 60, "away_score": 50},
        {"home_team": "C", "away_team": "A", "home_score": 45, "away_score": 55},
        {"home_team": "A", "away_team": "D", "home_score": 40, "away_score": 50},
        {"home_team": "E", "away_team": "A", "home_score": 48, "away_score": 52},
        {"home_team": "A", "away_team": "F", "home_score": 58, "away_score": 55},
        {"home_team": "A", "away_team": "G", "home_score": 60, "away_score": 50},
    ]
    cf = ContextualFeatures(matches)
    win_rate, avg_margin = cf.recent_form("A", match_index=5, window=5)
    # Matches 0-4 for team A: W(+10), W(+10), L(-10), W(+4), W(+3)
    assert win_rate == 4 / 5
    assert avg_margin == (10 + 10 - 10 + 4 + 3) / 5


def test_head_to_head():
    matches = [
        {"home_team": "A", "away_team": "B", "home_score": 60, "away_score": 50},
        {"home_team": "B", "away_team": "A", "home_score": 55, "away_score": 45},
        {"home_team": "A", "away_team": "B", "home_score": 70, "away_score": 60},
        {"home_team": "A", "away_team": "B", "home_score": 50, "away_score": 50},
    ]
    cf = ContextualFeatures(matches)
    h2h_win_rate = cf.head_to_head("A", "B", match_index=3)
    # Matches 0-2: A wins match 0 and 2, loses match 1
    assert h2h_win_rate == 2 / 3


VENUE_CITIES = {
    "Brisbane Entertainment Centre": "Brisbane",
    "John Cain Arena": "Melbourne",
    "RAC Arena": "Perth",
    "Ken Rosewall Arena": "Sydney",
}


def test_travel_distance():
    cf = ContextualFeatures([])
    dist = cf.travel_distance("Brisbane", "Perth")
    assert dist > 3000  # roughly 3600km


def test_is_home():
    cf = ContextualFeatures([])
    match = {"home_team": "Firebirds", "away_team": "Swifts"}
    assert cf.is_home("Firebirds", match) is True
    assert cf.is_home("Swifts", match) is False
```

**Step 2: Run to verify failure**

```bash
poetry run pytest tests/features/test_contextual.py -v
```

**Step 3: Implement ContextualFeatures**

`src/netball_model/features/contextual.py`:
```python
from __future__ import annotations

import math
from datetime import datetime

# Approximate lat/lon for SSN venue cities
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Brisbane": (-27.47, 153.03),
    "Melbourne": (-37.81, 144.96),
    "Sydney": (-33.87, 151.21),
    "Perth": (-31.95, 115.86),
    "Adelaide": (-34.93, 138.60),
    "Launceston": (-41.45, 147.14),
    "Gold Coast": (-28.02, 153.40),
    "Canberra": (-35.28, 149.13),
    "Hobart": (-42.88, 147.33),
}

# Map venue names to cities
VENUE_TO_CITY: dict[str, str] = {
    "Brisbane Entertainment Centre": "Brisbane",
    "Nissan Arena": "Brisbane",
    "Queensland State Netball Centre": "Brisbane",
    "John Cain Arena": "Melbourne",
    "Melbourne Arena": "Melbourne",
    "State Netball Hockey Centre": "Melbourne",
    "RAC Arena": "Perth",
    "Perth Arena": "Perth",
    "Ken Rosewall Arena": "Sydney",
    "Qudos Bank Arena": "Sydney",
    "Sydney Olympic Park": "Sydney",
    "Adelaide Entertainment Centre": "Adelaide",
    "Priceline Stadium": "Adelaide",
    "USC Stadium": "Gold Coast",
    "Gold Coast Convention Centre": "Gold Coast",
    "Silverdome": "Launceston",
    "MyState Bank Arena": "Launceston",
    "AIS Arena": "Canberra",
}

# Map team names to home cities
TEAM_HOME_CITY: dict[str, str] = {
    "Queensland Firebirds": "Brisbane",
    "Melbourne Vixens": "Melbourne",
    "NSW Swifts": "Sydney",
    "GIANTS Netball": "Sydney",
    "West Coast Fever": "Perth",
    "Adelaide Thunderbirds": "Adelaide",
    "Collingwood Magpies": "Melbourne",
    "Sunshine Coast Lightning": "Gold Coast",
}


class ContextualFeatures:
    def __init__(self, matches: list[dict]):
        self.matches = matches

    def rest_days(self, team: str, match_index: int) -> int | None:
        """Days since team's previous match."""
        current_date = self.matches[match_index].get("date", "")
        if not current_date:
            return None

        for i in range(match_index - 1, -1, -1):
            m = self.matches[i]
            if team in (m.get("home_team"), m.get("away_team")):
                prev_date = m.get("date", "")
                if prev_date:
                    d1 = datetime.fromisoformat(current_date[:10])
                    d2 = datetime.fromisoformat(prev_date[:10])
                    return (d1 - d2).days

        return None

    def recent_form(
        self, team: str, match_index: int, window: int = 5
    ) -> tuple[float, float]:
        """Rolling win rate and average margin over last `window` matches."""
        wins = 0
        margins = []

        for i in range(match_index - 1, -1, -1):
            if len(margins) >= window:
                break
            m = self.matches[i]
            home = m.get("home_team")
            away = m.get("away_team")
            hs = m.get("home_score", 0)
            as_ = m.get("away_score", 0)

            if team == home:
                margin = hs - as_
            elif team == away:
                margin = as_ - hs
            else:
                continue

            margins.append(margin)
            if margin > 0:
                wins += 1

        if not margins:
            return 0.5, 0.0

        return wins / len(margins), sum(margins) / len(margins)

    def head_to_head(
        self, team_a: str, team_b: str, match_index: int, window: int = 10
    ) -> float:
        """Win rate of team_a against team_b in their last `window` meetings."""
        wins = 0
        total = 0

        for i in range(match_index - 1, -1, -1):
            if total >= window:
                break
            m = self.matches[i]
            home = m.get("home_team")
            away = m.get("away_team")
            hs = m.get("home_score", 0)
            as_ = m.get("away_score", 0)

            if {home, away} != {team_a, team_b}:
                continue

            total += 1
            if team_a == home and hs > as_:
                wins += 1
            elif team_a == away and as_ > hs:
                wins += 1

        if total == 0:
            return 0.5

        return wins / total

    @staticmethod
    def travel_distance(city_a: str, city_b: str) -> float:
        """Approximate distance in km between two cities using Haversine."""
        if city_a == city_b:
            return 0.0

        coords_a = CITY_COORDS.get(city_a)
        coords_b = CITY_COORDS.get(city_b)
        if not coords_a or not coords_b:
            return 0.0

        lat1, lon1 = math.radians(coords_a[0]), math.radians(coords_a[1])
        lat2, lon2 = math.radians(coords_b[0]), math.radians(coords_b[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth radius in km
        return r * c

    @staticmethod
    def is_home(team: str, match: dict) -> bool:
        return match.get("home_team") == team
```

**Step 4: Run tests**

```bash
poetry run pytest tests/features/test_contextual.py -v
```
Expected: PASS

**Step 5: Commit**

```
feat: add contextual features (rest, form, H2H, travel)
```

---

## Task 9: Feature Builder

**Files:**
- Create: `src/netball_model/features/builder.py`
- Create: `tests/features/test_builder.py`

**Step 1: Write the failing test**

`tests/features/test_builder.py`:
```python
import pandas as pd

from netball_model.features.builder import FeatureBuilder


def test_build_feature_row():
    matches = [
        {
            "match_id": "1", "date": "2024-03-23", "round_num": 1,
            "home_team": "Firebirds", "away_team": "Swifts",
            "home_score": 60, "away_score": 55, "venue": "Brisbane Entertainment Centre",
        },
        {
            "match_id": "2", "date": "2024-03-30", "round_num": 2,
            "home_team": "Swifts", "away_team": "Firebirds",
            "home_score": 58, "away_score": 52, "venue": "Ken Rosewall Arena",
        },
    ]
    builder = FeatureBuilder(matches)
    row = builder.build_row(match_index=1)

    assert "home_elo" in row
    assert "away_elo" in row
    assert "home_rest_days" in row
    assert "away_rest_days" in row
    assert "home_form_win_rate" in row
    assert "home_form_avg_margin" in row
    assert "h2h_home_win_rate" in row
    assert "elo_diff" in row
    assert "margin" in row  # target
    assert row["margin"] == 58 - 52  # home - away for match index 1


def test_build_matrix():
    matches = [
        {
            "match_id": str(i), "date": f"2024-03-{20+i:02d}", "round_num": i,
            "home_team": "A", "away_team": "B",
            "home_score": 55 + i, "away_score": 50, "venue": "Brisbane Entertainment Centre",
        }
        for i in range(1, 6)
    ]
    builder = FeatureBuilder(matches)
    df = builder.build_matrix(start_index=1)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # skip first match (no prior data)
    assert "margin" in df.columns
    assert "home_elo" in df.columns
```

**Step 2: Run to verify failure**

```bash
poetry run pytest tests/features/test_builder.py -v
```

**Step 3: Implement FeatureBuilder**

`src/netball_model/features/builder.py`:
```python
from __future__ import annotations

import pandas as pd

from netball_model.features.contextual import (
    ContextualFeatures,
    TEAM_HOME_CITY,
    VENUE_TO_CITY,
)
from netball_model.features.elo import GlickoSystem


class FeatureBuilder:
    def __init__(self, matches: list[dict], pool: str = "ssn"):
        self.matches = matches
        self.pool = pool
        self.glicko = GlickoSystem()
        self.ctx = ContextualFeatures(matches)
        self._elo_computed_up_to = -1

    def _ensure_elo_up_to(self, match_index: int):
        """Replay matches to update Elo up to (but not including) match_index."""
        for i in range(self._elo_computed_up_to + 1, match_index):
            m = self.matches[i]
            hs = m.get("home_score", 0)
            as_ = m.get("away_score", 0)
            if hs > as_:
                winner = "home"
            elif as_ > hs:
                winner = "away"
            else:
                winner = "draw"
            self.glicko.update(
                m["home_team"], m["away_team"],
                winner=winner, margin=hs - as_, pool=self.pool,
            )
        self._elo_computed_up_to = match_index - 1

    def build_row(self, match_index: int) -> dict:
        self._ensure_elo_up_to(match_index)

        m = self.matches[match_index]
        home = m["home_team"]
        away = m["away_team"]

        home_elo = self.glicko.get_rating(home, self.pool)
        away_elo = self.glicko.get_rating(away, self.pool)

        home_rest = self.ctx.rest_days(home, match_index)
        away_rest = self.ctx.rest_days(away, match_index)

        home_form_wr, home_form_margin = self.ctx.recent_form(home, match_index)
        away_form_wr, away_form_margin = self.ctx.recent_form(away, match_index)

        h2h = self.ctx.head_to_head(home, away, match_index)

        venue = m.get("venue", "")
        venue_city = VENUE_TO_CITY.get(venue, "")
        home_city = TEAM_HOME_CITY.get(home, "")
        away_city = TEAM_HOME_CITY.get(away, "")
        home_travel = self.ctx.travel_distance(home_city, venue_city) if home_city and venue_city else 0
        away_travel = self.ctx.travel_distance(away_city, venue_city) if away_city and venue_city else 0

        win_prob = self.glicko.predict_win_prob(home, away, self.pool)

        hs = m.get("home_score", 0)
        as_ = m.get("away_score", 0)

        return {
            "match_id": m["match_id"],
            "home_team": home,
            "away_team": away,
            "home_elo": home_elo["rating"],
            "away_elo": away_elo["rating"],
            "home_elo_rd": home_elo["rd"],
            "away_elo_rd": away_elo["rd"],
            "elo_diff": home_elo["rating"] - away_elo["rating"],
            "elo_win_prob": win_prob,
            "home_rest_days": home_rest if home_rest is not None else 7,
            "away_rest_days": away_rest if away_rest is not None else 7,
            "rest_diff": (home_rest or 7) - (away_rest or 7),
            "home_form_win_rate": home_form_wr,
            "away_form_win_rate": away_form_wr,
            "home_form_avg_margin": home_form_margin,
            "away_form_avg_margin": away_form_margin,
            "h2h_home_win_rate": h2h,
            "home_travel_km": home_travel,
            "away_travel_km": away_travel,
            "travel_diff": away_travel - home_travel,
            "margin": hs - as_,
            "total_goals": hs + as_,
        }

    def build_matrix(self, start_index: int = 1) -> pd.DataFrame:
        rows = []
        for i in range(start_index, len(self.matches)):
            rows.append(self.build_row(i))
        return pd.DataFrame(rows)
```

**Step 4: Run tests**

```bash
poetry run pytest tests/features/test_builder.py -v
```
Expected: PASS

**Step 5: Commit**

```
feat: add feature builder combining Elo + contextual features
```

---

## Task 10: Model Training + Calibration

**Files:**
- Create: `src/netball_model/model/train.py`
- Create: `src/netball_model/model/calibration.py`
- Create: `tests/model/test_train.py`

**Step 1: Write the failing tests**

`tests/model/test_train.py`:
```python
import numpy as np
import pandas as pd

from netball_model.model.train import NetballModel
from netball_model.model.calibration import CalibrationModel


def _make_dummy_df(n=100):
    rng = np.random.default_rng(42)
    elo_diff = rng.normal(0, 100, n)
    noise = rng.normal(0, 8, n)
    margin = 0.05 * elo_diff + noise
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)],
        "home_team": ["A"] * n,
        "away_team": ["B"] * n,
        "elo_diff": elo_diff,
        "home_elo": 1500 + elo_diff / 2,
        "away_elo": 1500 - elo_diff / 2,
        "home_elo_rd": [200.0] * n,
        "away_elo_rd": [200.0] * n,
        "elo_win_prob": [0.5] * n,
        "home_rest_days": rng.integers(5, 10, n),
        "away_rest_days": rng.integers(5, 10, n),
        "rest_diff": [0] * n,
        "home_form_win_rate": rng.uniform(0.3, 0.7, n),
        "away_form_win_rate": rng.uniform(0.3, 0.7, n),
        "home_form_avg_margin": rng.normal(0, 5, n),
        "away_form_avg_margin": rng.normal(0, 5, n),
        "h2h_home_win_rate": [0.5] * n,
        "home_travel_km": [0.0] * n,
        "away_travel_km": rng.uniform(0, 3000, n),
        "travel_diff": rng.uniform(0, 3000, n),
        "margin": margin,
        "total_goals": rng.integers(90, 130, n).astype(float),
    })


def test_train_and_predict():
    df = _make_dummy_df(100)
    model = NetballModel()
    model.train(df)

    pred = model.predict(df.iloc[[0]])
    assert "predicted_margin" in pred.columns
    assert "predicted_total" in pred.columns
    assert len(pred) == 1


def test_feature_columns_excludes_targets():
    model = NetballModel()
    df = _make_dummy_df(50)
    model.train(df)
    assert "margin" not in model.feature_columns
    assert "total_goals" not in model.feature_columns
    assert "match_id" not in model.feature_columns
    assert "home_team" not in model.feature_columns


def test_calibration():
    residuals = np.random.normal(0, 10, 200)
    cal = CalibrationModel()
    cal.fit(residuals)

    # Predicted margin of +5 with std 10 should give >50% win prob
    prob = cal.win_probability(predicted_margin=5.0)
    assert prob > 0.5
    assert prob < 1.0

    # Predicted margin of 0 should give ~50%
    prob_even = cal.win_probability(predicted_margin=0.0)
    assert abs(prob_even - 0.5) < 0.05
```

**Step 2: Run to verify failure**

```bash
poetry run pytest tests/model/test_train.py -v
```

**Step 3: Implement NetballModel**

`src/netball_model/model/train.py`:
```python
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from netball_model.model.calibration import CalibrationModel

NON_FEATURE_COLUMNS = {
    "match_id", "home_team", "away_team", "margin", "total_goals",
}


class NetballModel:
    def __init__(self, alpha: float = 1.0):
        self.margin_model = Ridge(alpha=alpha)
        self.total_model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.calibration = CalibrationModel()
        self.feature_columns: list[str] = []

    def train(self, df: pd.DataFrame):
        self.feature_columns = [
            c for c in df.columns if c not in NON_FEATURE_COLUMNS
        ]

        X = df[self.feature_columns].values.astype(float)
        y_margin = df["margin"].values.astype(float)
        y_total = df["total_goals"].values.astype(float)

        X_scaled = self.scaler.fit_transform(X)

        self.margin_model.fit(X_scaled, y_margin)
        self.total_model.fit(X_scaled, y_total)

        # Calibrate on training residuals
        margin_preds = self.margin_model.predict(X_scaled)
        residuals = y_margin - margin_preds
        self.calibration.fit(residuals)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].values.astype(float)
        X_scaled = self.scaler.transform(X)

        margins = self.margin_model.predict(X_scaled)
        totals = self.total_model.predict(X_scaled)
        win_probs = [self.calibration.win_probability(m) for m in margins]

        result = df[["match_id", "home_team", "away_team"]].copy()
        result["predicted_margin"] = np.round(margins, 1)
        result["predicted_total"] = np.round(totals, 1)
        result["win_probability"] = np.round(win_probs, 4)
        return result

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> NetballModel:
        with open(path, "rb") as f:
            return pickle.load(f)
```

`src/netball_model/model/calibration.py`:
```python
from __future__ import annotations

import numpy as np
from scipy.stats import norm


class CalibrationModel:
    def __init__(self):
        self.residual_std: float = 10.0  # default

    def fit(self, residuals: np.ndarray):
        self.residual_std = float(np.std(residuals))

    def win_probability(self, predicted_margin: float) -> float:
        """P(actual_margin > 0) given predicted_margin."""
        return float(norm.cdf(predicted_margin / self.residual_std))
```

**Step 4: Add scipy to dependencies**

In `pyproject.toml`, add `scipy = "^1.12"` under `[tool.poetry.dependencies]`.

```bash
poetry add scipy
```

**Step 5: Run tests**

```bash
poetry run pytest tests/model/test_train.py -v
```
Expected: PASS

**Step 6: Commit**

```
feat: add Ridge regression model with probability calibration
```

---

## Task 11: Value Detector

**Files:**
- Create: `src/netball_model/value/detector.py`
- Create: `tests/value/test_detector.py`

**Step 1: Write the failing test**

`tests/value/test_detector.py`:
```python
from netball_model.value.detector import ValueDetector


def test_detect_value_home():
    detector = ValueDetector(min_edge=0.05)

    result = detector.evaluate(
        home_team="Firebirds",
        away_team="Swifts",
        model_win_prob=0.65,
        betfair_home_back=1.80,  # implied prob = 55.6%
    )

    assert result["model_prob"] == 0.65
    assert abs(result["implied_prob"] - 1 / 1.80) < 0.01
    assert result["edge"] > 0.05
    assert result["is_value"] is True
    assert result["bet_side"] == "home"


def test_no_value():
    detector = ValueDetector(min_edge=0.05)

    result = detector.evaluate(
        home_team="Firebirds",
        away_team="Swifts",
        model_win_prob=0.55,
        betfair_home_back=1.75,  # implied prob = 57.1%
    )

    assert result["is_value"] is False


def test_value_on_away():
    detector = ValueDetector(min_edge=0.05)

    result = detector.evaluate(
        home_team="Firebirds",
        away_team="Swifts",
        model_win_prob=0.35,
        betfair_home_back=1.60,  # implied home prob = 62.5%, away implied = 37.5%
        betfair_away_back=2.50,  # implied away prob = 40%
    )

    # Model says away win prob = 0.65, implied away = 40%
    assert result["bet_side"] == "away"
    assert result["is_value"] is True
```

**Step 2: Run to verify failure**

```bash
poetry run pytest tests/value/test_detector.py -v
```

**Step 3: Implement ValueDetector**

`src/netball_model/value/detector.py`:
```python
from __future__ import annotations


class ValueDetector:
    def __init__(self, min_edge: float = 0.05):
        self.min_edge = min_edge

    def evaluate(
        self,
        home_team: str,
        away_team: str,
        model_win_prob: float,
        betfair_home_back: float | None = None,
        betfair_away_back: float | None = None,
    ) -> dict:
        model_away_prob = 1.0 - model_win_prob

        home_implied = 1 / betfair_home_back if betfair_home_back else None
        away_implied = 1 / betfair_away_back if betfair_away_back else None

        home_edge = (model_win_prob - home_implied) if home_implied else None
        away_edge = (model_away_prob - away_implied) if away_implied else None

        # Pick the side with the bigger edge
        best_side = "home"
        best_edge = home_edge or 0
        best_model_prob = model_win_prob
        best_implied = home_implied or 0
        best_odds = betfair_home_back

        if away_edge is not None and away_edge > (home_edge or 0):
            best_side = "away"
            best_edge = away_edge
            best_model_prob = model_away_prob
            best_implied = away_implied or 0
            best_odds = betfair_away_back

        return {
            "home_team": home_team,
            "away_team": away_team,
            "bet_side": best_side,
            "model_prob": best_model_prob,
            "implied_prob": best_implied,
            "edge": round(best_edge, 4),
            "odds": best_odds,
            "is_value": best_edge >= self.min_edge,
        }
```

**Step 4: Run tests**

```bash
poetry run pytest tests/value/test_detector.py -v
```
Expected: PASS

**Step 5: Commit**

```
feat: add value detector comparing model vs exchange odds
```

---

## Task 12: Train CLI Command

**Files:**
- Modify: `src/netball_model/cli.py`

**Step 1: Add train command**

```python
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel

@main.command()
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--output", default="data/model.pkl", help="Model output path")
def train(db: str, output: str):
    """Backfill Elo, build features, train model."""
    db_conn = Database(db)
    matches = db_conn.get_matches()

    if len(matches) < 20:
        click.echo(f"Only {len(matches)} matches in DB. Need at least 20 to train.")
        return

    click.echo(f"Building features from {len(matches)} matches...")
    builder = FeatureBuilder(matches)
    df = builder.build_matrix(start_index=1)

    click.echo(f"Training on {len(df)} rows, {len(df.columns)} features...")
    model = NetballModel()
    model.train(df)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    click.echo(f"Model saved to {output}")

    # Print training summary
    preds = model.predict(df)
    mae = (df["margin"] - preds["predicted_margin"].astype(float)).abs().mean()
    click.echo(f"Training MAE: {mae:.1f} goals")
```

Add `from pathlib import Path` to imports.

**Step 2: Commit**

```
feat: add train CLI command
```

---

## Task 13: Predict CLI Command

**Files:**
- Modify: `src/netball_model/cli.py`
- Create: `src/netball_model/model/predict.py`

**Step 1: Implement predict module**

`src/netball_model/model/predict.py`:
```python
from __future__ import annotations

from rich.console import Console
from rich.table import Table


def display_predictions(predictions: list[dict]):
    """Render predictions as a rich table in the terminal."""
    console = Console()
    table = Table(title="Match Predictions", show_lines=True)

    table.add_column("Match", style="bold")
    table.add_column("Pred Margin", justify="right")
    table.add_column("Pred Total", justify="right")
    table.add_column("Model Win%", justify="right")
    table.add_column("Betfair Odds", justify="right")
    table.add_column("Implied%", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("Value?", justify="center")

    for p in predictions:
        edge_str = f"{p.get('edge', 0):.1%}"
        value_str = "YES" if p.get("is_value") else "-"
        value_style = "bold green" if p.get("is_value") else "dim"

        odds_str = f"{p.get('odds', '-')}" if p.get("odds") else "-"
        implied_str = f"{p.get('implied_prob', 0):.1%}" if p.get("implied_prob") else "-"

        table.add_row(
            f"{p['home_team']} v {p['away_team']}",
            f"{p['predicted_margin']:+.1f}",
            f"{p['predicted_total']:.0f}",
            f"{p['win_probability']:.1%}",
            odds_str,
            implied_str,
            edge_str,
            f"[{value_style}]{value_str}[/{value_style}]",
        )

    console.print(table)
```

**Step 2: Add predict command to CLI**

```python
from netball_model.model.predict import display_predictions
from netball_model.value.detector import ValueDetector

@main.command()
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--model-path", default="data/model.pkl", help="Model file path")
@click.option("--min-edge", default=0.05, type=float, help="Minimum edge threshold")
def predict(db: str, model_path: str, min_edge: float):
    """Predict upcoming matches and flag value bets."""
    db_conn = Database(db)
    model = NetballModel.load(model_path)
    detector = ValueDetector(min_edge=min_edge)

    matches = db_conn.get_matches()
    # Find matches without scores (upcoming)
    upcoming = [m for m in matches if m.get("home_score") is None]

    if not upcoming:
        click.echo("No upcoming matches found. Run ingest first.")
        return

    builder = FeatureBuilder(matches)
    results = []

    for i, m in enumerate(matches):
        if m["match_id"] not in {u["match_id"] for u in upcoming}:
            continue

        row = builder.build_row(i)
        import pandas as pd
        pred = model.predict(pd.DataFrame([row]))

        value = detector.evaluate(
            home_team=m["home_team"],
            away_team=m["away_team"],
            model_win_prob=float(pred["win_probability"].iloc[0]),
        )

        results.append({
            **value,
            "predicted_margin": float(pred["predicted_margin"].iloc[0]),
            "predicted_total": float(pred["predicted_total"].iloc[0]),
            "win_probability": float(pred["win_probability"].iloc[0]),
        })

    display_predictions(results)
```

**Step 3: Commit**

```
feat: add predict CLI with rich table output
```

---

## Task 14: Backtest Command

**Files:**
- Modify: `src/netball_model/cli.py`

**Step 1: Implement backtest command**

```python
@main.command()
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--train-seasons", required=True, help="Training seasons range (e.g. 2017-2023)")
@click.option("--test-season", required=True, type=int, help="Test season year")
def backtest(db: str, train_seasons: str, test_season: int):
    """Walk-forward backtest on a held-out season."""
    db_conn = Database(db)
    all_matches = db_conn.get_matches()

    start, end = map(int, train_seasons.split("-"))
    train_matches = [m for m in all_matches if start <= m["season"] <= end]
    test_matches = [m for m in all_matches if m["season"] == test_season]

    if not train_matches or not test_matches:
        click.echo("Insufficient data for backtest.")
        return

    click.echo(f"Training on {len(train_matches)} matches ({train_seasons})")
    click.echo(f"Testing on {len(test_matches)} matches ({test_season})")

    builder = FeatureBuilder(train_matches)
    train_df = builder.build_matrix(start_index=1)

    model = NetballModel()
    model.train(train_df)

    # Walk-forward: for each test match, build features using all prior matches
    all_for_test = train_matches + test_matches
    test_builder = FeatureBuilder(all_for_test)

    correct = 0
    total = 0
    abs_errors = []

    for i in range(len(train_matches), len(all_for_test)):
        row = test_builder.build_row(i)
        import pandas as pd
        pred = model.predict(pd.DataFrame([row]))

        pred_margin = float(pred["predicted_margin"].iloc[0])
        actual_margin = row["margin"]

        abs_errors.append(abs(pred_margin - actual_margin))
        if (pred_margin > 0 and actual_margin > 0) or (pred_margin < 0 and actual_margin < 0):
            correct += 1
        total += 1

    import numpy as np
    mae = np.mean(abs_errors)
    accuracy = correct / total if total > 0 else 0

    click.echo(f"\nBacktest Results ({test_season}):")
    click.echo(f"  Matches: {total}")
    click.echo(f"  Win/Loss Accuracy: {accuracy:.1%}")
    click.echo(f"  Mean Absolute Error: {mae:.1f} goals")
```

**Step 2: Commit**

```
feat: add walk-forward backtest command
```

---

## Task 15: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

`tests/test_integration.py`:
```python
"""End-to-end test using mock data: ingest -> train -> predict."""
import numpy as np

from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel
from netball_model.value.detector import ValueDetector


def _seed_matches(db: Database, n: int = 50):
    """Create synthetic matches."""
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
        match = {
            "match_id": f"test_{i:03d}",
            "competition_id": 99999,
            "season": 2024,
            "round_num": (i // 4) + 1,
            "game_num": (i % 4) + 1,
            "date": f"2024-{(i // 4) + 3:02d}-{(i % 28) + 1:02d}",
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
        }
        db.upsert_match(match)
        matches.append(match)
    return matches


def test_end_to_end(tmp_db):
    # 1. Setup DB + seed data
    db = Database(tmp_db)
    db.initialize()
    matches = _seed_matches(db, n=50)

    # 2. Build features
    builder = FeatureBuilder(matches)
    df = builder.build_matrix(start_index=1)
    assert len(df) == 49

    # 3. Train model
    model = NetballModel()
    model.train(df)

    # 4. Predict on last match
    last_row = builder.build_row(len(matches) - 1)
    import pandas as pd
    pred = model.predict(pd.DataFrame([last_row]))
    assert "predicted_margin" in pred.columns
    assert "win_probability" in pred.columns

    # 5. Value detection
    detector = ValueDetector(min_edge=0.05)
    result = detector.evaluate(
        home_team=matches[-1]["home_team"],
        away_team=matches[-1]["away_team"],
        model_win_prob=float(pred["win_probability"].iloc[0]),
        betfair_home_back=1.80,
    )
    assert "is_value" in result
    assert isinstance(result["is_value"], bool)

    # 6. Save/load model roundtrip
    model_path = tmp_db.parent / "model.pkl"
    model.save(model_path)
    loaded = NetballModel.load(model_path)
    pred2 = loaded.predict(pd.DataFrame([last_row]))
    assert float(pred2["predicted_margin"].iloc[0]) == float(pred["predicted_margin"].iloc[0])
```

**Step 2: Run the full test suite**

```bash
poetry run pytest tests/ -v
```
Expected: ALL PASS

**Step 3: Commit**

```
feat: add end-to-end integration test
```

---

## Task 16: Update .gitignore and Final Cleanup

**Step 1: Add project-specific entries to .gitignore**

Append:
```
# Project data
data/*.db
data/*.pkl
data/*.tar
data/*.json
```

**Step 2: Run full test suite one final time**

```bash
poetry run pytest tests/ -v --tb=short
```

**Step 3: Commit**

```
chore: update gitignore for project data files
```

---

## Summary

| Task | Component | Key Deliverable |
|------|-----------|----------------|
| 1 | Scaffolding | Poetry project, directory structure, CLI skeleton |
| 2 | Database | SQLite schema + CRUD (matches, players, odds, elo) |
| 3 | Champion Data Client | API client parsing match + player JSON |
| 4 | Ingest CLI | `netball ingest --season 2024` |
| 5 | Betfair Parser | TAR/JSON historical odds parser |
| 6 | Odds CLI | `netball odds --file betfair.tar` |
| 7 | Glicko-2 | Rating system with MOV scaling + separate pools |
| 8 | Contextual Features | Rest, form, H2H, travel distance |
| 9 | Feature Builder | Combine Elo + contextual into feature matrix |
| 10 | Model Training | Ridge regression + probability calibration |
| 11 | Value Detector | Model prob vs exchange odds comparison |
| 12 | Train CLI | `netball train` |
| 13 | Predict CLI | `netball predict` with rich table |
| 14 | Backtest CLI | `netball backtest --train-seasons 2017-2023 --test-season 2024` |
| 15 | Integration Test | End-to-end: seed -> features -> train -> predict -> value |
| 16 | Cleanup | Gitignore, final test pass |
