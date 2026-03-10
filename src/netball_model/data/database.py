from __future__ import annotations

import sqlite3
from contextlib import contextmanager
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

    @contextmanager
    def connection(self):
        """Yield a connection that commits on success and always closes."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self):
        with self.connection() as conn:
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

    _UPSERT_MATCH_SQL = """
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
    """

    def upsert_match(self, match: dict):
        with self.connection() as conn:
            conn.execute(self._UPSERT_MATCH_SQL, match)

    def upsert_matches(self, matches: list[dict]):
        """Batch upsert matches in a single transaction."""
        with self.connection() as conn:
            conn.executemany(self._UPSERT_MATCH_SQL, matches)

    _INSERT_PLAYER_STATS_SQL = """
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
    """

    def insert_player_stats(self, stats: dict):
        with self.connection() as conn:
            conn.execute(self._INSERT_PLAYER_STATS_SQL, stats)

    def insert_player_stats_batch(self, stats: list[dict]):
        """Batch insert player stats in a single transaction."""
        with self.connection() as conn:
            conn.executemany(self._INSERT_PLAYER_STATS_SQL, stats)

    def get_matches(self, season: int | None = None) -> list[dict]:
        with self.connection() as conn:
            if season:
                cursor = conn.execute(
                    "SELECT * FROM matches WHERE season = ? ORDER BY date", (season,)
                )
            else:
                cursor = conn.execute("SELECT * FROM matches ORDER BY date")
            return [dict(row) for row in cursor.fetchall()]

    def get_player_stats(self, match_id: str) -> list[dict]:
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM player_stats WHERE match_id = ?", (match_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_player_history(
        self, player_id: int, before_date: str, limit: int = 5
    ) -> list[dict]:
        """Return up to `limit` most recent stat rows for a player before `before_date`."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT ps.* FROM player_stats ps
                JOIN matches m ON ps.match_id = m.match_id
                WHERE ps.player_id = ? AND m.date < ?
                ORDER BY m.date DESC
                LIMIT ?
                """,
                (player_id, before_date, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_starters_for_match(self, match_id: str) -> list[dict]:
        """Return starting 7 per team (excludes substitutes with position '-')."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM player_stats WHERE match_id = ? AND position != '-'",
                (match_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    _UPSERT_ODDS_SQL = """
        INSERT OR REPLACE INTO odds_history (
            match_id, source, home_back_odds, home_lay_odds,
            away_back_odds, away_lay_odds,
            home_volume, away_volume, timestamp
        ) VALUES (
            :match_id, :source, :home_back_odds, :home_lay_odds,
            :away_back_odds, :away_lay_odds,
            :home_volume, :away_volume, :timestamp
        )
    """

    def upsert_odds(self, odds: dict):
        with self.connection() as conn:
            conn.execute(self._UPSERT_ODDS_SQL, odds)

    def upsert_odds_batch(self, odds_list: list[dict]):
        """Batch upsert odds records in a single transaction."""
        with self.connection() as conn:
            conn.executemany(self._UPSERT_ODDS_SQL, odds_list)

    def get_odds(self, source: str | None = None) -> list[dict]:
        """Return all odds records, optionally filtered by source."""
        with self.connection() as conn:
            if source:
                cursor = conn.execute(
                    "SELECT * FROM odds_history WHERE source = ? ORDER BY match_id",
                    (source,),
                )
            else:
                cursor = conn.execute("SELECT * FROM odds_history ORDER BY match_id")
            return [dict(row) for row in cursor.fetchall()]

    def upsert_elo(self, elo: dict):
        with self.connection() as conn:
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

    def get_latest_elo(self, team: str, pool: str = "ssn") -> dict | None:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM elo_ratings
                WHERE team = ? AND pool = ?
                ORDER BY id DESC LIMIT 1
                """,
                (team, pool),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
