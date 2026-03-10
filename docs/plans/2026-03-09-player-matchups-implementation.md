# Player Matchup Features — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add position-pair difference scores from rolling player profiles as new features to the Ridge regression model, and add flat-betting evaluation alongside Kelly.

**Architecture:** New `PlayerProfiler` class computes rolling 5-match stat averages per player. New `MatchupFeatures` class resolves starting lineups and computes positional difference scores. `FeatureBuilder` gains an optional `player_stats` dict and delegates to `MatchupFeatures`. Notebook adds flat-betting strategy and baseline vs matchup comparison.

**Tech Stack:** Python, pandas, sqlite3, scikit-learn (existing Ridge), pytest

**Design doc:** `docs/plans/2026-03-09-player-matchups-design.md`

---

### Task 1: Database — Add `get_player_history` Query

**Files:**
- Modify: `src/netball_model/data/database.py:175-180`
- Test: `tests/data/test_database.py`

**Step 1: Write the failing test**

Add to `tests/data/test_database.py`:

```python
def test_get_player_history(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    # Insert 3 matches on different dates
    for i, date in enumerate(["2024-03-01", "2024-03-08", "2024-03-15"]):
        db.upsert_match({
            "match_id": f"m_{i}", "competition_id": 1, "season": 2024,
            "round_num": i + 1, "game_num": 1, "date": date,
            "venue": "Test", "home_team": "A", "away_team": "B",
            "home_score": 55, "away_score": 50,
            "home_q1": 14, "home_q2": 14, "home_q3": 14, "home_q4": 13,
            "away_q1": 13, "away_q2": 13, "away_q3": 12, "away_q4": 12,
        })
        db.insert_player_stats({
            "match_id": f"m_{i}", "player_id": 100, "player_name": "",
            "team": "A", "position": "GS",
            "goals": 30 + i, "attempts": 35, "assists": 2, "rebounds": 3,
            "feeds": 4, "turnovers": 2, "gains": 0, "intercepts": 0,
            "deflections": 0, "penalties": 1, "centre_pass_receives": 0,
            "net_points": 0.0,
        })

    # Before the 3rd match, should get 2 rows (matches 0 and 1)
    history = db.get_player_history(player_id=100, before_date="2024-03-15", limit=5)
    assert len(history) == 2
    # Most recent first
    assert history[0]["goals"] == 31  # match 1 (2024-03-08)
    assert history[1]["goals"] == 30  # match 0 (2024-03-01)

    # With limit=1, should get only 1 row
    history = db.get_player_history(player_id=100, before_date="2024-03-15", limit=1)
    assert len(history) == 1
    assert history[0]["goals"] == 31
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/data/test_database.py::test_get_player_history -v`
Expected: FAIL with `AttributeError: 'Database' object has no attribute 'get_player_history'`

**Step 3: Write minimal implementation**

Add to `src/netball_model/data/database.py` after `get_player_stats` (after line 180):

```python
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
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/data/test_database.py::test_get_player_history -v`
Expected: PASS

**Step 5: Commit**

Message: `feat(db): add get_player_history query for rolling player profiles`

---

### Task 2: Database — Add `get_starters_for_match` Query

**Files:**
- Modify: `src/netball_model/data/database.py`
- Test: `tests/data/test_database.py`

**Step 1: Write the failing test**

Add to `tests/data/test_database.py`:

```python
def test_get_starters_for_match(tmp_db):
    db = Database(tmp_db)
    db.initialize()

    db.upsert_match({
        "match_id": "m_01", "competition_id": 1, "season": 2024,
        "round_num": 1, "game_num": 1, "date": "2024-03-01",
        "venue": "Test", "home_team": "A", "away_team": "B",
        "home_score": 55, "away_score": 50,
        "home_q1": 14, "home_q2": 14, "home_q3": 14, "home_q4": 13,
        "away_q1": 13, "away_q2": 13, "away_q3": 12, "away_q4": 12,
    })

    positions = ["GS", "GA", "WA", "C", "WD", "GD", "GK"]
    for team_idx, team in enumerate(["A", "B"]):
        for pos_idx, pos in enumerate(positions):
            db.insert_player_stats({
                "match_id": "m_01", "player_id": team_idx * 100 + pos_idx,
                "player_name": "", "team": team, "position": pos,
                "goals": 10, "attempts": 12, "assists": 1, "rebounds": 2,
                "feeds": 3, "turnovers": 1, "gains": 0, "intercepts": 0,
                "deflections": 0, "penalties": 1, "centre_pass_receives": 0,
                "net_points": 0.0,
            })
        # Add a substitute (position = "-")
        db.insert_player_stats({
            "match_id": "m_01", "player_id": team_idx * 100 + 50,
            "player_name": "", "team": team, "position": "-",
            "goals": 0, "attempts": 0, "assists": 0, "rebounds": 0,
            "feeds": 0, "turnovers": 0, "gains": 0, "intercepts": 0,
            "deflections": 0, "penalties": 0, "centre_pass_receives": 0,
            "net_points": 0.0,
        })

    starters = db.get_starters_for_match("m_01")
    assert len(starters) == 14  # 7 per team, no subs
    positions_found = {s["position"] for s in starters}
    assert "-" not in positions_found
    assert positions_found == set(positions)
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/data/test_database.py::test_get_starters_for_match -v`
Expected: FAIL with `AttributeError`

**Step 3: Write minimal implementation**

Add to `src/netball_model/data/database.py` after `get_player_history`:

```python
def get_starters_for_match(self, match_id: str) -> list[dict]:
    """Return starting 7 per team (excludes substitutes with position '-')."""
    with self.connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM player_stats WHERE match_id = ? AND position != '-'",
            (match_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/data/test_database.py::test_get_starters_for_match -v`
Expected: PASS

**Step 5: Commit**

Message: `feat(db): add get_starters_for_match query excluding substitutes`

---

### Task 3: PlayerProfiler — Rolling Averages + Derived Ratios

**Files:**
- Create: `src/netball_model/features/player_profile.py`
- Create: `tests/features/test_player_profile.py`

**Step 1: Write the failing tests**

Create `tests/features/test_player_profile.py`:

```python
from netball_model.features.player_profile import PlayerProfiler

# Positions where each derived ratio applies
SHOOTER_POSITIONS = {"GS", "GA"}
DEFENDER_POSITIONS = {"GD", "GK", "WD"}
FEEDER_POSITIONS = {"WA", "C", "GA"}


def _make_stat_rows(n=5, goals=30, attempts=35, feeds=10, turnovers=2,
                    gains=3, intercepts=2, position="GS"):
    """Create n stat rows for testing."""
    return [
        {
            "player_id": 100, "position": position, "team": "A",
            "goals": goals, "attempts": attempts, "assists": 5,
            "rebounds": 3, "feeds": feeds, "turnovers": turnovers,
            "gains": gains, "intercepts": intercepts, "deflections": 1,
            "penalties": 2, "centre_pass_receives": 4,
        }
        for _ in range(n)
    ]


def test_compute_profile_basic():
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=5, goals=30, attempts=35)
    profile = profiler.compute_profile(rows, position="GS")

    assert profile["goals"] == 30.0
    assert profile["attempts"] == 35.0
    assert profile["assists"] == 5.0
    assert profile["matches_used"] == 5


def test_compute_profile_shooting_pct():
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=3, goals=30, attempts=40, position="GS")
    profile = profiler.compute_profile(rows, position="GS")

    assert abs(profile["shooting_pct"] - 0.75) < 0.001


def test_compute_profile_clean_steal_rate():
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=3, gains=4, intercepts=2, position="GD")
    profile = profiler.compute_profile(rows, position="GD")

    assert abs(profile["clean_steal_rate"] - 0.5) < 0.001


def test_compute_profile_delivery_efficiency():
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=3, feeds=20, turnovers=4, position="WA")
    profile = profiler.compute_profile(rows, position="WA")

    assert abs(profile["delivery_efficiency"] - 5.0) < 0.001


def test_compute_profile_zero_divisor():
    """Derived ratios should be 0 when denominator is 0."""
    profiler = PlayerProfiler()
    rows = _make_stat_rows(n=3, goals=0, attempts=0, position="GS")
    profile = profiler.compute_profile(rows, position="GS")

    assert profile["shooting_pct"] == 0.0


def test_compute_profile_empty():
    profiler = PlayerProfiler()
    profile = profiler.compute_profile([], position="GS")

    assert profile is None
```

**Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/features/test_player_profile.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/netball_model/features/player_profile.py`:

```python
"""Rolling player profile computation from historical stat rows."""
from __future__ import annotations

RAW_STATS = [
    "goals", "attempts", "assists", "rebounds", "feeds",
    "turnovers", "gains", "intercepts", "deflections",
    "penalties", "centre_pass_receives",
]


class PlayerProfiler:
    """Computes rolling average profiles from a player's recent stat rows."""

    def compute_profile(
        self, stat_rows: list[dict], position: str
    ) -> dict | None:
        """Compute a profile from up to N recent stat rows.

        Returns None if stat_rows is empty.
        """
        if not stat_rows:
            return None

        n = len(stat_rows)
        profile: dict = {"matches_used": n}

        for stat in RAW_STATS:
            total = sum(row.get(stat, 0) or 0 for row in stat_rows)
            profile[stat] = total / n

        # Derived ratios
        if position in ("GS", "GA"):
            profile["shooting_pct"] = (
                profile["goals"] / profile["attempts"]
                if profile["attempts"] > 0 else 0.0
            )
        if position in ("GD", "GK", "WD"):
            profile["clean_steal_rate"] = (
                profile["intercepts"] / profile["gains"]
                if profile["gains"] > 0 else 0.0
            )
        if position in ("WA", "C", "GA"):
            profile["delivery_efficiency"] = (
                profile["feeds"] / profile["turnovers"]
                if profile["turnovers"] > 0 else 0.0
            )

        return profile
```

**Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/features/test_player_profile.py -v`
Expected: 6 PASS

**Step 5: Commit**

Message: `feat: add PlayerProfiler for rolling player stat averages and derived ratios`

---

### Task 4: MatchupFeatures — Position Difference Scores

**Files:**
- Create: `src/netball_model/features/matchups.py`
- Create: `tests/features/test_matchups.py`

**Step 1: Write the failing tests**

Create `tests/features/test_matchups.py`:

```python
from netball_model.features.matchups import MatchupFeatures, MATCHUP_PAIRS


def _make_profile(position, goals=30, attempts=35, assists=5, feeds=10,
                  turnovers=2, gains=3, intercepts=2, deflections=1,
                  penalties=2, rebounds=3, centre_pass_receives=4,
                  shooting_pct=None, clean_steal_rate=None,
                  delivery_efficiency=None):
    """Build a fake player profile dict."""
    p = {
        "goals": goals, "attempts": attempts, "assists": assists,
        "feeds": feeds, "turnovers": turnovers, "gains": gains,
        "intercepts": intercepts, "deflections": deflections,
        "penalties": penalties, "rebounds": rebounds,
        "centre_pass_receives": centre_pass_receives,
        "matches_used": 5,
    }
    if position in ("GS", "GA"):
        p["shooting_pct"] = shooting_pct if shooting_pct is not None else (
            goals / attempts if attempts > 0 else 0.0
        )
    if position in ("GD", "GK", "WD"):
        p["clean_steal_rate"] = clean_steal_rate if clean_steal_rate is not None else (
            intercepts / gains if gains > 0 else 0.0
        )
    if position in ("WA", "C", "GA"):
        p["delivery_efficiency"] = delivery_efficiency if delivery_efficiency is not None else (
            feeds / turnovers if turnovers > 0 else 0.0
        )
    return p


def test_compute_matchup_features_gs_vs_gk():
    mf = MatchupFeatures()

    home_profiles = {"GS": _make_profile("GS", goals=35, attempts=40, shooting_pct=0.875)}
    away_profiles = {"GK": _make_profile("GK", gains=5, intercepts=3)}

    features = mf.compute_features(home_profiles, away_profiles)

    # GS vs GK uses: shooting_pct, goals, attempts, rebounds
    assert "gs_vs_gk_goals_diff" in features
    assert features["gs_vs_gk_goals_diff"] == 35 - 0  # GK goals default 0 in _make_profile
    assert "gs_vs_gk_shooting_pct_diff" in features


def test_compute_matchup_features_returns_all_pairs():
    mf = MatchupFeatures()

    positions = ["GS", "GA", "WA", "C", "WD", "GD", "GK"]
    home_profiles = {p: _make_profile(p) for p in positions}
    away_profiles = {p: _make_profile(p) for p in positions}

    features = mf.compute_features(home_profiles, away_profiles)

    # Should have features for all 5 matchup pairs
    for home_pos, away_pos, _ in MATCHUP_PAIRS:
        prefix = f"{home_pos.lower()}_vs_{away_pos.lower()}_"
        matching = [k for k in features if k.startswith(prefix)]
        assert len(matching) > 0, f"No features for {prefix}"


def test_compute_matchup_features_missing_position():
    """Missing positions should produce 0.0 for all their features."""
    mf = MatchupFeatures()

    # Only home GS present, no away GK
    home_profiles = {"GS": _make_profile("GS", goals=35)}
    away_profiles = {}

    features = mf.compute_features(home_profiles, away_profiles)

    # Should still have GS vs GK features, defaulting missing side to 0
    assert "gs_vs_gk_goals_diff" in features
```

**Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/features/test_matchups.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/netball_model/features/matchups.py`:

```python
"""Position-pair matchup features from player profiles."""
from __future__ import annotations

# (home_position, away_position, [stats to diff])
MATCHUP_PAIRS: list[tuple[str, str, list[str]]] = [
    ("GS", "GK", ["shooting_pct", "goals", "attempts", "rebounds"]),
    ("GA", "GD", ["shooting_pct", "goals", "assists", "feeds", "intercepts"]),
    ("WA", "WD", ["feeds", "assists", "turnovers", "centre_pass_receives", "deflections"]),
    ("C", "C", ["assists", "feeds", "turnovers", "gains", "intercepts"]),
    ("WD", "WA", ["gains", "intercepts", "deflections", "penalties"]),
]


class MatchupFeatures:
    """Compute position-pair difference scores from player profiles."""

    def compute_features(
        self,
        home_profiles: dict[str, dict],
        away_profiles: dict[str, dict],
    ) -> dict[str, float]:
        """Compute difference features for all matchup pairs.

        Each profile dict is keyed by position (e.g. "GS") with values
        from PlayerProfiler.compute_profile().

        Missing positions produce 0.0 for all their features.
        """
        features: dict[str, float] = {}

        for home_pos, away_pos, stats in MATCHUP_PAIRS:
            home_p = home_profiles.get(home_pos)
            away_p = away_profiles.get(away_pos)
            prefix = f"{home_pos.lower()}_vs_{away_pos.lower()}_"

            for stat in stats:
                home_val = home_p.get(stat, 0.0) if home_p else 0.0
                away_val = away_p.get(stat, 0.0) if away_p else 0.0
                features[f"{prefix}{stat}_diff"] = home_val - away_val

        return features
```

**Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/features/test_matchups.py -v`
Expected: 3 PASS

**Step 5: Commit**

Message: `feat: add MatchupFeatures for position-pair difference scores`

---

### Task 5: FeatureBuilder — Wire Player Matchup Features

**Files:**
- Modify: `src/netball_model/features/builder.py`
- Modify: `tests/features/test_builder.py`

**Step 1: Write the failing test**

Add to `tests/features/test_builder.py`:

```python
def test_build_row_with_player_stats():
    """FeatureBuilder should include matchup features when player_stats is provided."""
    matches = [
        {
            "match_id": "m_0", "date": "2024-03-23T12:00:00+11:00", "round_num": 1,
            "home_team": "A", "away_team": "B",
            "home_score": 60, "away_score": 55, "venue": "Brisbane Entertainment Centre",
        },
        {
            "match_id": "m_1", "date": "2024-03-30T12:00:00+11:00", "round_num": 2,
            "home_team": "B", "away_team": "A",
            "home_score": 58, "away_score": 52, "venue": "Ken Rosewall Arena",
        },
    ]

    positions = ["GS", "GA", "WA", "C", "WD", "GD", "GK"]
    player_stats = {}
    for mid in ["m_0", "m_1"]:
        starters = []
        for team_idx, team in enumerate(["A", "B"]):
            for pos_idx, pos in enumerate(positions):
                starters.append({
                    "player_id": team_idx * 100 + pos_idx, "team": team,
                    "position": pos, "goals": 30 if pos in ("GS", "GA") else 0,
                    "attempts": 35 if pos in ("GS", "GA") else 0,
                    "assists": 5, "rebounds": 3, "feeds": 10, "turnovers": 2,
                    "gains": 3, "intercepts": 2, "deflections": 1,
                    "penalties": 2, "centre_pass_receives": 4,
                })
        player_stats[mid] = starters

    builder = FeatureBuilder(matches, player_stats=player_stats)
    row = builder.build_row(match_index=1)

    # Should have matchup features alongside existing features
    assert "home_elo" in row
    assert "gs_vs_gk_goals_diff" in row
    assert "ga_vs_gd_shooting_pct_diff" in row
    assert "c_vs_c_assists_diff" in row


def test_build_row_without_player_stats():
    """FeatureBuilder should work as before when player_stats is None."""
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
    assert "gs_vs_gk_goals_diff" not in row
```

**Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/features/test_builder.py::test_build_row_with_player_stats -v`
Expected: FAIL (FeatureBuilder doesn't accept `player_stats` parameter yet)

**Step 3: Write minimal implementation**

Modify `src/netball_model/features/builder.py`:

1. Add imports at the top:
```python
from netball_model.features.player_profile import PlayerProfiler
from netball_model.features.matchups import MatchupFeatures
```

2. Update `__init__` (line 22):
```python
def __init__(self, matches: list[dict], pool: str = "ssn",
             player_stats: dict[str, list[dict]] | None = None):
    self.matches = matches
    self.pool = pool
    self.glicko = GlickoSystem()
    self.ctx = ContextualFeatures(matches)
    self._elo_computed_up_to = -1
    self._player_stats = player_stats  # match_id -> [starters]
    self._profiler = PlayerProfiler() if player_stats else None
    self._matchups = MatchupFeatures() if player_stats else None
```

3. Add new method after `_ensure_elo_up_to`:
```python
def _build_matchup_features(self, match_index: int) -> dict[str, float]:
    """Build position-pair difference features for the match at match_index."""
    m = self.matches[match_index]
    match_id = m["match_id"]
    home_team = m["home_team"]
    away_team = m["away_team"]
    match_date = (m.get("date") or "")[:10]

    # Get starters for this match (for lineup); build profiles from history
    starters = self._player_stats.get(match_id, [])
    if not starters:
        return {}

    home_profiles: dict[str, dict] = {}
    away_profiles: dict[str, dict] = {}

    for starter in starters:
        pos = starter["position"]
        if pos == "-":
            continue
        pid = starter["player_id"]
        team = starter["team"]

        # Build rolling profile from prior matches
        history = self._get_player_history(pid, match_index)
        profile = self._profiler.compute_profile(history, pos)

        if profile is None:
            continue

        if team == home_team:
            home_profiles[pos] = profile
        elif team == away_team:
            away_profiles[pos] = profile

    return self._matchups.compute_features(home_profiles, away_profiles)

def _get_player_history(self, player_id: int, before_index: int) -> list[dict]:
    """Get a player's stat rows from matches before before_index (up to 5)."""
    history = []
    for i in range(before_index - 1, -1, -1):
        if len(history) >= 5:
            break
        mid = self.matches[i]["match_id"]
        match_starters = self._player_stats.get(mid, [])
        for s in match_starters:
            if s["player_id"] == player_id:
                history.append(s)
                break
    return history
```

4. Update `build_row` (line 76 return dict): merge matchup features into the return dict:
```python
    row = {
        "match_id": m["match_id"],
        # ... existing fields unchanged ...
        "total_goals": hs + as_,
    }

    if self._player_stats is not None:
        row.update(self._build_matchup_features(match_index))

    return row
```

**Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/features/test_builder.py -v`
Expected: All PASS (existing tests plus 2 new tests)

**Step 5: Commit**

Message: `feat: integrate player matchup features into FeatureBuilder`

---

### Task 6: Services — Wire Player Stats Through Backtest Pipeline

**Files:**
- Modify: `src/netball_model/services.py`
- Test: `tests/test_services.py`

**Step 1: Write the failing test**

Add to `tests/test_services.py`:

```python
def test_backtest_season_with_player_stats(tmp_db, seed_matches):
    """Backtest should work when player stats are available."""
    db = Database(tmp_db)
    db.initialize()
    matches = seed_matches(db, n=50)

    # Add player stats for each match
    positions = ["GS", "GA", "WA", "C", "WD", "GD", "GK"]
    for m in matches:
        for team_idx, team in enumerate([m["home_team"], m["away_team"]]):
            for pos_idx, pos in enumerate(positions):
                db.insert_player_stats({
                    "match_id": m["match_id"],
                    "player_id": hash(team) % 10000 + pos_idx,
                    "player_name": "", "team": team, "position": pos,
                    "goals": 30 if pos in ("GS", "GA") else 0,
                    "attempts": 35 if pos in ("GS", "GA") else 0,
                    "assists": 5, "rebounds": 3, "feeds": 10, "turnovers": 2,
                    "gains": 3, "intercepts": 2, "deflections": 1,
                    "penalties": 2, "centre_pass_receives": 4,
                    "net_points": 0.0,
                })

    result = backtest_season(db, (2024, 2024), 2024, use_player_stats=True)

    assert result["matches"] > 0
    assert 0 <= result["accuracy"] <= 1
    assert result["mae"] >= 0
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_services.py::test_backtest_season_with_player_stats -v`
Expected: FAIL (`backtest_season` doesn't accept `use_player_stats`)

**Step 3: Write minimal implementation**

Modify `src/netball_model/services.py`, update `backtest_season`:

```python
def backtest_season(
    db: Database, train_range: tuple[int, int], test_season: int,
    use_player_stats: bool = False,
) -> dict:
    """Walk-forward backtest. Returns dict with matches, accuracy, mae."""
    all_matches = db.get_matches()

    start, end = train_range
    train_matches = [m for m in all_matches if start <= m["season"] <= end]
    test_matches = [m for m in all_matches if m["season"] == test_season]

    if not train_matches or not test_matches:
        raise ValueError("Insufficient data for backtest.")

    # Load player stats if requested
    player_stats = None
    if use_player_stats:
        player_stats = {}
        all_for_backtest = train_matches + test_matches
        for m in all_for_backtest:
            starters = db.get_starters_for_match(m["match_id"])
            if starters:
                player_stats[m["match_id"]] = starters

    builder = FeatureBuilder(train_matches, player_stats=player_stats)
    train_df = builder.build_matrix(start_index=1)

    model = NetballModel()
    model.train(train_df)

    # Walk-forward: for each test match, build features using all prior matches
    all_for_test = train_matches + test_matches
    test_builder = FeatureBuilder(all_for_test, player_stats=player_stats)

    correct = 0
    total = 0
    abs_errors = []

    for i in range(len(train_matches), len(all_for_test)):
        row = test_builder.build_row(i)
        pred = model.predict(pd.DataFrame([row]))

        pred_margin = float(pred["predicted_margin"].iloc[0])
        actual_margin = row["margin"]

        abs_errors.append(abs(pred_margin - actual_margin))
        if (pred_margin > 0 and actual_margin > 0) or (pred_margin < 0 and actual_margin < 0):
            correct += 1
        total += 1

    mae = float(np.mean(abs_errors))
    accuracy = correct / total if total > 0 else 0.0

    return {
        "test_season": test_season,
        "matches": total,
        "accuracy": accuracy,
        "mae": mae,
    }
```

**Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_services.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `poetry run pytest -v`
Expected: All PASS

**Step 6: Commit**

Message: `feat: wire player stats through backtest pipeline`

---

### Task 7: Notebook — Add Flat Betting Strategy

**Files:**
- Modify: `notebooks/model_evaluation.ipynb`

**Step 1: Add flat betting to kelly_scales dict**

In cell 20 (Kelly implementation cell), add `simulate_flat` function:

```python
def simulate_flat(
    bets: list[dict],
    bankroll: float = STARTING_BANKROLL,
    stake: float = 50.0,
) -> dict:
    """Simulate flat betting — fixed stake per bet.

    Only bets when Kelly fraction > 0 (model sees edge).
    """
    initial_bankroll = bankroll
    curve = [bankroll]
    bets_placed = 0
    wins = 0
    total_wagered = 0.0

    for bet in bets:
        f = kelly_fraction(bet["model_prob"], bet["odds"])
        if f <= 0:
            curve.append(bankroll)
            continue

        actual_stake = min(stake, bankroll)
        if actual_stake <= 0:
            curve.append(bankroll)
            continue

        total_wagered += actual_stake
        bets_placed += 1

        if bet["won"]:
            bankroll += actual_stake * (bet["odds"] - 1)
            wins += 1
        else:
            bankroll -= actual_stake

        curve.append(bankroll)

    roi = (bankroll - initial_bankroll) / total_wagered if total_wagered > 0 else 0.0
    win_rate = wins / bets_placed if bets_placed > 0 else 0.0

    return {
        "bankroll_curve": curve,
        "final_bankroll": bankroll,
        "bets_placed": bets_placed,
        "wins": wins,
        "win_rate": win_rate,
        "total_wagered": total_wagered,
        "roi": roi,
    }
```

**Step 2: Update cell 24 (per-season results)**

Add flat betting to the summary table. In the simulation loop, after the Kelly scales loop, add:

```python
        # Flat betting
        result = simulate_flat(bets, bankroll=STARTING_BANKROLL, stake=50.0)
        summary_rows.append({
            "Season": test_season,
            "Strategy": "Flat $50",
            "Bets": result["bets_placed"],
            "Wins": result["wins"],
            "Win Rate": result["win_rate"],
            "Final Bankroll": result["final_bankroll"],
            "ROI": result["roi"],
            "Total Wagered": result["total_wagered"],
        })
```

**Step 3: Update cell 28 (aggregate results)**

Add flat betting to the aggregate simulation. After the Kelly loop, add:

```python
    # Flat betting aggregate
    flat_result = simulate_flat(all_bets, bankroll=STARTING_BANKROLL, stake=50.0)
    aggregate_results["Flat $50"] = flat_result
    ax.plot(
        flat_result["bankroll_curve"],
        label=f"Flat $50 (${flat_result['final_bankroll']:,.0f})",
        linewidth=1.5, linestyle="--",
    )
```

**Step 4: Re-execute notebook**

Run: `poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/model_evaluation.ipynb`

**Step 5: Commit**

Message: `feat(notebook): add flat betting strategy to evaluation`

---

### Task 8: Notebook — Baseline vs Matchup Comparison

**Files:**
- Modify: `notebooks/model_evaluation.ipynb`

**Step 1: Add a new section after the existing betting simulation**

Add a new markdown cell:
```markdown
## Matchup Features Comparison

Compare the baseline model (Elo/form/travel only) against the model with player matchup features.
```

**Step 2: Add comparison cell**

```python
from netball_model.features.builder import FeatureBuilder
from netball_model.features.player_profile import PlayerProfiler
from netball_model.features.matchups import MatchupFeatures

# Load all player stats
player_stats = {}
for m in all_matches:
    starters = db.get_starters_for_match(m["match_id"])
    if starters:
        player_stats[m["match_id"]] = starters

print(f"Loaded player stats for {len(player_stats)} matches")

# Run walk-forward with matchup features
matchup_season_results = {}
skipped = 0

for test_season in available_seasons[2:]:
    train_end = test_season - 1
    train_matches = [m for m in all_matches if first <= m["season"] <= train_end]
    test_matches = [m for m in all_matches if m["season"] == test_season]

    if not train_matches or not test_matches:
        continue

    builder = FeatureBuilder(train_matches, player_stats=player_stats)
    train_df = builder.build_matrix(start_index=1)
    model = NetballModel()
    model.train(train_df)

    all_for_test = train_matches + test_matches
    test_builder = FeatureBuilder(all_for_test, player_stats=player_stats)

    bets = []
    for i in range(len(train_matches), len(all_for_test)):
        match = all_for_test[i]
        match_id = match["match_id"]

        if match_id not in odds_lookup:
            skipped += 1
            continue

        home_odds, away_odds = odds_lookup[match_id]
        if home_odds is None or away_odds is None:
            skipped += 1
            continue

        row = test_builder.build_row(i)
        pred = model.predict(pd.DataFrame([row]))

        home_win_prob = float(pred["win_probability"].iloc[0])
        away_win_prob = 1 - home_win_prob
        actual_margin = match["home_score"] - match["away_score"]
        home_won = actual_margin > 0

        home_kelly = kelly_fraction(home_win_prob, home_odds)
        away_kelly = kelly_fraction(away_win_prob, away_odds)

        if home_kelly > 0:
            bets.append({
                "match_id": match_id, "season": test_season, "side": "home",
                "team": match["home_team"], "model_prob": home_win_prob,
                "odds": home_odds, "won": home_won, "kelly": home_kelly,
            })
        elif away_kelly > 0:
            bets.append({
                "match_id": match_id, "season": test_season, "side": "away",
                "team": match["away_team"], "model_prob": away_win_prob,
                "odds": away_odds, "won": not home_won, "kelly": away_kelly,
            })

    matchup_season_results[test_season] = bets

# Compare baseline vs matchup
print("\nBaseline vs Matchup Features — Quarter Kelly")
print("=" * 70)
print(f"{'Season':>6}  {'Baseline':>12}  {'+ Matchups':>12}  {'Diff':>10}")
print("-" * 70)

for season in sorted(set(list(season_results.keys()) + list(matchup_season_results.keys()))):
    base_bets = season_results.get(season, [])
    match_bets = matchup_season_results.get(season, [])

    base_result = simulate_kelly(base_bets, STARTING_BANKROLL, kelly_scale=0.25) if base_bets else None
    match_result = simulate_kelly(match_bets, STARTING_BANKROLL, kelly_scale=0.25) if match_bets else None

    base_pnl = base_result["final_bankroll"] - STARTING_BANKROLL if base_result else 0
    match_pnl = match_result["final_bankroll"] - STARTING_BANKROLL if match_result else 0

    print(f"{season:>6}  ${base_pnl:>+10,.2f}  ${match_pnl:>+10,.2f}  ${match_pnl - base_pnl:>+8,.2f}")
```

**Step 3: Re-execute notebook**

Run: `poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/model_evaluation.ipynb`

**Step 4: Commit**

Message: `feat(notebook): add baseline vs matchup features comparison`

---

### Task 9: Full Test Suite + Final Validation

**Step 1: Run full test suite**

Run: `poetry run pytest -v`
Expected: All PASS

**Step 2: Verify notebook executes cleanly**

Run: `poetry run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace notebooks/model_evaluation.ipynb`

**Step 3: Review results**

Check the comparison table in the notebook:
- Does adding matchup features change OOS accuracy?
- Does betting ROI improve for any season?
- Are the difference score features meaningful (check Ridge coefficients)?

**Step 4: Commit**

Message: `chore: validate player matchup features end-to-end`

---

## Summary

| Task | Component | New Files | Tests |
|------|-----------|-----------|-------|
| 1 | DB: `get_player_history` | — | 1 test |
| 2 | DB: `get_starters_for_match` | — | 1 test |
| 3 | `PlayerProfiler` | `features/player_profile.py` | 6 tests |
| 4 | `MatchupFeatures` | `features/matchups.py` | 3 tests |
| 5 | `FeatureBuilder` integration | — | 2 tests |
| 6 | `services.py` pipeline | — | 1 test |
| 7 | Notebook: flat betting | — | — |
| 8 | Notebook: matchup comparison | — | — |
| 9 | Full validation | — | — |

**Total: ~14 new tests, 3 new files, 4 modified files**

Phase 2 (clustering) will be planned separately after Phase 1 results are evaluated.
