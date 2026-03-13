# Model Enhancement Implementation Plan

> **For agentic workers:** REQUIRED: Use `TeamCreate` to create a team of agents and coordinate via `TaskCreate`/`TaskUpdate`/`SendMessage`. This plan is designed for a **team of agents** working in parallel. See the Agent Execution Strategy section for how to dispatch.

**Goal:** Add bet365 screenshot OCR, 2026 squad data with player movement tracking, individual player Glicko-2 ratings, time-weighted training with roster continuity, and line/total value detection.

**Architecture:** 5 independent subsystems built by parallel agents, then a final integration task wires them into the existing FeatureBuilder/services/app. Each subsystem has its own tests and can be developed in isolation.

**Tech Stack:** Python 3.11+, Poetry, EasyOCR, Pillow, glicko2, scikit-learn (Ridge), SQLite, scipy.stats.norm

**Spec:** `docs/superpowers/specs/2026-03-13-model-enhancement-design.md`

---

## Agent Execution Strategy

Use `TeamCreate` to create a team named `model-enhancement`. Create all tasks upfront with `TaskCreate`, setting dependencies via `addBlockedBy` so agents naturally pick up work as it becomes unblocked. Each agent works in an **isolated worktree** (set `isolation: "worktree"` when spawning via `Agent` tool).

### Setup

1. **Create team:** `TeamCreate` with `team_name: "model-enhancement"`
2. **Create all 6 tasks** with `TaskCreate` — set `addBlockedBy` on Tasks 4 & 5 to block on Task 2 (squads), and Task 6 to block on Tasks 1-5
3. **Spawn agents** for Wave 1 immediately, Wave 2 agents after squads-agent completes

### Wave 1 — Spawn Immediately (3 agents, parallel)

| Agent Name | Task | Worktree Branch |
|------------|------|----------------|
| ocr-agent | Task 1: Screenshot OCR | `feature/screenshot-ocr` |
| squads-agent | Task 2: Squads + Player Movements | `feature/squads-movements` |
| value-agent | Task 3: Line/Total Value Detection | `feature/value-markets` |

Spawn all 3 via `Agent` tool with `team_name: "model-enhancement"`, `isolation: "worktree"`, and `mode: "bypassPermissions"`. Assign tasks with `TaskUpdate` (set `owner`).

### Wave 2 — After squads-agent completes (2 agents, parallel)

| Agent Name | Task | Worktree Branch |
|------------|------|----------------|
| player-elo-agent | Task 4: Player Glicko-2 + Rankings | `feature/player-elo` |
| time-weight-agent | Task 5: Time Weighting + Round Number | `feature/time-weighting` |

These tasks are blocked by Task 2. Once squads-agent marks Task 2 complete, spawn these agents.

### Wave 3 — After all agents complete

| Agent Name | Task | Branch |
|------------|------|--------|
| integration-agent | Task 6: Integration | `feature/player-matchups` (current) |

Blocked by Tasks 1-5. Merge all worktree branches, then spawn integration-agent.

### Team Coordination

- **Task tracking:** All agents check `TaskList` after completing work to find next available tasks
- **Communication:** Use `SendMessage` for agent-to-lead communication (e.g., completion, blockers)
- **Shutdown:** Send `shutdown_request` to each agent when their work is done
- **Cleanup:** `TeamDelete` after all work is merged and verified

---

## Chunk 1: Wave 1 Tasks (Parallel)

### Task 1: Bet365 Screenshot OCR

**Agent:** `ocr-agent`

**Files:**
- Create: `src/netball_model/data/bet365_screenshot.py`
- Create: `tests/data/test_bet365_screenshot.py`
- Create: `tests/data/fixtures/bet365_screenshot.png` (copy from user-provided image)
- Modify: `pyproject.toml` (add easyocr dependency)

**Context for agent:** The bet365 screenshot has a dark background (~35-40 luminance), yellow odds values, white team names/labels, and green/gray section headers. The layout is: match title at top ("Team A v Team B"), then 3 sections ("Match Lines", "1st Half", "1st Quarter"), each containing a table with rows "To Win", "Total", "Line". The existing scraper's output schema uses keys like `handicap_home_odds` (not `home_handicap_odds`). See spec Section 1 for full output schema (27 keys). Missing sections return `None` values. Failed OCR raises `ValueError`.

- [ ] **Step 1: Add easyocr dependency**

```bash
poetry add easyocr
```

- [ ] **Step 2: Write test for section text parsing**

Create `tests/data/test_bet365_screenshot.py`:

```python
import pytest
from netball_model.data.bet365_screenshot import _parse_section_text


class TestParseSectionText:
    def test_to_win_row(self):
        text = "To Win\n1.30\n3.50"
        result = _parse_section_text(text)
        assert result["home_odds"] == 1.30
        assert result["away_odds"] == 3.50

    def test_total_row(self):
        text = "Total\nO 125.5 1.87\nU 125.5 1.87"
        result = _parse_section_text(text)
        assert result["total_line"] == 125.5
        assert result["over_odds"] == 1.87
        assert result["under_odds"] == 1.87

    def test_line_row(self):
        text = "Line\n-5.5 1.85\n+5.5 1.95"
        result = _parse_section_text(text)
        assert result["handicap_line"] == -5.5
        assert result["handicap_home_odds"] == 1.85
        assert result["handicap_away_odds"] == 1.95

    def test_missing_row_returns_none(self):
        text = "To Win\n1.30\n3.50"
        result = _parse_section_text(text)
        assert result.get("total_line") is None
        assert result.get("handicap_line") is None

    def test_full_section(self):
        text = (
            "Match Lines\n"
            "Melbourne Mavericks    GIANTS Netball\n"
            "To Win\n1.30\n3.50\n"
            "Total\nO 125.5 1.87\nU 125.5 1.87\n"
            "Line\n-5.5 1.85\n+5.5 1.95"
        )
        result = _parse_section_text(text)
        assert result["home_odds"] == 1.30
        assert result["away_odds"] == 3.50
        assert result["total_line"] == 125.5
        assert result["handicap_line"] == -5.5
```

- [ ] **Step 3: Run test to verify it fails**

```bash
poetry run pytest tests/data/test_bet365_screenshot.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 4: Implement `_parse_section_text`**

Create `src/netball_model/data/bet365_screenshot.py`:

```python
"""Parse bet365 screenshots into structured odds data using OCR."""
from __future__ import annotations

import re
from pathlib import Path

TO_WIN_RE = re.compile(r"(\d+\.\d{2})")
TOTAL_RE = re.compile(r"([OU])\s+(\d+\.?\d*)\s+(\d+\.\d{2})")
LINE_RE = re.compile(r"([+-]\d+\.?\d*)\s+(\d+\.\d{2})")
MATCH_TITLE_RE = re.compile(r"(.+?)\s+v\s+(.+)")


def _parse_section_text(text: str) -> dict:
    """Parse raw text from one section into odds dict."""
    result = {}
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    to_win_odds = []
    total_matches = []
    line_matches = []

    i = 0
    while i < len(lines):
        stripped = lines[i]
        if stripped.startswith("To Win") or stripped == "To Win":
            # Collect the next odds values
            remaining = " ".join(lines[i:i+3])
            odds = TO_WIN_RE.findall(remaining)
            # Filter out any that are part of Total/Line patterns
            to_win_odds = [float(o) for o in odds[:2]]
            i += 1
            continue

        if stripped.startswith("Total") or stripped == "Total":
            remaining = " ".join(lines[i:i+3])
            total_matches = TOTAL_RE.findall(remaining)
            i += 1
            continue

        if stripped == "Line" or (stripped.startswith("Line") and "Match" not in stripped):
            remaining = " ".join(lines[i:i+3])
            line_matches = LINE_RE.findall(remaining)
            i += 1
            continue

        i += 1

    if len(to_win_odds) >= 2:
        result["home_odds"] = to_win_odds[0]
        result["away_odds"] = to_win_odds[1]

    for m in total_matches:
        direction, line_val, odds_val = m
        if direction == "O":
            result["total_line"] = float(line_val)
            result["over_odds"] = float(odds_val)
        elif direction == "U":
            result["under_odds"] = float(odds_val)

    if len(line_matches) >= 2:
        result["handicap_line"] = float(line_matches[0][0])
        result["handicap_home_odds"] = float(line_matches[0][1])
        result["handicap_away_odds"] = float(line_matches[1][1])

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
poetry run pytest tests/data/test_bet365_screenshot.py -v
```

- [ ] **Step 6: Write test for full `parse_screenshot` with mocked OCR**

Add to `tests/data/test_bet365_screenshot.py`:

```python
from unittest.mock import patch, MagicMock
from netball_model.data.bet365_screenshot import parse_screenshot


class TestParseScreenshot:
    def test_full_parse_mocked_ocr(self, tmp_path):
        """Test full pipeline with mocked EasyOCR output."""
        # Create a dummy image file
        from PIL import Image
        img = Image.new("RGB", (1200, 800), color=(30, 30, 30))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        # Mock EasyOCR to return structured text regions
        mock_results = [
            # Match title
            ([0, 0, 400, 30], "Super Netball", 0.9),
            ([0, 30, 600, 60], "Melbourne Mavericks v GIANTS Netball", 0.95),
            # Match Lines section
            ([0, 100, 200, 130], "Match Lines", 0.9),
            ([0, 140, 100, 160], "To Win", 0.9),
            ([200, 140, 300, 160], "1.30", 0.95),
            ([500, 140, 600, 160], "3.50", 0.95),
            ([0, 170, 100, 190], "Total", 0.9),
            ([200, 170, 400, 190], "O 125.5 1.87", 0.95),
            ([500, 170, 700, 190], "U 125.5 1.87", 0.95),
            ([0, 200, 100, 220], "Line", 0.9),
            ([200, 200, 400, 220], "-5.5 1.85", 0.95),
            ([500, 200, 700, 220], "+5.5 1.95", 0.95),
        ]

        with patch("netball_model.data.bet365_screenshot.easyocr") as mock_easyocr:
            reader = MagicMock()
            reader.readtext.return_value = mock_results
            mock_easyocr.Reader.return_value = reader

            result = parse_screenshot(str(img_path))

        assert result["home_team"] == "Melbourne Mavericks"
        assert result["away_team"] == "GIANTS Netball"
        assert result["home_odds"] == 1.30
        assert result["away_odds"] == 3.50
        assert result["total_line"] == 125.5
        assert result["handicap_line"] == -5.5

    def test_missing_sections_return_none(self, tmp_path):
        """When only Match Lines is visible, h1_ and q1_ keys are None."""
        from PIL import Image
        img = Image.new("RGB", (1200, 400), color=(30, 30, 30))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        mock_results = [
            ([0, 30, 600, 60], "Melbourne Mavericks v GIANTS Netball", 0.95),
            ([0, 100, 200, 130], "Match Lines", 0.9),
            ([0, 140, 100, 160], "To Win", 0.9),
            ([200, 140, 300, 160], "1.30", 0.95),
            ([500, 140, 600, 160], "3.50", 0.95),
        ]

        with patch("netball_model.data.bet365_screenshot.easyocr") as mock_easyocr:
            reader = MagicMock()
            reader.readtext.return_value = mock_results
            mock_easyocr.Reader.return_value = reader

            result = parse_screenshot(str(img_path))

        assert result["home_odds"] == 1.30
        assert result["h1_home_odds"] is None
        assert result["q1_home_odds"] is None

    def test_invalid_image_raises(self, tmp_path):
        bad_path = tmp_path / "nonexistent.png"
        with pytest.raises((ValueError, FileNotFoundError)):
            parse_screenshot(str(bad_path))
```

- [ ] **Step 7: Implement `parse_screenshot`**

Complete the `parse_screenshot` function in `bet365_screenshot.py`. Key logic:

1. Load image with Pillow, verify it exists
2. Run EasyOCR reader on the image
3. Extract match title from "X v Y" pattern in OCR results
4. Group OCR results by vertical bands into sections (Match Lines, 1st Half, 1st Quarter) using section header text
5. For each section, reconstruct text and call `_parse_section_text()`
6. Prefix h1_/q1_ keys for half/quarter sections
7. Fill missing keys with None

```python
import easyocr
from PIL import Image

_SECTION_HEADERS = ["Match Lines", "1st Half", "1st Quarter"]
_SECTION_PREFIXES = {"Match Lines": "", "1st Half": "h1_", "1st Quarter": "q1_"}

_ALL_KEYS = [
    "home_team", "away_team", "match_date",
    "home_odds", "away_odds", "handicap_line", "handicap_home_odds",
    "handicap_away_odds", "total_line", "over_odds", "under_odds",
    "h1_home_odds", "h1_away_odds", "h1_handicap_line", "h1_handicap_home_odds",
    "h1_handicap_away_odds", "h1_total_line", "h1_over_odds", "h1_under_odds",
    "q1_home_odds", "q1_away_odds", "q1_handicap_line", "q1_handicap_home_odds",
    "q1_handicap_away_odds", "q1_total_line", "q1_over_odds", "q1_under_odds",
]


def parse_screenshot(image_path: str | Path) -> dict:
    """Parse a bet365 screenshot and return structured odds data."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(str(path))

    if not results:
        raise ValueError("OCR produced no results from the image")

    # Extract all text with positions: (bbox, text, confidence)
    texts = [(r[0], r[1], r[2]) for r in results]

    # Find match title
    home_team = None
    away_team = None
    for bbox, text, conf in texts:
        m = MATCH_TITLE_RE.search(text)
        if m and " v " in text:
            from netball_model.data.team_names import normalise_team
            home_team = normalise_team(m.group(1).strip())
            away_team = normalise_team(m.group(2).strip())
            break

    # Group text into sections by vertical position
    sections = _group_into_sections(texts)

    # Parse each section
    output = {k: None for k in _ALL_KEYS}
    output["home_team"] = home_team
    output["away_team"] = away_team
    output["match_date"] = None

    for section_name, section_text in sections.items():
        prefix = _SECTION_PREFIXES.get(section_name, "")
        parsed = _parse_section_text(section_text)
        for key, value in parsed.items():
            output[f"{prefix}{key}"] = value

    return output


def _group_into_sections(texts: list) -> dict[str, str]:
    """Group OCR text elements into sections based on header text."""
    # Find section header positions
    section_starts = []
    for bbox, text, conf in texts:
        for header in _SECTION_HEADERS:
            if header.lower() in text.lower():
                y_pos = bbox[0][1] if isinstance(bbox[0], list) else bbox[1]
                section_starts.append((y_pos, header))
                break

    if not section_starts:
        # No sections found, treat all as Match Lines
        all_text = "\n".join(t[1] for t in texts)
        return {"Match Lines": all_text}

    section_starts.sort(key=lambda x: x[0])

    # Assign text to sections by vertical position
    sections = {}
    for i, (y_start, name) in enumerate(section_starts):
        y_end = section_starts[i + 1][0] if i + 1 < len(section_starts) else float("inf")
        section_texts = []
        for bbox, text, conf in texts:
            text_y = bbox[0][1] if isinstance(bbox[0], list) else bbox[1]
            if y_start <= text_y < y_end:
                section_texts.append(text)
        sections[name] = "\n".join(section_texts)

    return sections
```

- [ ] **Step 8: Run all screenshot tests**

```bash
poetry run pytest tests/data/test_bet365_screenshot.py -v
```

- [ ] **Step 9: Run full test suite to ensure no regressions**

```bash
poetry run pytest -v
```

---

### Task 2: 2026 Squads + Player Movement Tracking

**Agent:** `squads-agent`

**Files:**
- Create: `src/netball_model/data/squads.py`
- Create: `src/netball_model/data/player_movements.py`
- Create: `tests/data/test_squads.py`
- Create: `tests/data/test_player_movements.py`

**Context for agent:** The database has 539 matches (2017-2025), 254 players, 11k+ player stat rows. The `player_stats` table has columns: `match_id, player_id, player_name, team, position`. Use `difflib.SequenceMatcher` with threshold 0.85 for fuzzy name matching. Print unmatched names to stderr. See spec Section 2 for full squad data and player movement table.

- [ ] **Step 1: Write squad data tests**

Create `tests/data/test_squads.py`:

```python
from netball_model.data.squads import get_squad, get_all_squads, SQUADS_2026

POSITIONS = {"GS", "GA", "WA", "C", "WD", "GD", "GK"}
TEAMS_2026 = [
    "Adelaide Thunderbirds", "GIANTS Netball", "Melbourne Mavericks",
    "Melbourne Vixens", "NSW Swifts", "Queensland Firebirds",
    "Sunshine Coast Lightning", "West Coast Fever",
]


def test_all_teams_present():
    squads = get_all_squads(2026)
    assert set(squads.keys()) == set(TEAMS_2026)


def test_each_team_has_seven_positions():
    for team in TEAMS_2026:
        squad = get_squad(team, 2026)
        assert set(squad.keys()) == POSITIONS, f"{team} missing positions"


def test_no_duplicate_players_across_teams():
    all_players = []
    for team, squad in get_all_squads(2026).items():
        all_players.extend(squad.values())
    assert len(all_players) == len(set(all_players)), "Duplicate player found"


def test_get_squad_specific_team():
    squad = get_squad("Melbourne Vixens", 2026)
    assert squad["GS"] == "Sophie Garbin"
    assert squad["C"] == "Kate Moloney"


def test_unknown_team_raises():
    import pytest
    with pytest.raises(KeyError):
        get_squad("Nonexistent Team", 2026)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/data/test_squads.py -v
```

- [ ] **Step 3: Implement squads module**

Create `src/netball_model/data/squads.py` with the full `SQUADS_2026` dict from the spec (lines 120-193), plus:

```python
"""2026 SSN squad data and lookup functions."""
from __future__ import annotations

SQUADS_2026 = {
    # ... (copy full dict from spec)
}

_SQUADS_BY_SEASON = {
    2026: SQUADS_2026,
}


def get_squad(team: str, season: int = 2026) -> dict[str, str]:
    """Return {position: player_name} for a team in a season."""
    squads = _SQUADS_BY_SEASON.get(season)
    if squads is None:
        raise ValueError(f"No squad data for season {season}")
    return squads[team]  # KeyError if team not found


def get_all_squads(season: int = 2026) -> dict[str, dict[str, str]]:
    """Return all team squads for a season."""
    squads = _SQUADS_BY_SEASON.get(season)
    if squads is None:
        raise ValueError(f"No squad data for season {season}")
    return dict(squads)
```

- [ ] **Step 4: Run squad tests**

```bash
poetry run pytest tests/data/test_squads.py -v
```

- [ ] **Step 5: Write player movements tests**

Create `tests/data/test_player_movements.py`:

```python
import pytest
from netball_model.data.database import Database
from netball_model.data.player_movements import (
    get_roster_continuity,
    get_player_movements,
    get_team_continuity_all,
)


@pytest.fixture
def db_with_two_seasons(tmp_path):
    """DB with players across two seasons, some moving teams."""
    db = Database(tmp_path / "test.db")
    db.initialize()

    # Season 2024: Team A has players 1-7, Team B has players 8-14
    for i, pos in enumerate(["GS", "GA", "WA", "C", "WD", "GD", "GK"]):
        db.insert_player_stats({
            "match_id": "m2024_01", "player_id": i + 1,
            "player_name": f"Player{i+1}", "team": "Team A",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })
        db.insert_player_stats({
            "match_id": "m2024_01", "player_id": i + 8,
            "player_name": f"Player{i+8}", "team": "Team B",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })

    # Create match records
    for mid, season in [("m2024_01", 2024), ("m2025_01", 2025)]:
        db.upsert_match({
            "match_id": mid, "competition_id": 1, "season": season,
            "round_num": 1, "game_num": 1, "date": f"{season}-03-15",
            "venue": "Test", "home_team": "Team A", "away_team": "Team B",
            "home_score": 60, "away_score": 55,
            "home_q1": 15, "home_q2": 15, "home_q3": 15, "home_q4": 15,
            "away_q1": 14, "away_q2": 14, "away_q3": 14, "away_q4": 13,
        })

    # Season 2025: Players 1-5 stay on A, players 6-7 move to B
    for i, pos in enumerate(["GS", "GA", "WA", "C", "WD"]):
        db.insert_player_stats({
            "match_id": "m2025_01", "player_id": i + 1,
            "player_name": f"Player{i+1}", "team": "Team A",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })
    # Player 6 and 7 moved to Team B
    for i, pos in enumerate(["GD", "GK"]):
        db.insert_player_stats({
            "match_id": "m2025_01", "player_id": i + 6,
            "player_name": f"Player{i+6}", "team": "Team B",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })
    # New players 15-16 on Team A
    for i, pos in enumerate(["GD", "GK"]):
        db.insert_player_stats({
            "match_id": "m2025_01", "player_id": i + 15,
            "player_name": f"Player{i+15}", "team": "Team A",
            "position": pos, "goals": 0, "attempts": 0, "assists": 0,
            "rebounds": 0, "feeds": 0, "turnovers": 0, "gains": 0,
            "intercepts": 0, "deflections": 0, "penalties": 0,
            "centre_pass_receives": 0, "net_points": 0,
        })

    return db


def test_roster_continuity_full_retention(db_with_two_seasons):
    # Team B kept all its original players (8-14) plus gained 6-7
    # But continuity is about starters in season N who were on same team in N-1
    # In 2025, Team B's starters include players from 2024 Team B
    # This depends on implementation — test the concept
    continuity = get_roster_continuity("Team A", 2025, db_with_two_seasons)
    # 5 out of 7 starters stayed (players 1-5), 2 are new (15-16)
    assert abs(continuity - 5 / 7) < 0.01


def test_continuity_all_teams(db_with_two_seasons):
    result = get_team_continuity_all(2025, db_with_two_seasons)
    assert "Team A" in result
    assert 0.0 <= result["Team A"] <= 1.0
```

- [ ] **Step 6: Implement player movements module**

Create `src/netball_model/data/player_movements.py`:

```python
"""Track player movements between seasons for roster continuity."""
from __future__ import annotations

import difflib
import sys

from netball_model.data.database import Database


def get_player_movements(season: int, db: Database) -> list[dict]:
    """Detect player movements into a given season."""
    prev_season = season - 1
    current_matches = db.get_matches(season=season)
    prev_matches = db.get_matches(season=prev_season)

    # Build player->team maps for each season
    prev_teams = _build_player_team_map(prev_matches, db)
    curr_teams = _build_player_team_map(current_matches, db)

    # For 2026, supplement with hardcoded squad data
    if season == 2026 and not curr_teams:
        from netball_model.data.squads import get_all_squads
        squads = get_all_squads(2026)
        for team, positions in squads.items():
            for pos, name in positions.items():
                pid = _fuzzy_match_player_id(name, prev_matches, db)
                if pid is not None:
                    curr_teams[pid] = team
                # Skip players with no DB match — they are new to the league

    movements = []
    all_players = set(prev_teams.keys()) | set(curr_teams.keys())

    for player in all_players:
        prev_team = prev_teams.get(player)
        curr_team = curr_teams.get(player)

        if prev_team and curr_team:
            movement_type = "stayed" if prev_team == curr_team else "moved"
        elif curr_team and not prev_team:
            movement_type = "new"
        else:
            movement_type = "retired"

        movements.append({
            "player_name": str(player),
            "player_id": player if isinstance(player, int) else None,
            "from_team": prev_team,
            "to_team": curr_team,
            "movement_type": movement_type,
        })

    return movements


def get_roster_continuity(team: str, season: int, db: Database) -> float:
    """Fraction of current season's starters who played for same team last season."""
    prev_season = season - 1
    current_matches = db.get_matches(season=season)
    prev_matches = db.get_matches(season=prev_season)

    curr_players = set()
    for m in current_matches:
        starters = db.get_starters_for_match(m["match_id"])
        for s in starters:
            if s["team"] == team:
                curr_players.add(s["player_id"])

    # For 2026 with no matches yet, use squad data
    if not curr_players and season == 2026:
        from netball_model.data.squads import get_squad
        try:
            squad = get_squad(team, 2026)
            # Map names to IDs from previous season
            for name in squad.values():
                pid = _find_player_id_by_name(name, prev_matches, db)
                if pid:
                    curr_players.add(pid)
        except (KeyError, ValueError):
            pass

    if not curr_players:
        return 0.0

    prev_players_on_team = set()
    for m in prev_matches:
        starters = db.get_starters_for_match(m["match_id"])
        for s in starters:
            if s["team"] == team:
                prev_players_on_team.add(s["player_id"])

    retained = curr_players & prev_players_on_team
    return len(retained) / len(curr_players) if curr_players else 0.0


def get_team_continuity_all(season: int, db: Database) -> dict[str, float]:
    """Return roster continuity for all teams in a season."""
    matches = db.get_matches(season=season)
    teams = set()
    for m in matches:
        teams.add(m["home_team"])
        teams.add(m["away_team"])

    # For 2026, also include teams from squad data
    if season == 2026:
        from netball_model.data.squads import get_all_squads
        teams.update(get_all_squads(2026).keys())

    return {team: get_roster_continuity(team, season, db) for team in teams}


def _build_player_team_map(matches: list[dict], db: Database) -> dict[int, str]:
    """Map player_id -> most recent team from match list."""
    player_teams: dict[int, str] = {}
    for m in matches:
        starters = db.get_starters_for_match(m["match_id"])
        for s in starters:
            player_teams[s["player_id"]] = s["team"]
    return player_teams


def _find_player_id_by_name(name: str, matches: list[dict], db: Database) -> int | None:
    """Find player_id by fuzzy name match against player_stats."""
    best_id = None
    best_ratio = 0.0
    seen = set()

    for m in matches:
        stats = db.get_starters_for_match(m["match_id"])
        for s in stats:
            if s["player_id"] in seen:
                continue
            seen.add(s["player_id"])
            ratio = difflib.SequenceMatcher(None, name.lower(), s["player_name"].lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = s["player_id"]

    if best_ratio < 0.85:
        print(f"Warning: No match for '{name}' (best: {best_ratio:.2f})", file=sys.stderr)
        return None

    return best_id


def _fuzzy_match_player_id(name: str, prev_matches: list[dict], db: Database) -> int | None:
    """Try to match a player name to an existing player_id via fuzzy search."""
    return _find_player_id_by_name(name, prev_matches, db)
```

- [ ] **Step 7: Run all movement tests**

```bash
poetry run pytest tests/data/test_player_movements.py -v
```

- [ ] **Step 8: Run full test suite**

```bash
poetry run pytest -v
```

---

### Task 3: Line/Total Value Detection

**Agent:** `value-agent`

**Files:**
- Modify: `src/netball_model/value/detector.py`
- Rewrite: `tests/value/test_detector.py` (existing 3 tests use old signature)
- Modify: `src/netball_model/model/calibration.py`

**Context for agent:** The existing `ValueDetector.evaluate()` takes positional args (`home_team, away_team, model_win_prob, home_odds, away_odds`) and returns a single dict. The new signature takes `(prediction: dict, odds: dict, threshold)` and returns a `list[dict]` with up to 3 value bets (h2h, handicap, total). The `CalibrationModel` needs a new `total_residual_std` field. Formulas: handicap `P(home covers) = 1 - Phi((L - M) / sigma)` where L is `odds["handicap_line"]` directly. Totals: `P(over) = 1 - Phi((Lt - T) / sigma_t)`. See spec Section 5.

- [ ] **Step 1: Write tests for new value detector**

Rewrite `tests/value/test_detector.py`:

```python
import pytest
from netball_model.value.detector import ValueDetector


class TestH2HValue:
    def test_home_value(self):
        detector = ValueDetector(min_edge=0.05)
        results = detector.evaluate(
            prediction={"margin": 10.0, "total_goals": 120.0, "win_prob": 0.65,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.80, "away_odds": 2.10},
        )
        h2h = [r for r in results if r["market"] == "h2h"]
        assert len(h2h) >= 1
        best = max(h2h, key=lambda r: r["edge"])
        assert best["side"] == "home"
        assert best["edge"] > 0.05

    def test_no_value(self):
        detector = ValueDetector(min_edge=0.05)
        results = detector.evaluate(
            prediction={"margin": 2.0, "total_goals": 120.0, "win_prob": 0.55,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.75, "away_odds": 2.20},
        )
        # No bet should have edge > 5%
        h2h_value = [r for r in results if r["market"] == "h2h" and r["edge"] >= 0.05]
        assert len(h2h_value) == 0

    def test_away_value(self):
        detector = ValueDetector(min_edge=0.05)
        results = detector.evaluate(
            prediction={"margin": -5.0, "total_goals": 120.0, "win_prob": 0.35,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.60, "away_odds": 2.50},
        )
        h2h = [r for r in results if r["market"] == "h2h" and r["edge"] >= 0.05]
        assert any(r["side"] == "away" for r in h2h)


class TestHandicapValue:
    def test_home_covers_line(self):
        detector = ValueDetector(min_edge=0.0)
        results = detector.evaluate(
            prediction={"margin": 8.5, "total_goals": 120.0, "win_prob": 0.7,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.30, "away_odds": 3.50,
                  "handicap_line": -5.5, "handicap_home_odds": 1.85,
                  "handicap_away_odds": 1.95},
        )
        handicap = [r for r in results if r["market"] == "handicap"]
        assert len(handicap) >= 1
        home_hc = [r for r in handicap if r["side"] == "home"]
        assert len(home_hc) == 1
        # P(margin > 5.5) when predicted margin = 8.5, std = 12
        # = 1 - Phi((-5.5 - 8.5)/12) = 1 - Phi(-1.17) ≈ 0.879
        assert home_hc[0]["model_prob"] > 0.8

    def test_no_handicap_when_missing(self):
        detector = ValueDetector(min_edge=0.0)
        results = detector.evaluate(
            prediction={"margin": 8.5, "total_goals": 120.0, "win_prob": 0.7,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.30, "away_odds": 3.50},
        )
        handicap = [r for r in results if r["market"] == "handicap"]
        assert len(handicap) == 0


class TestTotalValue:
    def test_over_value(self):
        detector = ValueDetector(min_edge=0.0)
        results = detector.evaluate(
            prediction={"margin": 5.0, "total_goals": 130.0, "win_prob": 0.6,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.50, "away_odds": 2.60,
                  "total_line": 125.5, "over_odds": 1.87, "under_odds": 1.87},
        )
        totals = [r for r in results if r["market"] == "total"]
        assert len(totals) >= 1
        over = [r for r in totals if r["side"] == "over"]
        assert len(over) == 1
        # P(total > 125.5) when predicted = 130, std = 15
        # = 1 - Phi((125.5-130)/15) = 1 - Phi(-0.3) ≈ 0.618
        assert 0.55 < over[0]["model_prob"] < 0.70

    def test_no_total_when_missing(self):
        detector = ValueDetector(min_edge=0.0)
        results = detector.evaluate(
            prediction={"margin": 5.0, "total_goals": 130.0, "win_prob": 0.6,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.50, "away_odds": 2.60},
        )
        totals = [r for r in results if r["market"] == "total"]
        assert len(totals) == 0


class TestCalibrationModel:
    def test_total_residual_std(self):
        import numpy as np
        from netball_model.model.calibration import CalibrationModel
        cal = CalibrationModel()
        margin_residuals = np.array([1.0, -2.0, 3.0, -1.0])
        total_residuals = np.array([5.0, -3.0, 4.0, -6.0])
        cal.fit(margin_residuals, total_residuals=total_residuals)
        assert cal.residual_std > 0
        assert cal.total_residual_std > 0
        assert abs(cal.total_residual_std - np.std(total_residuals)) < 0.01

    def test_backward_compat_no_total(self):
        import numpy as np
        from netball_model.model.calibration import CalibrationModel
        cal = CalibrationModel()
        cal.fit(np.array([1.0, -2.0, 3.0]))
        assert cal.residual_std > 0
        assert cal.total_residual_std == 10.0  # default
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/value/test_detector.py -v
```

- [ ] **Step 3: Implement CalibrationModel extension**

Modify `src/netball_model/model/calibration.py`:

```python
from __future__ import annotations

import numpy as np
from scipy.stats import norm


class CalibrationModel:
    def __init__(self):
        self.residual_std: float = 10.0  # default
        self.total_residual_std: float = 10.0  # default

    def fit(self, residuals: np.ndarray, total_residuals: np.ndarray | None = None):
        self.residual_std = float(np.std(residuals))
        if total_residuals is not None:
            self.total_residual_std = float(np.std(total_residuals))

    def win_probability(self, predicted_margin: float) -> float:
        """P(actual_margin > 0) given predicted_margin."""
        return float(norm.cdf(predicted_margin / self.residual_std))
```

- [ ] **Step 4: Implement new ValueDetector**

Rewrite `src/netball_model/value/detector.py`:

```python
from __future__ import annotations

from scipy.stats import norm


class ValueDetector:
    def __init__(self, min_edge: float = 0.05):
        self.min_edge = min_edge

    def evaluate(
        self,
        prediction: dict,
        odds: dict,
        threshold: float | None = None,
    ) -> list[dict]:
        """Evaluate all available markets for value.

        Args:
            prediction: {"margin", "total_goals", "win_prob",
                         "residual_std", "total_residual_std"}
            odds: {"home_odds", "away_odds", optionally "handicap_line",
                   "handicap_home_odds", "handicap_away_odds",
                   "total_line", "over_odds", "under_odds"}
            threshold: Override min_edge for this call
        """
        edge_threshold = threshold if threshold is not None else self.min_edge
        results = []

        # H2H
        results.extend(self._evaluate_h2h(prediction, odds, edge_threshold))

        # Handicap
        if odds.get("handicap_line") is not None and odds.get("handicap_home_odds"):
            results.extend(self._evaluate_handicap(prediction, odds, edge_threshold))

        # Totals
        if odds.get("total_line") is not None and odds.get("over_odds"):
            results.extend(self._evaluate_total(prediction, odds, edge_threshold))

        return results

    def _evaluate_h2h(self, prediction: dict, odds: dict, threshold: float) -> list[dict]:
        results = []
        home_odds = odds.get("home_odds")
        away_odds = odds.get("away_odds")
        win_prob = prediction["win_prob"]

        if home_odds:
            implied = 1 / home_odds
            edge = win_prob - implied
            results.append({
                "market": "h2h", "side": "home",
                "model_prob": round(win_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": home_odds, "line": None,
            })

        if away_odds:
            away_prob = 1 - win_prob
            implied = 1 / away_odds
            edge = away_prob - implied
            results.append({
                "market": "h2h", "side": "away",
                "model_prob": round(away_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": away_odds, "line": None,
            })

        return results

    def _evaluate_handicap(self, prediction: dict, odds: dict, threshold: float) -> list[dict]:
        results = []
        margin = prediction["margin"]
        sigma = prediction["residual_std"]
        line = odds["handicap_line"]

        # P(home covers) = 1 - Phi((L - M) / sigma)
        home_prob = 1 - norm.cdf((line - margin) / sigma)
        away_prob = 1 - home_prob

        home_odds = odds.get("handicap_home_odds")
        if home_odds:
            implied = 1 / home_odds
            edge = home_prob - implied
            results.append({
                "market": "handicap", "side": "home",
                "model_prob": round(home_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": home_odds, "line": line,
            })

        away_odds = odds.get("handicap_away_odds")
        if away_odds:
            implied = 1 / away_odds
            edge = away_prob - implied
            results.append({
                "market": "handicap", "side": "away",
                "model_prob": round(away_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": away_odds, "line": -line,
            })

        return results

    def _evaluate_total(self, prediction: dict, odds: dict, threshold: float) -> list[dict]:
        results = []
        total = prediction["total_goals"]
        sigma = prediction["total_residual_std"]
        line = odds["total_line"]

        # P(over) = 1 - Phi((L - T) / sigma)
        over_prob = 1 - norm.cdf((line - total) / sigma)
        under_prob = 1 - over_prob

        over_odds = odds.get("over_odds")
        if over_odds:
            implied = 1 / over_odds
            edge = over_prob - implied
            results.append({
                "market": "total", "side": "over",
                "model_prob": round(over_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": over_odds, "line": line,
            })

        under_odds = odds.get("under_odds")
        if under_odds:
            implied = 1 / under_odds
            edge = under_prob - implied
            results.append({
                "market": "total", "side": "under",
                "model_prob": round(under_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": under_odds, "line": line,
            })

        return results
```

- [ ] **Step 5: Run value detection tests**

```bash
poetry run pytest tests/value/test_detector.py -v
```

- [ ] **Step 6: Update callers of old evaluate() signature**

The old `evaluate()` is called in `cli.py:267-273` and `app.py`. These will be updated in the integration task (Task 6). For now, ensure the new tests pass.

- [ ] **Step 7: Run full test suite**

```bash
poetry run pytest -v
```

Note: `tests/test_cli.py` and other callers of the old signature may break. That's expected — they'll be fixed in Task 6 (Integration).

---

## Chunk 2: Wave 2 Tasks (After squads-agent completes)

### Task 4: Player Glicko-2 + Named Matchup Rankings

**Agent:** `player-elo-agent`

**Files:**
- Create: `src/netball_model/features/player_elo.py`
- Create: `src/netball_model/features/player_rankings.py`
- Create: `tests/features/test_player_elo.py`
- Create: `tests/features/test_player_rankings.py`
- Modify: `src/netball_model/data/database.py:29-107` (add player_elo_ratings table + new methods)

**Context for agent:** Uses the `glicko2` library (already installed). Each player gets a Glicko-2 rating per position. Matchup outcomes are determined by computing interaction features from `matchups.py` functions (`_gs_vs_gk`, `_ga_vs_gd`, etc.). The `PlayerProfiler` from `player_profile.py` computes rolling stats. Bidirectional: 9 pairings per match. MOV scaling: `log(|feature_sum| + 1)`. See spec Section 3 for full details.

- [ ] **Step 1: Add player_elo_ratings table to database.py**

In `src/netball_model/data/database.py`, add to the `initialize()` method's `CREATE TABLE` block (after line 101):

```sql
CREATE TABLE IF NOT EXISTS player_elo_ratings (
    player_id INTEGER,
    player_name TEXT,
    position TEXT,
    pool TEXT DEFAULT 'ssn',
    match_id TEXT,
    rating REAL,
    rd REAL,
    vol REAL,
    PRIMARY KEY (player_id, position, pool, match_id)
);
CREATE INDEX IF NOT EXISTS idx_player_elo ON player_elo_ratings(player_id, position, pool);
```

Add new methods after `get_latest_elo` (line 313):

```python
def upsert_player_elo(self, elo: dict):
    with self.connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO player_elo_ratings
               (player_id, player_name, position, pool, match_id, rating, rd, vol)
               VALUES (:player_id, :player_name, :position, :pool, :match_id, :rating, :rd, :vol)""",
            elo,
        )

def get_latest_player_elo(self, player_id: int, position: str, pool: str = "ssn") -> dict | None:
    with self.connection() as conn:
        cursor = conn.execute(
            """SELECT * FROM player_elo_ratings
               WHERE player_id = ? AND position = ? AND pool = ?
               ORDER BY rowid DESC LIMIT 1""",
            (player_id, position, pool),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

def get_all_player_elos(self, pool: str = "ssn") -> list[dict]:
    with self.connection() as conn:
        cursor = conn.execute(
            """SELECT pe.* FROM player_elo_ratings pe
               INNER JOIN (
                   SELECT player_id, position, MAX(rowid) as max_id
                   FROM player_elo_ratings WHERE pool = ?
                   GROUP BY player_id, position
               ) latest ON pe.rowid = latest.max_id""",
            (pool,),
        )
        return [dict(row) for row in cursor.fetchall()]
```

- [ ] **Step 2: Write player Glicko-2 tests**

Create `tests/features/test_player_elo.py`:

```python
import pytest
import math
from netball_model.features.player_elo import PlayerGlicko2, PlayerRating


class TestPlayerGlicko2Init:
    def test_new_player_defaults(self):
        pg = PlayerGlicko2()
        r = pg.get_rating(999, "GS")
        assert r.rating == 1500.0
        assert r.rd == 350.0
        assert abs(r.vol - 0.06) < 0.001

    def test_get_matchup_prediction_equal_ratings(self):
        pg = PlayerGlicko2()
        prob = pg.get_matchup_prediction(1, "GS", 2, "GK")
        assert abs(prob - 0.5) < 0.01


class TestProcessMatch:
    def test_updates_all_pairings(self):
        pg = PlayerGlicko2()
        home_starters = [
            {"player_id": i, "player_name": f"H{i}", "team": "A", "position": pos,
             "goals": 10, "attempts": 15, "assists": 5, "rebounds": 3, "feeds": 8,
             "turnovers": 2, "gains": 3, "intercepts": 2, "deflections": 2,
             "penalties": 1, "centre_pass_receives": 4, "net_points": 10}
            for i, pos in enumerate(["GS", "GA", "WA", "C", "WD", "GD", "GK"], 1)
        ]
        away_starters = [
            {"player_id": i, "player_name": f"A{i}", "team": "B", "position": pos,
             "goals": 5, "attempts": 10, "assists": 3, "rebounds": 2, "feeds": 5,
             "turnovers": 4, "gains": 1, "intercepts": 1, "deflections": 1,
             "penalties": 2, "centre_pass_receives": 3, "net_points": 5}
            for i, pos in enumerate(["GS", "GA", "WA", "C", "WD", "GD", "GK"], 8)
        ]
        match = {"match_id": "m1", "season": 2024}

        pg.process_match(match, home_starters, away_starters)

        # All 14 players should have ratings updated from default
        for pid in range(1, 15):
            pos = ["GS", "GA", "WA", "C", "WD", "GD", "GK"][(pid - 1) % 7]
            r = pg.get_rating(pid, pos)
            # RD should decrease from 350 (uncertainty reduced after match)
            assert r.rd < 350.0

    def test_winner_rating_increases(self):
        pg = PlayerGlicko2()
        # Create starters where home GS dominates (high goals, low turnovers)
        home_starters = [
            {"player_id": 1, "player_name": "H1", "team": "A", "position": "GS",
             "goals": 30, "attempts": 35, "assists": 0, "rebounds": 5, "feeds": 2,
             "turnovers": 0, "gains": 0, "intercepts": 0, "deflections": 0,
             "penalties": 0, "centre_pass_receives": 0, "net_points": 30},
        ] + [
            {"player_id": i, "player_name": f"H{i}", "team": "A", "position": pos,
             "goals": 0, "attempts": 0, "assists": 0, "rebounds": 0, "feeds": 0,
             "turnovers": 0, "gains": 0, "intercepts": 0, "deflections": 0,
             "penalties": 0, "centre_pass_receives": 0, "net_points": 0}
            for i, pos in enumerate(["GA", "WA", "C", "WD", "GD", "GK"], 2)
        ]
        away_starters = [
            {"player_id": i, "player_name": f"A{i}", "team": "B", "position": pos,
             "goals": 0, "attempts": 0, "assists": 0, "rebounds": 0, "feeds": 0,
             "turnovers": 5, "gains": 0, "intercepts": 0, "deflections": 0,
             "penalties": 5, "centre_pass_receives": 0, "net_points": 0}
            for i, pos in enumerate(["GS", "GA", "WA", "C", "WD", "GD", "GK"], 8)
        ]
        pg.process_match({"match_id": "m1", "season": 2024}, home_starters, away_starters)

        # Home GS should gain rating (dominated the GS vs GK matchup)
        assert pg.get_rating(1, "GS").rating > 1500.0


class TestSeasonReset:
    def test_regress_ratings(self):
        pg = PlayerGlicko2()
        pg._ratings[(1, "GS")] = PlayerRating(rating=1700, rd=50, vol=0.06)
        pg.regress_ratings(factor=0.2, mean=1500.0)
        expected = 1700 * 0.8 + 1500 * 0.2
        assert abs(pg.get_rating(1, "GS").rating - expected) < 0.1

    def test_increase_rd(self):
        pg = PlayerGlicko2()
        pg._ratings[(1, "GS")] = PlayerRating(rating=1600, rd=50, vol=0.06)
        pg.increase_rd(amount=30)
        assert abs(pg.get_rating(1, "GS").rd - 80) < 0.1

    def test_increase_rd_capped(self):
        pg = PlayerGlicko2()
        pg._ratings[(1, "GS")] = PlayerRating(rating=1600, rd=340, vol=0.06)
        pg.increase_rd(amount=30)
        assert pg.get_rating(1, "GS").rd == 350.0
```

- [ ] **Step 3: Implement PlayerGlicko2**

Create `src/netball_model/features/player_elo.py`:

```python
"""Individual player Glicko-2 ratings based on positional matchups."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import glicko2

from netball_model.features.player_profile import PlayerProfiler
from netball_model.features.matchups import (
    _gs_vs_gk, _ga_vs_gd, _wa_vs_wd, _c_vs_c, _wd_vs_wa,
)


@dataclass
class PlayerRating:
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06


# 9 bidirectional pairings: (attacker_pos, defender_pos, matchup_fn, win_keys)
_PAIRINGS = [
    # Home attacking -> Away defending
    ("GS", "GK", _gs_vs_gk, ["gs_vs_gk_shooting_pressure"]),
    ("GA", "GD", _ga_vs_gd, ["ga_vs_gd_shooting_pressure", "ga_vs_gd_feed_vs_intercept"]),
    ("WA", "WD", _wa_vs_wd, ["wa_vs_wd_delivery_vs_disruption", "wa_vs_wd_supply_line"]),
    # Midcourt
    ("C", "C", _c_vs_c, ["c_vs_c_distribution_battle", "c_vs_c_disruption_battle"]),
    ("WD", "WA", _wd_vs_wa, ["wd_vs_wa_pressure_effectiveness", "wd_vs_wa_aerial_battle"]),
]


class PlayerGlicko2:
    def __init__(self):
        self._ratings: dict[tuple[int, str], PlayerRating] = {}

    def get_rating(self, player_id: int, position: str) -> PlayerRating:
        key = (player_id, position)
        if key not in self._ratings:
            self._ratings[key] = PlayerRating()
        return self._ratings[key]

    def process_match(
        self, match: dict, home_starters: list[dict], away_starters: list[dict]
    ) -> None:
        """Update player ratings based on positional matchup outcomes."""
        profiler = PlayerProfiler()

        # Build profiles: {position: profile_dict}
        home_profiles = self._build_profiles(home_starters, profiler)
        away_profiles = self._build_profiles(away_starters, profiler)

        # Build player_id lookup: {position: player_id}
        home_ids = {s["position"]: s["player_id"] for s in home_starters}
        away_ids = {s["position"]: s["player_id"] for s in away_starters}

        # Forward pairings: home attacks, away defends
        for att_pos, def_pos, fn, keys in _PAIRINGS:
            if att_pos not in home_profiles or def_pos not in away_profiles:
                continue
            features = fn(home_profiles[att_pos], away_profiles[def_pos])
            win_val = sum(features.get(k, 0) for k in keys) / len(keys)
            mov_scale = math.log(abs(win_val) + 1)
            att_id, def_id = home_ids.get(att_pos), away_ids.get(def_pos)
            if att_id is not None and def_id is not None:
                self._update_pair(att_id, att_pos, def_id, def_pos, win_val > 0, mov_scale)

        # Reverse pairings: away attacks, home defends
        for att_pos, def_pos, fn, keys in _PAIRINGS:
            if att_pos == "C" and def_pos == "C":
                continue  # C vs C already handled (symmetric)
            if att_pos == "WD" and def_pos == "WA":
                continue  # WD vs WA already covered in forward pass
            if att_pos not in away_profiles or def_pos not in home_profiles:
                continue
            features = fn(away_profiles[att_pos], home_profiles[def_pos])
            win_val = sum(features.get(k, 0) for k in keys) / len(keys)
            mov_scale = math.log(abs(win_val) + 1)
            att_id, def_id = away_ids.get(att_pos), home_ids.get(def_pos)
            if att_id is not None and def_id is not None:
                self._update_pair(att_id, att_pos, def_id, def_pos, win_val > 0, mov_scale)

    def _build_profiles(self, starters: list[dict], profiler: PlayerProfiler) -> dict:
        """Build {position: profile_dict} from starter stat rows."""
        profiles = {}
        for s in starters:
            pos = s["position"]
            if pos == "-":
                continue
            profiles[pos] = profiler.compute_profile([s], pos)
        return profiles

    def _update_pair(
        self, att_id: int, att_pos: str, def_id: int, def_pos: str,
        att_wins: bool, mov_scale: float
    ) -> None:
        """Apply Glicko-2 update to a matched pair of players."""
        att_r = self.get_rating(att_id, att_pos)
        def_r = self.get_rating(def_id, def_pos)

        att_g2 = glicko2.Player(rating=att_r.rating, rd=att_r.rd, vol=att_r.vol)
        def_g2 = glicko2.Player(rating=def_r.rating, rd=def_r.rd, vol=def_r.vol)

        att_score = 1.0 if att_wins else 0.0
        def_score = 1.0 - att_score

        # Scale the rating change by margin of victory
        att_g2.update_player([def_r.rating], [def_r.rd], [att_score])
        def_g2.update_player([att_r.rating], [att_r.rd], [def_score])

        # Apply MOV scaling to the rating delta
        att_delta = (att_g2.rating - att_r.rating) * max(mov_scale, 0.5)
        def_delta = (def_g2.rating - def_r.rating) * max(mov_scale, 0.5)

        att_r.rating = att_r.rating + att_delta
        att_r.rd = att_g2.rd
        att_r.vol = att_g2.vol
        def_r.rating = def_r.rating + def_delta
        def_r.rd = def_g2.rd
        def_r.vol = def_g2.vol

    def get_matchup_prediction(
        self, player_a_id: int, pos_a: str, player_b_id: int, pos_b: str
    ) -> float:
        """Predict win probability for player A vs player B."""
        a = self.get_rating(player_a_id, pos_a)
        b = self.get_rating(player_b_id, pos_b)
        expected = 1 / (1 + 10 ** ((b.rating - a.rating) / 400))
        return expected

    def regress_ratings(self, factor: float = 0.2, mean: float = 1500.0) -> None:
        """Regress all player ratings toward the mean by factor (season reset)."""
        for pr in self._ratings.values():
            pr.rating = pr.rating * (1 - factor) + mean * factor

    def increase_rd(self, amount: float = 30.0) -> None:
        """Increase RD for all players (off-season uncertainty)."""
        for pr in self._ratings.values():
            pr.rd = min(pr.rd + amount, 350.0)
```

- [ ] **Step 4: Run player elo tests**

```bash
poetry run pytest tests/features/test_player_elo.py -v
```

- [ ] **Step 5: Write and run database tests for player elo methods**

Add to `tests/data/test_database.py` (or create if needed):

```python
def test_upsert_and_get_player_elo(tmp_db):
    db = tmp_db
    db.upsert_player_elo({
        "player_id": 1, "player_name": "Test Player", "position": "GS",
        "pool": "ssn", "match_id": "m1", "rating": 1600.0, "rd": 80.0, "vol": 0.06
    })
    result = db.get_latest_player_elo(1, "GS")
    assert result is not None
    assert result["rating"] == 1600.0

def test_get_all_player_elos(tmp_db):
    db = tmp_db
    for pid, rating in [(1, 1600), (2, 1400)]:
        db.upsert_player_elo({
            "player_id": pid, "player_name": f"P{pid}", "position": "GS",
            "pool": "ssn", "match_id": "m1", "rating": rating, "rd": 80, "vol": 0.06
        })
    results = db.get_all_player_elos()
    assert len(results) == 2
```

Run: `poetry run pytest tests/data/test_database.py -v`

- [ ] **Step 6: Write player rankings tests**

Create `tests/features/test_player_rankings.py` with tests for `get_position_rankings`, `get_matchup_prediction`, `get_team_matchup_report`.

- [ ] **Step 6: Implement player rankings**

Create `src/netball_model/features/player_rankings.py` — wrapper around `PlayerGlicko2` that provides named lookup.

- [ ] **Step 7: Run all tests**

```bash
poetry run pytest tests/features/test_player_elo.py tests/features/test_player_rankings.py -v
poetry run pytest -v
```

---

### Task 5: Time-Weighted Training + Round Number

**Agent:** `time-weight-agent`

**Files:**
- Modify: `src/netball_model/model/train.py:13-15,26-43` (NON_FEATURE_COLUMNS + train method)
- Modify: `src/netball_model/features/elo.py:16-19` (add regress_ratings, increase_rd)
- Modify: `src/netball_model/features/contextual.py:55-63` (add round_number methods)
- Create: `tests/model/test_time_weighting.py`

**Context for agent:** Ridge `.fit()` supports `sample_weight` parameter. Weights: `exp(-0.5 * years_ago) * (0.5 + 0.5 * roster_continuity)`. The `_sample_weight` key goes in the feature row dict and into `NON_FEATURE_COLUMNS`. `GlickoSystem` needs `regress_ratings(factor, mean)` and `increase_rd(amount)` methods. `ContextualFeatures` needs `round_features(match)` returning `round_number` and `season_progress`. See spec Section 4.

- [ ] **Step 1: Write time weighting tests**

Create `tests/model/test_time_weighting.py`:

```python
import math
import numpy as np
import pandas as pd
import pytest
from netball_model.model.train import NetballModel, NON_FEATURE_COLUMNS


def test_sample_weight_in_non_feature_columns():
    assert "_sample_weight" in NON_FEATURE_COLUMNS


def test_weight_calculation():
    """Verify exponential decay formula."""
    lambda_ = 0.5
    assert abs(math.exp(-lambda_ * 0) - 1.0) < 0.01
    assert abs(math.exp(-lambda_ * 1) - 0.607) < 0.01
    assert abs(math.exp(-lambda_ * 2) - 0.368) < 0.01


def test_train_with_sample_weights(dummy_feature_df):
    """Model should accept _sample_weight column without error."""
    df = dummy_feature_df(100)
    df["_sample_weight"] = np.random.default_rng(42).uniform(0.1, 1.0, 100)

    model = NetballModel()
    model.train(df)

    assert "_sample_weight" not in model.feature_columns
    preds = model.predict(df.drop(columns=["_sample_weight"]))
    assert len(preds) == 100


def test_train_without_sample_weights_still_works(dummy_feature_df):
    """Backward compatibility: train without _sample_weight."""
    df = dummy_feature_df(100)
    model = NetballModel()
    model.train(df)
    assert len(model.feature_columns) > 0
```

- [ ] **Step 2: Write Glicko-2 season reset tests**

Add to `tests/features/test_elo.py`:

```python
def test_regress_ratings():
    from netball_model.features.elo import GlickoSystem
    gs = GlickoSystem()
    gs.update("A", "B", "home", margin=10, pool="ssn")
    rating_before = gs.get_rating("A", "ssn")["rating"]
    gs.regress_ratings(factor=0.2, mean=1500, pool="ssn")
    rating_after = gs.get_rating("A", "ssn")["rating"]
    expected = rating_before * 0.8 + 1500 * 0.2
    assert abs(rating_after - expected) < 0.1


def test_increase_rd():
    from netball_model.features.elo import GlickoSystem
    gs = GlickoSystem()
    gs.update("A", "B", "home", margin=10, pool="ssn")
    rd_before = gs.get_rating("A", "ssn")["rd"]
    gs.increase_rd(amount=30, pool="ssn")
    rd_after = gs.get_rating("A", "ssn")["rd"]
    assert abs(rd_after - (rd_before + 30)) < 0.1
```

- [ ] **Step 3: Write round number tests**

Add to `tests/features/test_contextual.py`:

```python
def test_round_features():
    from netball_model.features.contextual import ContextualFeatures
    matches = [
        {"home_team": "A", "away_team": "B", "season": 2024, "round_num": 3,
         "date": "2024-04-01", "home_score": 60, "away_score": 55, "venue": ""},
    ]
    ctx = ContextualFeatures(matches)
    result = ctx.round_features(matches[0])
    assert result["round_number"] == 3
    assert abs(result["season_progress"] - 3 / 14) < 0.01
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
poetry run pytest tests/model/test_time_weighting.py tests/features/test_elo.py::test_regress_ratings tests/features/test_elo.py::test_increase_rd tests/features/test_contextual.py::test_round_features -v
```

- [ ] **Step 5: Implement GlickoSystem.regress_ratings and increase_rd**

Add to `src/netball_model/features/elo.py` (after `get_all_ratings`, line 119):

```python
def regress_ratings(self, factor: float = 0.2, mean: float = 1500.0, pool: str = "ssn"):
    """Regress all team ratings toward the mean by factor."""
    if pool not in self._ratings:
        return
    for tr in self._ratings[pool].values():
        tr.rating = tr.rating * (1 - factor) + mean * factor

def increase_rd(self, amount: float = 30.0, pool: str = "ssn"):
    """Increase rating deviation for all teams (off-season uncertainty)."""
    if pool not in self._ratings:
        return
    for tr in self._ratings[pool].values():
        tr.rd = min(tr.rd + amount, 350.0)  # cap at initial RD
```

- [ ] **Step 6: Implement ContextualFeatures.round_features**

Add to `src/netball_model/features/contextual.py` (after `is_home`, line 187):

```python
def round_features(self, match: dict) -> dict:
    """Extract round number and season progress."""
    round_num = match.get("round_num", 1)
    season = match.get("season")

    # Determine max rounds for this season
    max_rounds = 14  # default
    if season and self.matches:
        season_rounds = [m.get("round_num", 0) for m in self.matches if m.get("season") == season]
        if season_rounds:
            max_rounds = max(max_rounds, max(season_rounds))

    return {
        "round_number": round_num,
        "season_progress": round_num / max_rounds if max_rounds > 0 else 0.0,
    }
```

- [ ] **Step 7: Implement sample weight support in NetballModel.train**

Modify `src/netball_model/model/train.py`:

Add `"_sample_weight"` to `NON_FEATURE_COLUMNS` (line 14):

```python
NON_FEATURE_COLUMNS = {
    "match_id", "home_team", "away_team", "margin", "total_goals", "_sample_weight",
}
```

Modify `train()` method to extract and use sample weights:

```python
def train(self, df: pd.DataFrame):
    self.feature_columns = [
        c for c in df.columns if c not in NON_FEATURE_COLUMNS
    ]

    X = df[self.feature_columns].values.astype(float)
    y_margin = df["margin"].values.astype(float)
    y_total = df["total_goals"].values.astype(float)

    # Extract sample weights if present
    sample_weight = None
    if "_sample_weight" in df.columns:
        sample_weight = df["_sample_weight"].values.astype(float)

    X_scaled = self.scaler.fit_transform(X)

    self.margin_model.fit(X_scaled, y_margin, sample_weight=sample_weight)
    self.total_model.fit(X_scaled, y_total, sample_weight=sample_weight)

    # Calibrate on training residuals
    margin_preds = self.margin_model.predict(X_scaled)
    margin_residuals = y_margin - margin_preds

    total_preds = self.total_model.predict(X_scaled)
    total_residuals = y_total - total_preds

    # Note: CalibrationModel.fit() is extended in Task 3 (Value Detection)
    # to accept total_residuals. Keep backward-compatible call here;
    # the integration task (Task 6) wires total_residuals after merging.
    self.calibration.fit(margin_residuals)
```

- [ ] **Step 8: Run all tests**

```bash
poetry run pytest tests/model/test_time_weighting.py tests/features/test_elo.py tests/features/test_contextual.py -v
poetry run pytest -v
```

---

## Chunk 3: Integration

### Task 6: Wire Everything Together

**Agent:** `integration-agent` (runs on `feature/player-matchups` branch after all worktrees merged)

**Files:**
- Modify: `src/netball_model/features/builder.py` (add player elo features, round features, sample weights, season reset)
- Modify: `src/netball_model/services.py:41-60` (train_model loads player stats, passes weights)
- Modify: `src/netball_model/cli.py:236-282` (update predict command for new ValueDetector signature)
- Modify: `app.py` (add screenshot upload, expand value table)
- Modify: `tests/test_services.py` (update for new train_model behavior)

**Context for agent:** This task wires the 5 independent subsystems into the existing codebase. The FeatureBuilder needs to: (1) drive PlayerGlicko2 alongside team GlickoSystem, (2) add round_number/season_progress features, (3) compute _sample_weight per row, (4) trigger season resets for both team and player ratings. The services.py train_model() needs to load player_stats and pass them to FeatureBuilder. The CLI predict command needs the new ValueDetector dict-based API. The Streamlit app needs screenshot upload and multi-market value display.

- [ ] **Step 1: Merge all worktree branches into feature/player-matchups**

Merge each completed branch:
```bash
git merge feature/screenshot-ocr --no-ff
git merge feature/squads-movements --no-ff
git merge feature/value-markets --no-ff
git merge feature/player-elo --no-ff
git merge feature/time-weighting --no-ff
```

Resolve any conflicts. **Known conflict**: both `feature/value-markets` and `feature/time-weighting` modify `train.py` — the merged result must include both `_sample_weight` in `NON_FEATURE_COLUMNS` and `total_residuals` in the calibration call.

- [ ] **Step 2: Wire CalibrationModel.fit() total_residuals**

After merging, update `train.py`'s `train()` method to call `self.calibration.fit(margin_residuals, total_residuals=total_residuals)` — this combines the CalibrationModel extension from Task 3 with the total_residuals computation from Task 5.

```python
    # In train() after computing residuals:
    self.calibration.fit(margin_residuals, total_residuals=total_residuals)
```

- [ ] **Step 3: Update FeatureBuilder.__init__**

Modify `src/netball_model/features/builder.py` to accept player stats and roster continuity:

```python
from netball_model.features.player_elo import PlayerGlicko2
from netball_model.data.player_movements import get_team_continuity_all

class FeatureBuilder:
    def __init__(self, matches, pool="ssn", player_stats=None, roster_continuity=None):
        # ... existing init ...
        self._player_stats = player_stats  # dict: match_id -> list[dict]
        self._player_glicko = PlayerGlicko2() if player_stats else None
        self._player_elo_computed_up_to = -1
        self._roster_continuity = roster_continuity or {}  # dict: (team, season) -> float
```

- [ ] **Step 4: Add season reset + player elo replay to _ensure_elo_up_to**

The existing `_ensure_elo_up_to()` replays team Glicko. Extend it to also:
1. Detect season boundaries (when `match["season"] != prev_match["season"]`)
2. Call `regress_ratings` and `increase_rd` on both team GlickoSystem AND PlayerGlicko2
3. Replay `PlayerGlicko2.process_match()` for each match alongside team updates

```python
def _ensure_elo_up_to(self, target_index: int):
    start = self._elo_computed_up_to + 1
    prev_season = self.matches[start - 1]["season"] if start > 0 else None

    for i in range(start, target_index):
        m = self.matches[i]

        # Season boundary detection
        if prev_season is not None and m["season"] != prev_season:
            self.glicko.regress_ratings(factor=0.2, mean=1500.0, pool=self.pool)
            self.glicko.increase_rd(amount=30.0, pool=self.pool)
            if self._player_glicko:
                self._player_glicko.regress_ratings(factor=0.2, mean=1500.0)
                self._player_glicko.increase_rd(amount=30.0)

        # Existing team Glicko update
        if m.get("home_score") is not None:
            winner = "home" if m["home_score"] > m["away_score"] else "away"
            margin = m["home_score"] - m["away_score"]
            self.glicko.update(m["home_team"], m["away_team"], winner, margin, self.pool)

        # Player Glicko update
        if self._player_glicko and self._player_stats:
            starters = self._player_stats.get(m["match_id"], [])
            home_starters = [s for s in starters if s["team"] == m["home_team"] and s["position"] != "-"]
            away_starters = [s for s in starters if s["team"] == m["away_team"] and s["position"] != "-"]
            if home_starters and away_starters:
                self._player_glicko.process_match(m, home_starters, away_starters)

        prev_season = m["season"]

    self._elo_computed_up_to = target_index - 1
    if self._player_glicko:
        self._player_elo_computed_up_to = target_index - 1
```

- [ ] **Step 5: Add _build_player_elo_features and _compute_sample_weight to FeatureBuilder**

```python
def _build_player_elo_features(self, match_index: int) -> dict:
    """Compute 8 player elo features for the match at match_index."""
    m = self.matches[match_index]
    starters = self._player_stats.get(m["match_id"], [])
    home_starters = [s for s in starters if s["team"] == m["home_team"] and s["position"] != "-"]
    away_starters = [s for s in starters if s["team"] == m["away_team"] and s["position"] != "-"]

    home_ratings = [self._player_glicko.get_rating(s["player_id"], s["position"]).rating for s in home_starters]
    away_ratings = [self._player_glicko.get_rating(s["player_id"], s["position"]).rating for s in away_starters]

    home_avg = sum(home_ratings) / len(home_ratings) if home_ratings else 1500.0
    away_avg = sum(away_ratings) / len(away_ratings) if away_ratings else 1500.0

    features = {
        "home_player_elo_avg": home_avg,
        "away_player_elo_avg": away_avg,
        "player_elo_diff": home_avg - away_avg,
    }

    # 5 positional diffs
    pos_pairs = [("GS", "GK"), ("GA", "GD"), ("WA", "WD"), ("C", "C"), ("WD", "WA")]
    home_by_pos = {s["position"]: s for s in home_starters}
    away_by_pos = {s["position"]: s for s in away_starters}

    for att_pos, def_pos in pos_pairs:
        key = f"{att_pos.lower()}_{def_pos.lower()}_player_elo_diff"
        h = home_by_pos.get(att_pos)
        a = away_by_pos.get(def_pos)
        if h and a:
            hr = self._player_glicko.get_rating(h["player_id"], att_pos).rating
            ar = self._player_glicko.get_rating(a["player_id"], def_pos).rating
            features[key] = hr - ar
        else:
            features[key] = 0.0

    return features


def _compute_sample_weight(self, match_index: int) -> float:
    """Compute time-decay * roster-continuity weight for a match."""
    import math
    m = self.matches[match_index]
    current_season = self.matches[-1].get("season", m.get("season", 2025))
    match_season = m.get("season", current_season)
    years_ago = current_season - match_season

    base_weight = math.exp(-0.5 * years_ago)

    # Roster continuity adjustment (average of home and away)
    home_cont = self._roster_continuity.get((m["home_team"], match_season), 1.0)
    away_cont = self._roster_continuity.get((m["away_team"], match_season), 1.0)
    avg_continuity = (home_cont + away_cont) / 2
    weight = base_weight * (0.5 + 0.5 * avg_continuity)

    return max(weight, 0.01)  # floor to avoid zero weights
```

- [ ] **Step 6: Add player elo features, round features, and sample weight to build_row**

After the existing matchup features block, add:

```python
# Player Elo features (8 features)
if self._player_glicko and self._player_stats:
    player_elo_features = self._build_player_elo_features(match_index)
    row.update(player_elo_features)

# Round features (2 features)
round_feats = self.ctx.round_features(m)
row.update(round_feats)

# Sample weight (excluded from model features via NON_FEATURE_COLUMNS)
row["_sample_weight"] = self._compute_sample_weight(match_index)
```

- [ ] **Step 7: Update services.py::train_model**

Modify `services.py` to load player stats and pass roster continuity to FeatureBuilder. Follow the same pattern as `backtest_season()` (lines 80-85):

```python
def train_model(db: Database, output: str = "data/model.pkl"):
    matches = db.get_matches()
    if not matches:
        raise ValueError("No matches in database")

    # Load player stats for all matches (same pattern as backtest_season)
    player_stats = {}
    for m in matches:
        starters = db.get_starters_for_match(m["match_id"])
        if starters:
            player_stats[m["match_id"]] = starters

    # Compute roster continuity for all seasons
    from netball_model.data.player_movements import get_team_continuity_all
    roster_continuity = {}
    seasons = sorted(set(m["season"] for m in matches))
    for season in seasons:
        if season == seasons[0]:
            continue  # No previous season to compare
        continuity = get_team_continuity_all(season, db)
        for team, cont in continuity.items():
            roster_continuity[(team, season)] = cont

    builder = FeatureBuilder(
        matches, player_stats=player_stats, roster_continuity=roster_continuity
    )
    # ... rest of training as before ...
```

- [ ] **Step 8: Expose residual_std and total_residual_std from NetballModel**

The new `ValueDetector.evaluate()` requires `prediction["residual_std"]` and `prediction["total_residual_std"]`. Add a property or method to `NetballModel`:

```python
# In NetballModel class:
@property
def residual_std(self) -> float:
    return self.calibration.residual_std

@property
def total_residual_std(self) -> float:
    return self.calibration.total_residual_std
```

- [ ] **Step 9: Update CLI predict command**

Change `cli.py:267-282` to construct the prediction and odds dicts for the new `ValueDetector.evaluate()` API:

```python
        prediction = {
            "margin": float(pred["predicted_margin"].iloc[0]),
            "total_goals": float(pred["predicted_total"].iloc[0]),
            "win_prob": float(pred["win_probability"].iloc[0]),
            "residual_std": model.residual_std,
            "total_residual_std": model.total_residual_std,
        }

        odds_dict = {}
        if stored_odds:
            odds_dict = {
                "home_odds": stored_odds.get("home_back_odds"),
                "away_odds": stored_odds.get("away_back_odds"),
                "handicap_line": stored_odds.get("handicap_line"),
                "handicap_home_odds": stored_odds.get("handicap_home_odds"),
                "handicap_away_odds": stored_odds.get("handicap_away_odds"),
                "total_line": stored_odds.get("total_line"),
                "over_odds": stored_odds.get("over_odds"),
                "under_odds": stored_odds.get("under_odds"),
            }

        value_bets = detector.evaluate(prediction, odds_dict)
```

- [ ] **Step 10: Update Streamlit app**

In `app.py`, add two integration points:

**Screenshot upload** (alongside the existing "Paste odds" expander around line 234):
```python
with st.expander("Upload bet365 screenshot"):
    uploaded = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg"])
    if uploaded and st.button("Parse screenshot"):
        import tempfile
        from netball_model.data.bet365_screenshot import parse_screenshot
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(uploaded.getvalue())
            parsed = parse_screenshot(f.name)
        if parsed:
            for book_key in ["home_odds", "away_odds"]:
                st.session_state[f"{book_key}_bet365_screenshot"] = parsed.get(book_key)
            st.success(f"Parsed: {parsed.get('home_team')} vs {parsed.get('away_team')}")
```

**Multi-market value table** — update the existing `detector.evaluate()` calls (around lines 296-303) to use the new dict-based API, and display the results in a table with columns: Market, Side, Model Prob, Implied Prob, Edge, Odds, Line.

Also update the manual match dict (line ~166-176) to include `"season": match_date.year` for proper round_features/weight computation.

- [ ] **Step 11: Run full test suite**

```bash
poetry run pytest -v
```

All 92+ existing tests must pass plus all new tests from Tasks 1-5.

- [ ] **Step 12: Manual smoke test**

```bash
poetry run netball train --db data/netball.db
poetry run netball predict --db data/netball.db
```

Verify the model trains with new features and predictions include line/total value.
