# Model Enhancement: Player Elo, Time Weighting, Markets & Screenshot OCR

**Goal:** Five enhancements to the netball prediction model — bet365 screenshot OCR, 2026 squad data with player movement tracking, individual player Glicko-2 ratings with named matchup rankings, time-weighted training with roster continuity, and line/total value detection.

**Implementation:** Designed for parallel execution using a team of agents (subagent-driven development). Each subsystem is independent enough for a separate agent, with clear integration points.

## Subsystem Overview

| # | Subsystem | New Files | Modified Files |
|---|-----------|-----------|---------------|
| 1 | Bet365 Screenshot OCR | `data/bet365_screenshot.py` | `app.py` (project root), `cli.py` |
| 2 | 2026 Squads + Player Movements | `data/squads.py`, `data/player_movements.py` | — |
| 3 | Player Glicko-2 + Matchup Rankings | `features/player_elo.py`, `features/player_rankings.py` | `features/builder.py`, `data/database.py` |
| 4 | Time-Weighted Training + Round Number | — | `model/train.py`, `features/elo.py`, `features/contextual.py`, `features/builder.py` |
| 5 | Line/Total Value Detection | — | `value/detector.py`, `app.py` (project root) |

New file paths relative to `src/netball_model/`. Note: `app.py` is the Streamlit app at the **project root** (`/app.py`), not under `src/`.

---

## 1. Bet365 Screenshot OCR

**Purpose:** Parse bet365 screenshots into structured odds data, replacing the fragile Playwright scraper.

**Module:** `src/netball_model/data/bet365_screenshot.py`

### Pipeline

1. **Preprocess** — Convert to grayscale, threshold to isolate text (yellow/white on dark background at ~35-40 luminance). Split image into sections by detecting the green/gray header bars ("Match Lines", "1st Half", "1st Quarter").

2. **OCR** — Use EasyOCR (better than Tesseract for styled text on dark backgrounds). Extract all text with bounding boxes (x, y, width, height coordinates).

3. **Structure** — Group text elements by vertical position (rows) and horizontal position (columns). The layout is a table:
   - Header row: team names (left-center = home, right-center = away)
   - Data rows: market label (leftmost), home value (middle), away value (right)
   - Market labels: "To Win", "Total", "Line"

4. **Parse** — Apply regex to extracted text per row type:
   - **To Win:** Two decimal odds — `(\d+\.\d{2})`
   - **Total:** `O (\d+\.?\d*) (\d+\.\d{2})` and `U (\d+\.?\d*) (\d+\.\d{2})`
   - **Line:** `([+-]\d+\.?\d*) (\d+\.\d{2})` (two instances, home and away)

5. **Team extraction** — Match title text "Team A v Team B" at the top of the image. Normalise via `normalise_team()`.

### Output Schema

Matches the existing scraper's key naming convention (27 keys including `match_date`). Keys for any section not visible in the screenshot are set to `None`.

```python
{
    "home_team": str,
    "away_team": str,
    "match_date": str | None,  # extracted from image if visible
    # Match Lines
    "home_odds": float | None,
    "away_odds": float | None,
    "handicap_line": float | None,
    "handicap_home_odds": float | None,  # note: handicap_ prefix per existing scraper
    "handicap_away_odds": float | None,
    "total_line": float | None,
    "over_odds": float | None,
    "under_odds": float | None,
    # 1st Half (h1_ prefix)
    "h1_home_odds": float | None,
    "h1_away_odds": float | None,
    "h1_handicap_line": float | None,
    "h1_handicap_home_odds": float | None,
    "h1_handicap_away_odds": float | None,
    "h1_total_line": float | None,
    "h1_over_odds": float | None,
    "h1_under_odds": float | None,
    # 1st Quarter (q1_ prefix)
    "q1_home_odds": float | None,
    "q1_away_odds": float | None,
    "q1_handicap_line": float | None,
    "q1_handicap_home_odds": float | None,
    "q1_handicap_away_odds": float | None,
    "q1_total_line": float | None,
    "q1_over_odds": float | None,
    "q1_under_odds": float | None,
}
```

**Error handling:** If a section is not visible in the screenshot (e.g., user only captured Match Lines), the corresponding keys are set to `None`. If OCR fails entirely, raise `ValueError` with a descriptive message.

### Public API

```python
def parse_screenshot(image_path: str | Path) -> dict:
    """Parse a bet365 screenshot and return structured odds data."""
```

### Integration

- **Streamlit app:** Add file upload widget alongside existing paste tool. Call `parse_screenshot()` on uploaded image, populate odds fields.
- **CLI:** Add `netball parse-screenshot <image_path>` command.

### Dependencies

- `easyocr` — add to pyproject.toml
- `Pillow` — already available via other deps

### Testing

- Save the provided screenshot as a test fixture
- Unit tests for the parsing/structuring logic with mocked OCR output (deterministic)
- Integration test with the real fixture image

---

## 2. 2026 Squads + Player Movement Tracking

**Purpose:** Store 2026 SSN rosters, map players to DB records, detect inter-season movements for roster continuity calculations.

### Part A: Squad Data

**Module:** `src/netball_model/data/squads.py`

```python
SQUADS_2026 = {
    "Adelaide Thunderbirds": {
        "GS": "Elmere van der Berg",
        "GA": "Georgie Horjus",
        "WA": "Kayla Graham",
        "C": "Kate Heffernan",
        "WD": "Latanya Wilson",
        "GD": "Matilda Garrett",
        "GK": "Shamera Sterling-Humphrey",
    },
    "GIANTS Netball": {
        "GS": "Lucy Austin",
        "GA": "Sophie Dwyer",
        "WA": "Whitney Souness",
        "C": "Hope White",
        "WD": "Amy Sligar",
        "GD": "Erin O'Brien",
        "GK": "Jane Watson",
    },
    "Melbourne Mavericks": {
        "GS": "Shimona Nelson",
        "GA": "Reilley Batcheldor",
        "WA": "Sacha McDonald",
        "C": "Jamie-Lee Price",
        "WD": "Amy Parmenter",
        "GD": "Kim Brown",
        "GK": "Tara Hinchliffe",
    },
    "Melbourne Vixens": {
        "GS": "Sophie Garbin",
        "GA": "Kiera Austin",
        "WA": "Hannah Mundy",
        "C": "Kate Moloney",
        "WD": "Kate Eddy",
        "GD": "Jo Weston",
        "GK": "Rudi Ellis",
    },
    "NSW Swifts": {
        "GS": "Grace Nweke",
        "GA": "Helen Housby",
        "WA": "Gina Crampton",
        "C": "Maddy Proud",
        "WD": "Maddy Turner",
        "GD": "Teigan O'Shannassy",
        "GK": "Sarah Klau",
    },
    "Queensland Firebirds": {
        "GS": "Mary Cholhok",
        "GA": "Te Paea Selby-Rickit",
        "WA": "Macy Gardner",
        "C": "Maddy Gordon",
        "WD": "Lara Dunkley",
        "GD": "Ruby Bakewell-Doran",
        "GK": "Kelly Jackson",
    },
    "Sunshine Coast Lightning": {
        "GS": "Donnell Wallam",
        "GA": "Gabby Sinclair",
        "WA": "Leesa Mi Mi",
        "C": "Liz Watson",
        "WD": "Mahalia Cassidy",
        "GD": "Karin Burger",
        "GK": "Ash Ervin",
    },
    "West Coast Fever": {
        "GS": "Romelda Aiken-George",
        "GA": "Sasha Glasgow",
        "WA": "Alice Teague-Neeld",
        "C": "Jordan Cransberg",
        "WD": "Jess Anstiss",
        "GD": "Fran Williams",
        "GK": "Kadie-Ann Dehaney",
    },
}
```

**Public API:**

```python
def get_squad(team: str, season: int = 2026) -> dict[str, str]:
    """Return {position: player_name} for a team. Currently only 2026 hardcoded."""

def get_all_squads(season: int = 2026) -> dict[str, dict[str, str]]:
    """Return all team squads for a season."""
```

### Part B: Player Movement Tracker

**Module:** `src/netball_model/data/player_movements.py`

**How it works:** For historical seasons, query the `player_stats` table to see which players appeared for which teams. For 2026, use the hardcoded `SQUADS_2026` data.

```python
def get_player_movements(season: int) -> list[dict]:
    """Detect player movements into a given season.

    Returns list of:
    {
        "player_name": str,
        "player_id": int | None,
        "from_team": str | None,  # None = new to league
        "to_team": str,
        "movement_type": "stayed" | "moved" | "new" | "retired"
    }
    """

def get_roster_continuity(team: str, season: int) -> float:
    """Fraction of starters retained from previous season. Range [0.0, 1.0]."""

def get_team_continuity_all(season: int) -> dict[str, float]:
    """Return roster continuity for all teams in a season."""
```

**Player ID mapping:** Cross-reference player names in `SQUADS_2026` against the `player_stats` table using `difflib.SequenceMatcher` with a threshold of 0.85. Flag unmatched names (ratio < 0.85) for manual review by printing to stderr.

### Key 2026 Player Movements (from research)

| Player | From | To | Notes |
|--------|------|----|-------|
| Jamie-Lee Price | GIANTS | Mavericks | Key signing |
| Lucy Austin | Thunderbirds | GIANTS | |
| Reilley Batcheldor | Lightning | Mavericks | |
| Tara Hinchliffe | Lightning | Mavericks | |
| Sasha Glasgow | Mavericks | Fever | |
| Tayla Fraser | Mavericks | Swifts | |
| Romelda Aiken-George | (retired) | Fever | Maternity replacement |
| Karin Burger | NZ (Tactix) | Lightning | International signing |
| Donnell Wallam | NZ (Mystics) | Lightning | Returning Australian |
| Whitney Souness | NZ (Pulse) | GIANTS | International |
| Jane Watson | NZ (Tactix) | GIANTS | International |
| Maddy Gordon | NZ (Pulse) | Firebirds | International |
| Kelly Jackson | NZ (Pulse) | Firebirds | International |
| Te Paea Selby-Rickit | NZ (Tactix) | Firebirds | International |
| Kate Heffernan | NZ (Southern Steel) | Thunderbirds | International |
| Elmere van der Berg | UK (Manchester Thunder) | Thunderbirds | International |
| Gina Crampton | NZ (Pulse) | Swifts | Temporary replacement |
| Gabby Sinclair | UK (Birmingham) | Lightning | International |
| Jo Harten | GIANTS | Retired | |

---

## 3. Player Glicko-2 + Named Matchup Rankings

**Purpose:** Rate individual players via positional Glicko-2. Enable named matchup queries.

### Player Glicko-2 System

**Module:** `src/netball_model/features/player_elo.py`

**Rating per player per position.** A player who plays both GA and WA has separate ratings for each position.

**Matchup outcome determination** — reuses the interaction features from `matchups.py`. Feature names in the table below are shorthand; the actual keys are prefixed by the pair (e.g., `gs_vs_gk_shooting_pressure`):

| Positional Pair | "Win" condition (attacker perspective) |
|----------------|------------------------------------------|
| GS vs GK | `gs_vs_gk_shooting_pressure > 0` |
| GA vs GD | `mean(ga_vs_gd_shooting_pressure, ga_vs_gd_feed_vs_intercept) > 0` |
| WA vs WD | `mean(wa_vs_wd_delivery_vs_disruption, wa_vs_wd_supply_line) > 0` |
| C vs C | `mean(c_vs_c_distribution_battle, c_vs_c_disruption_battle) > 0` |
| WD vs WA | `mean(wd_vs_wa_pressure_effectiveness, wd_vs_wa_aerial_battle) > 0` |

**Bidirectional rating updates:** The existing `matchups.py` computes features from the home perspective only (home GS vs away GK, etc.). For player Glicko-2, we need every player rated in every match. So `process_match()` computes matchup features in BOTH directions:

1. **Home attacking → Away defending:** Home GS vs Away GK, Home GA vs Away GD, Home WA vs Away WD
2. **Away attacking → Home defending:** Away GS vs Home GK, Away GA vs Home GD, Away WA vs Home WD
3. **Midcourt (both directions):** Home C vs Away C, Home WD vs Away WA, Away WD vs Home WA

This gives 9 pairings per match (C vs C is symmetric so counted once). Every position on both teams gets at least one rating update per match.

**Reverse pairing win conditions:** The same win condition logic applies but from the perspective of the other attacker. For "Away GS vs Home GK", compute the same `gs_vs_gk_shooting_pressure` formula but with away GS stats and home GK stats (swapped). If the feature value is > 0, the away GS wins; if < 0, the home GK wins. Same for GA vs GD and WA vs WD reverses. The `matchups.py` functions already accept arbitrary player profiles — just swap which is "home" and "away" in the function call.

**Rating update process** (per match):
1. Compute player profiles for all 14 starters using `PlayerProfiler` (the `process_match()` method internally creates a `PlayerProfiler` instance and calls the relevant `matchups.py` functions to compute interaction features)
2. For each of the 9 pairings, determine the "winner" per the conditions above
3. Apply Glicko-2 update to both players in the pair
4. Apply margin-of-victory scaling: `log(|feature_sum| + 1)` as multiplier on rating delta

**Between-season handling:**
- Glicko-2 RD increases naturally for inactive players (built into the algorithm)
- Additionally, regress all ratings 20% toward the mean (1500) at season boundaries
- Players moving teams carry their individual rating (it's theirs, not the team's)
- New players initialised at 1500 with high RD (350, the Glicko-2 default)

**Storage:** New `player_elo_ratings` table in SQLite, created via `CREATE TABLE IF NOT EXISTS` in `database.py::initialize()` (matching the existing migration pattern):

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
```

**New database methods** (added to `Database` class in `database.py`):
- `upsert_player_elo(player_id, player_name, position, pool, match_id, rating, rd, vol)` — insert or update a player rating
- `get_latest_player_elo(player_id, position, pool) -> dict | None` — get most recent rating for a player at a position
- `get_all_player_elos(pool) -> list[dict]` — get latest ratings for all players (for rankings)

**Public API:**

```python
class PlayerGlicko2:
    def __init__(self):
        self.ratings: dict[tuple[int, str], PlayerRating] = {}  # (player_id, position) -> rating

    def process_match(self, match: dict, home_starters: list[dict], away_starters: list[dict]) -> None:
        """Update player ratings based on positional matchup outcomes."""

    def get_rating(self, player_id: int, position: str) -> PlayerRating:
        """Get current rating for a player at a position."""

    def get_matchup_prediction(self, player_a_id: int, pos_a: str, player_b_id: int, pos_b: str) -> float:
        """Predict win probability for player A vs player B."""
```

### Named Matchup Rankings

**Module:** `src/netball_model/features/player_rankings.py`

```python
def get_position_rankings(position: str) -> list[dict]:
    """Return ranked list of players at a position.

    Returns: [{"rank": 1, "player_name": "...", "rating": 1650, "rd": 45, "team": "..."}, ...]
    """

def get_matchup_prediction(player_a: str, player_b: str) -> dict:
    """Predict outcome of a named matchup.

    Returns: {"player_a": "...", "player_b": "...", "a_win_prob": 0.68, "rating_diff": 60}
    """

def get_team_matchup_report(home_team: str, away_team: str, home_squad: dict, away_squad: dict) -> list[dict]:
    """Full matchup report for all 5 positional pairs between two teams."""
```

### Model Integration

New features added to `FeatureBuilder.build_row()`:
- `home_player_elo_avg` — average Glicko-2 rating across home team's 7 starters
- `away_player_elo_avg` — same for away
- `player_elo_diff` — `home_player_elo_avg - away_player_elo_avg`
- 5 per-position diffs: `gs_gk_player_elo_diff`, `ga_gd_player_elo_diff`, `wa_wd_player_elo_diff`, `c_c_player_elo_diff`, `wd_wa_player_elo_diff`

Total: 8 new features (3 aggregate + 5 positional). These supplement the existing 17 interaction-based matchup features.

---

## 4. Time-Weighted Training + Roster Continuity + Round Number

**Purpose:** Make recent data matter more, account for roster turnover, capture early-season effects.

### Time Decay

**Where:** `src/netball_model/model/train.py` — `NetballModel.train()`

Ridge regression supports `sample_weight`. Each match gets:

```python
base_weight = math.exp(-lambda_ * years_ago)
```

Where `years_ago = current_season - match_season` and `lambda_ = 0.5` (configurable).

Effect at `lambda_ = 0.5`:

| Years ago | Weight |
|-----------|--------|
| 0 (current) | 1.00 |
| 1 | 0.61 |
| 2 | 0.37 |
| 3 | 0.22 |
| 5 | 0.08 |
| 8 | 0.02 |

### Roster Continuity Adjustment

The base time-decay weight is further modified:

```python
weight = base_weight * (0.5 + 0.5 * roster_continuity)
```

Where `roster_continuity` is from `player_movements.get_roster_continuity()`. Effect:

| Starters retained | Continuity | Multiplier |
|-------------------|------------|------------|
| 7/7 | 1.00 | 1.00 |
| 5/7 | 0.71 | 0.86 |
| 3/7 | 0.43 | 0.71 |
| 1/7 | 0.14 | 0.57 |
| 0/7 | 0.00 | 0.50 |

This means even a completely new roster retains 50% of the base weight (the venue, league context, and coaching still matter).

### Glicko-2 Season Reset

**Where:** `src/netball_model/features/builder.py` — `FeatureBuilder._ensure_elo_up_to()`

The `GlickoSystem` class itself has no season awareness (it only processes individual match results). Season-boundary logic lives in `FeatureBuilder._ensure_elo_up_to()`, which iterates through matches chronologically and has access to `match["season"]`. When it detects a season change (`match["season"] != prev_match["season"]`):
- Call a new `GlickoSystem.regress_ratings(factor=0.2, mean=1500)` method that adjusts all team ratings: `new_rating = rating * 0.8 + 1500 * 0.2`
- Call a new `GlickoSystem.increase_rd(amount=30)` method that increases RD by a fixed amount to reflect off-season uncertainty
- Also call `PlayerGlicko2.regress_ratings(factor=0.2, mean=1500)` to apply the same season reset to player ratings. The `FeatureBuilder` holds references to both the team `GlickoSystem` and the `PlayerGlicko2` instance, and drives season resets for both from the same detection point.

### Round Number Feature

**Where:** `src/netball_model/features/contextual.py`

Two new features:
- `round_number` — raw round number from `match["round_num"]` (this field exists in match dicts from `db.get_matches()` and is populated from the `round_num` column in the `matches` table)
- `season_progress` — `round_num / max_rounds` normalised to [0, 1]

`max_rounds` is determined per season from the data (typically 14 regular season rounds). For the current in-progress season where max is unknown, default to 14.

### Changes to FeatureBuilder

`build_row()` gains:
- `round_number` and `season_progress` from contextual
- Sample weight metadata: a `_sample_weight` key in the row dict, computed as `base_weight * (0.5 + 0.5 * roster_continuity)`. This key must be added to `NON_FEATURE_COLUMNS` so it is excluded during training feature selection. `NetballModel.train()` extracts `_sample_weight` from the DataFrame before fitting and passes it as `sample_weight` to both Ridge regression `.fit()` calls (margin and total models).

---

## 5. Line/Total Value Detection

**Purpose:** Extend value detection beyond head-to-head to assess handicap lines and total goals markets.

**Where:** `src/netball_model/value/detector.py`

### Current State

`ValueDetector.evaluate()` takes `home_odds, away_odds` and returns an edge assessment for the H2H market only.

### Extended Design

```python
class ValueDetector:
    def evaluate(self, prediction: dict, odds: dict, threshold: float = 0.05) -> list[dict]:
        """Evaluate all available markets for value.

        Args:
            prediction: {"margin": float, "total_goals": float, "win_prob": float,
                         "residual_std": float, "total_residual_std": float}
            odds: {"home_odds": float, "away_odds": float,
                   "handicap_line": float, "handicap_home_odds": float, "handicap_away_odds": float,
                   "total_line": float, "over_odds": float, "under_odds": float}
            threshold: Minimum edge to flag as value

        Returns: List of value bets found, each:
            {"market": "h2h"|"handicap"|"total",
             "side": "home"|"away"|"over"|"under",
             "model_prob": float, "implied_prob": float, "edge": float,
             "odds": float, "line": float | None}
        """
```

### Market Calculations

**Head-to-head** (unchanged):
- `model_prob = prediction["win_prob"]` for home
- `implied_prob = 1 / odds["home_odds"]`
- Both sides checked

**Handicap:**
- Model predicts margin M with residual std σ
- Book offers home at handicap line L (e.g., L = -5.5 means home must win by more than 5.5)
- `P(home covers) = 1 - Phi((L - M) / σ)` where L is `odds["handicap_line"]` directly (the sign is already encoded in the value: negative = home favored)
- Example: M=8.5, L=-5.5, σ=12 → `(L - M)/σ = (-5.5 - 8.5)/12 = -1.17` → `P = 1 - Phi(-1.17) = 0.879`
- Compare to implied probability from `odds["handicap_home_odds"]`

**Totals:**
- Model predicts total T with total residual std σ_t
- Book offers over/under at line L_t (e.g., 125.5)
- `P(over) = 1 - Phi((L_t - T) / σ_t)`
- Example: T=130, L_t=125.5, σ_t=15 → P = 1 - Phi(-0.3) = 0.618
- Compare to implied probability from `over_odds`

### Calibration Model Extension

`CalibrationModel` currently stores `residual_std` from the margin model. Extend to also store `total_residual_std` from the total goals model:

```python
class CalibrationModel:
    def __init__(self):
        self.residual_std: float = 0.0
        self.total_residual_std: float = 0.0
```

The `CalibrationModel.fit()` method currently takes `residuals` (margin). Extend it to also accept an optional `total_residuals` parameter (default `None` for backward compatibility). When provided, compute `total_residual_std = np.std(total_residuals)`. In `NetballModel.train()`, compute total residuals from the total model and pass both to `CalibrationModel.fit()`.

### Streamlit Integration

The value summary table in `app.py` expands from 1 row per match (H2H only) to up to 3 rows per match (H2H, Handicap, Total). Each row shows: market type, recommended side, model probability, implied probability, edge, and odds.

---

## Testing Strategy

Each subsystem has independent tests:

| Subsystem | Test File | Key Tests |
|-----------|-----------|-----------|
| Screenshot OCR | `tests/data/test_bet365_screenshot.py` | Parse mocked OCR output; structure grouping; regex parsing; full pipeline with fixture |
| Squads | `tests/data/test_squads.py` | Squad retrieval; all teams present; 7 positions per team |
| Player Movements | `tests/data/test_player_movements.py` | Movement detection from DB data; roster continuity calculation |
| Player Glicko-2 | `tests/features/test_player_elo.py` | Rating initialization; matchup outcome determination; rating updates; between-season regression |
| Player Rankings | `tests/features/test_player_rankings.py` | Position rankings; named matchup predictions |
| Time Weighting | `tests/model/test_time_weighting.py` | Weight calculation; decay curve; roster continuity adjustment |
| Round Number | `tests/features/test_contextual.py` (extend) | Round number and season_progress feature extraction |
| Value Detection | `tests/value/test_detector.py` (rewrite) | Handicap value calc; total value calc; edge thresholding. Note: existing 3 tests use the old `evaluate()` signature with positional args; must be rewritten to match the new dict-based signature. |

---

## Agent Parallelisation Plan

For implementation, these subsystems can be developed in parallel by a team of agents:

| Agent | Subsystem | Blocked By |
|-------|-----------|------------|
| Agent 1 | Screenshot OCR (#1) | Nothing |
| Agent 2 | Squads + Player Movements (#2) | Nothing |
| Agent 3 | Player Glicko-2 + Rankings (#3) | Agent 2 (needs squad data structure defined) |
| Agent 4 | Time Weighting + Round Number (#4) | Agent 2 (needs roster continuity API) |
| Agent 5 | Line/Total Value Detection (#5) | Nothing |

Agents 1, 2, and 5 can start immediately. Agents 3 and 4 start once Agent 2 has defined the public API (not necessarily completed — just the interface contract).

**Integration point:** After all agents complete, a final integration pass:

1. **`services.py::train_model()`** — Currently creates `FeatureBuilder(matches)` without player_stats. Must be updated to:
   - Load player stats by iterating over matches and calling `db.get_player_stats(match_id)` for each (same pattern as `backtest_season()` at lines 80-85 of `services.py`), then pass the resulting dict to `FeatureBuilder`
   - Initialize and replay `PlayerGlicko2` through the match list
   - Pass roster continuity data for sample weight computation
   - Extract `_sample_weight` from the feature DataFrame and pass to `model.train()`

2. **`FeatureBuilder.build_row()`** — Wire in the 8 new player elo features and 2 round features

3. **Run full test suite** — All 92+ existing tests must still pass, plus new tests from each subsystem
