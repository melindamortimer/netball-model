# Bet365 Odds Scraper + Pipeline Automation

**Goal:** Automate pre-game odds collection from bet365 using Playwright, store in DB, integrate with the predict command and Streamlit app for value detection, and automate the weekly update pipeline (ingest results + retrain).

## Architecture

Four changes:

1. **`src/netball_model/data/bet365.py`** — Playwright scraper module
2. **Database schema extension** — Add handicap/totals columns to `odds_history`
3. **CLI commands** — `netball scrape-odds` and `netball update`
4. **Predict integration** — `predict` command and Streamlit app auto-load stored odds

## 1. Scraper Module (`bet365.py`)

### Navigation Flow

1. Launch Playwright Chromium (non-headless so user can see/intervene)
2. Navigate to bet365.com.au → Netball → Super Netball
3. SSN page lists upcoming matches — extract match list
4. For each match: click into match page, extract three markets, navigate back
5. Return structured list of odds dicts

### Anti-Detection

- `playwright-stealth` plugin for fingerprint masking
- Random delays 2-5s between navigations
- Realistic viewport (1280x800) and user-agent
- Non-headless by default

### Team Name Normalisation

Reuse the existing `TEAM_NAME_MAP` from `betsapi.py` — bet365 Australia uses similar display names. Add any missing mappings (e.g. "Melbourne Mavericks" already maps correctly).

Move `TEAM_NAME_MAP` and `normalise_team()` from `betsapi.py` to a shared `src/netball_model/data/team_names.py` so both modules use the same mapping.

### Output Format

```python
{
    "home_team": "Melbourne Mavericks",  # normalised
    "away_team": "GIANTS Netball",       # normalised
    "match_date": "2026-03-14",
    "home_odds": 1.38,
    "away_odds": 3.00,
    "handicap_home_odds": 1.85,
    "handicap_line": -4.5,
    "handicap_away_odds": 1.95,
    "total_line": 125.5,
    "over_odds": 1.87,
    "under_odds": 1.87,
}
```

### Error Handling

- If a match page fails to load, log warning, take screenshot to `data/screenshots/`, continue to next
- If a market section is missing from a match page, store what's available (H2H only is fine)
- Configurable timeout per page (default 15s)

## 2. Database Schema Extension

Add columns to `odds_history` via `ALTER TABLE` migration in `database.py.initialize()`:

```sql
ALTER TABLE odds_history ADD COLUMN handicap_home_odds REAL;
ALTER TABLE odds_history ADD COLUMN handicap_line REAL;
ALTER TABLE odds_history ADD COLUMN handicap_away_odds REAL;
ALTER TABLE odds_history ADD COLUMN total_line REAL;
ALTER TABLE odds_history ADD COLUMN over_odds REAL;
ALTER TABLE odds_history ADD COLUMN under_odds REAL;
```

Existing 321 BetsAPI rows get NULLs for new columns. New bet365 rows populate all fields.

Add a `get_odds_for_match(match_id)` method that returns the most recent odds record for a specific match (used by predict command and Streamlit app).

## 3. CLI Commands

### `netball scrape-odds`

```
netball scrape-odds [--db PATH] [--headless]
```

1. Open bet365 SSN page via Playwright
2. Scrape all visible upcoming matches
3. Normalise team names
4. Match to DB matches by (home_team, away_team, date)
5. Store in `odds_history` with `source="bet365"`
6. Print summary: N matches scraped, N matched to DB, N unmatched

Requires matches already ingested (`netball ingest` first).

### `netball update`

```
netball update --season YEAR [--db PATH] [--output PATH]
```

Full weekly pipeline in one command:
1. `ingest_season(season)` — pulls latest results + player stats from Champion Data
2. `train_model(db, output)` — rebuilds features with new data and retrains
3. Print summary: matches ingested, training MAE

## 4. Predict Integration

### CLI `predict` command

Currently calls `detector.evaluate()` without odds. Change to:
1. After building prediction for each upcoming match, query `db.get_odds_for_match(match_id)`
2. If odds exist, pass `home_odds` and `away_odds` to `detector.evaluate()`
3. Display shows odds, implied prob, edge, and value flag (currently shows "-" for all)

### Streamlit app

After prediction, check DB for stored odds:
1. Call `db.get_odds_for_match(match_id)` for the selected match
2. If odds exist, auto-fill the bet365 row in the odds table
3. User can still manually override or add other bookmakers

## 5. Dependencies

Add to `pyproject.toml`:
- `playwright` — browser automation
- `playwright-stealth` — anti-detection

Post-install: `playwright install chromium`

## Testing Strategy

- Unit tests for team name normalisation (shared module)
- Unit tests for `get_odds_for_match` DB method
- Integration test for scraper output format (mock Playwright page)
- Existing 92 tests must continue passing
