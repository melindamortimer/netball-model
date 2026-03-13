# Bet365 Scraper Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-13-bet365-scraper-design.md`

---

## Task 1: Extract shared team name module

Move `TEAM_NAME_MAP` and `normalise_team()` from `betsapi.py` to a shared module.

**Files:**
- Create: `src/netball_model/data/team_names.py`
- Modify: `src/netball_model/data/betsapi.py` (import from shared module)

---

## Task 2: Add Playwright dependencies

**Files:**
- Modify: `pyproject.toml`

Run: `poetry add playwright playwright-stealth`
Then: `poetry run playwright install chromium`

---

## Task 3: Extend database schema

Add handicap/totals columns to `odds_history` and a `get_odds_for_match()` method.

**Files:**
- Modify: `src/netball_model/data/database.py`

---

## Task 4: Build the Playwright scraper

**Files:**
- Create: `src/netball_model/data/bet365.py`

---

## Task 5: Add CLI commands

Add `scrape-odds` and `update` commands.

**Files:**
- Modify: `src/netball_model/cli.py`
- Modify: `src/netball_model/services.py`

---

## Task 6: Integrate odds into predict command and Streamlit app

**Files:**
- Modify: `src/netball_model/cli.py` (predict command)
- Modify: `src/netball_model/display.py` (rename Betfair column)
- Modify: `app.py` (auto-load from DB)

---

## Task 7: Tests

**Files:**
- Create: `tests/data/test_team_names.py`
- Create: `tests/data/test_bet365.py`
- Modify: existing tests if needed

Run: `poetry run pytest -v` — all tests pass
