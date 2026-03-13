# Bet365 Scraper Navigation & Extraction Fix

**Goal:** Rewrite the navigation and extraction logic in `bet365.py` to match the actual bet365 UI, and extend the output to include 1st Half and 1st Quarter odds.

**Scope:** This spec covers only the scraper module (`bet365.py`). Downstream changes (DB schema for `h1_`/`q1_` columns, CLI `scrape-odds` command updates, home/away swap logic for new fields) are out of scope and will be addressed in a follow-up.

## Problem

The current scraper's navigation doesn't match the real bet365 site structure:
- The SSN page uses expandable dropdown sections, not separate competition links
- Match discovery scans every `div, span` on the page looking for team names (slow, fragile)
- Odds extraction tries to find market sections by header text and walk up ancestors (unreliable)

## Corrected Navigation Flow

1. **Homepage**: Load bet365.com.au, dismiss popups (existing logic works)
2. **Netball**: Click "Netball" in the left sidebar (existing logic works)
3. **Super Netball dropdown**: On the Netball page (Matches tab), locate "Super Netball" via `page.get_by_text("Super Netball", exact=True)` and click to expand. Wait for child content to appear.
4. **Matches sub-section**: Within the expanded Super Netball container, locate "Matches" via `page.get_by_text("Matches", exact=True)` (scoped to the SSN section if possible) and click to reveal the match list.
5. **Collect match links**: Find all text elements matching the "Team A v Team B" pattern (regex `\w.+ v \w.+`) inside the expanded section. Each is a clickable link. Store their text labels for later re-location.
6. **Per match**: Click into match page, extract odds, go back. After `page.go_back()`, re-expand the Super Netball dropdown and Matches sub-section if collapsed (the dropdown may reset after navigation). Re-locate the next match link by its text label rather than relying on stale locators.

## Extraction Strategy (per match page)

Each match page shows a header ("Team A v Team B") and three table sections: "Match Lines", "1st Half", "1st Quarter". Each table has rows for "To Win", "Total", and "Line".

**Approach: Section-based text extraction**

For each of the three table sections:
1. Locate the section header text ("Match Lines", "1st Half", "1st Quarter")
2. Walk up to the ancestor container that wraps the full table (try ancestor div depths 2-5, pick the one containing odds-like numbers)
3. Extract inner text of the container
4. Parse line-by-line with regex. Expected inner text looks like:
   ```
   Match Lines
   Melbourne Mavericks    GIANTS Netball
   To Win    1.30    3.50
   Total    O 125.5 1.87    U 125.5 1.87
   Line    -5.5 1.85    +5.5 1.95
   ```
5. Row parsing (line-by-line, keyed on row label):
   - **To Win row**: Find a line containing "To Win", extract the two decimal odds `(\d+\.\d{2})` from it
   - **Total row**: Find a line containing "Total", extract `O (\d+\.?\d*) (\d+\.\d{2})` and `U (\d+\.?\d*) (\d+\.\d{2})`
   - **Line row**: Find a line containing "Line" (but not "Match Lines"), extract `([+-]\d+\.?\d*) (\d+\.\d{2})` twice

   By matching on row labels first, we avoid ambiguity between To Win odds and Total/Line odds.

**Team names**: Extracted from the page header "Team A v Team B", split on " v ", normalised via `normalise_team()`.

**Match date**: Extracted from page text using existing date patterns (DD/MM/YYYY, DD Month YYYY, etc.).

## Output Schema

Extends the existing dict with `h1_` and `q1_` prefixed fields:

```python
{
    "home_team": "Melbourne Mavericks",
    "away_team": "GIANTS Netball",
    "match_date": "2026-03-15",
    # Match Lines (full match)
    "home_odds": 1.30,
    "away_odds": 3.50,
    "total_line": 125.5,
    "over_odds": 1.87,
    "under_odds": 1.87,
    "handicap_line": -5.5,
    "handicap_home_odds": 1.85,
    "handicap_away_odds": 1.95,
    # 1st Half
    "h1_home_odds": 1.40,
    "h1_away_odds": 2.75,
    "h1_total_line": 62.5,
    "h1_over_odds": 1.83,
    "h1_under_odds": 1.83,
    "h1_handicap_line": -2.5,
    "h1_handicap_home_odds": 1.80,
    "h1_handicap_away_odds": 1.90,
    # 1st Quarter
    "q1_home_odds": 1.50,
    "q1_away_odds": 2.50,
    "q1_total_line": 30.5,
    "q1_over_odds": 1.80,
    "q1_under_odds": 1.90,
    "q1_handicap_line": -1.5,
    "q1_handicap_home_odds": 1.83,
    "q1_handicap_away_odds": 1.83,
}
```

All `h1_` and `q1_` fields default to `None` if extraction fails for that section.

## Code Changes

**File**: `src/netball_model/data/bet365.py`

### Methods to rewrite:
- **`_navigate_to_ssn`**: Click Netball sidebar, click "Super Netball" dropdown, click "Matches" sub-section
- **`_collect_match_links`**: Inside expanded Matches section, find "Team v Team" links by text pattern. Returns a list of match label strings (not locators). No more brute-force DOM scanning.
- **`scrape_ssn_odds`**: The main loop changes. Instead of iterating over pre-collected locators, it iterates over the list of match label strings. For each match: call `_ensure_ssn_expanded` to re-expand dropdowns if needed, locate the match link by its text label, click, extract, go back.
- **`_scrape_single_match`**: Simplified — receives the match label string (not a locator dict). Locates the match by text, clicks, extracts, goes back.
- **`_extract_match_odds`**: Find three sections by header, delegate to `_parse_table_section`
- **`_extract_team_names`**: Just parse "Team A v Team B" from the match page header

### New methods:
- **`_parse_table_section(section_locator) -> dict`**: Extract inner text from a section, parse To Win/Total/Line rows. Returns dict with 8 unprefixed keys: `home_odds`, `away_odds`, `total_line`, `over_odds`, `under_odds`, `handicap_line`, `handicap_home_odds`, `handicap_away_odds`.
- **`_ensure_ssn_expanded(page)`**: Check if the Super Netball dropdown and Matches sub-section are expanded. If not, click to expand them. Called before each match link click to handle re-collapsed state after `page.go_back()`.

### Prefix mapping in `_extract_match_odds`:
Calls `_parse_table_section` three times. For Match Lines, uses keys as-is. For 1st Half and 1st Quarter, prepends `h1_` or `q1_` to each key before merging into the result dict.

### Methods to remove:
- `_extract_h2h` — replaced by `_parse_table_section`
- `_extract_handicap` — replaced by `_parse_table_section`
- `_extract_totals` — replaced by `_parse_table_section`
- `_find_market_section` — section finding simplified inline
- `_extract_odds_from_section` — replaced by text parsing
- `_extract_line_from_section` — replaced by text parsing

### Methods kept as-is:
- `__init__` (public API signature unchanged)
- `_dismiss_popups`
- `_random_delay` (module-level)
- `_parse_odds`, `_parse_line` (module-level helpers)
- `_extract_match_date` (date parsing still useful)

## Error Handling

- If "Super Netball" dropdown not found, raise `RuntimeError` (same as now)
- If "Matches" sub-section not found, raise `RuntimeError`
- If a section (e.g. "1st Quarter") is missing from a match page, set those fields to `None`
- If a row (e.g. "Total") is missing within a section, set those fields to `None`
- Per-match failures logged and skipped (same as now)
