# Batch Screenshot Prediction Flow

## Summary

Replace the "Manual entry" mode in `app.py` with a batch screenshot workflow. The user uploads bet365 screenshots for all matches in a round, reviews/edits the parsed odds in a table, then gets predictions and value analysis for every match at once.

## Implementation Note

Build using a **team of agents** — parallelize independent work streams (OCR parsing improvements, app UI rewrite, data storage layer, prediction output formatting) across multiple agents working concurrently.

## Flow

Two modes via radio button:
- **"Upcoming (from DB)"** — unchanged
- **"Screenshot round"** — replaces "Manual entry"

The screenshot round mode is a 3-step linear wizard tracked via `st.session_state["wizard_step"]`:

```
Upload screenshots → Parse all → Review/edit table → Confirm & Predict → Results table
```

No back navigation. The flow is linear: collect everything first, parse in batch, review everything, predict everything.

## Step 1: Upload

**UI elements:**
- `st.file_uploader` with `accept_multiple_files=True` — user drops all screenshots at once
- Paste component (existing `components/paste_image/`) below — each paste appends to `st.session_state["pasted_images"]`. A count label shows "N screenshots collected" with small thumbnails
- No OCR runs during this step — just collecting images
- **"Parse all"** button triggers OCR on all images with a progress bar ("Parsing 1/4..."). Results populate `st.session_state["parsed_matches"]` and auto-advance to the review step

**Paste deduplication:** The paste component returns the same base64 value on every Streamlit rerun. To avoid duplicate appends, track a hash (or length) of the last appended image in `st.session_state["last_paste_hash"]`. Only append when the current paste value differs from the last stored hash.

**Screenshot format:** One match per screenshot. Each shows the bet365 "Match Lines" section with To Win, Total, and Line rows. Team names in the header ("X v Y").

**OCR pipeline:** `parse_screenshot()` in `bet365_screenshot.py` takes a file path, not bytes. Use the existing `_parse_screenshot_image()` wrapper in `app.py` which writes bytes to a temp file, calls `parse_screenshot()`, and cleans up. Each image is processed through this wrapper. If OCR raises an exception, catch it and mark that image as failed (don't abort the batch). Failed images show a warning with the error message.

**Team name normalization:** OCR-extracted team names are passed through `normalise_team()` from `team_names.py`. If `normalise_team()` returns `None` (unrecognized name), use the first team from the SSN team list as a placeholder and flag the row in the review table so the user corrects it.

## Step 2: Review & Edit

**UI element:** `st.data_editor` with one row per parsed match.

**Columns:**

| Column | Type | Default (new rows) | Notes |
|--------|------|-------------------|-------|
| Date | date input | today | Editable — screenshots may span multiple match days |
| Home Team | dropdown (8 SSN teams) | first team | Constrained to valid team names |
| Away Team | dropdown (8 SSN teams) | second team | Constrained to valid team names |
| H2H Home | float | None | Home win odds |
| H2H Away | float | None | Away win odds |
| Line | float | None | Handicap line (home perspective, e.g. +1.5) |
| HC Home | float | None | Handicap home odds |
| HC Away | float | None | Handicap away odds |
| Total | float | None | Total goals line (e.g. 125.5) |
| Over | float | None | Over odds |
| Under | float | None | Under odds |

**Column mapping from `parse_screenshot()` output:**
- `home_team` → Home Team (after `normalise_team()`)
- `away_team` → Away Team (after `normalise_team()`)
- `home_odds` → H2H Home
- `away_odds` → H2H Away
- `handicap_line` → Line
- `handicap_home_odds` → HC Home
- `handicap_away_odds` → HC Away
- `total_line` → Total
- `over_odds` → Over
- `under_odds` → Under

**Editing capabilities:**
- Every cell is editable — fix OCR misreads directly
- Rows can be deleted (garbage parses)
- Rows can be added manually (match not screenshotted) — new rows get today's date and placeholder team names
- Team name columns use dropdown select constrained to the 8 SSN teams

**Buttons:**
- **"Confirm & Predict"** — saves data and runs predictions

## Step 3: Predict & Results

On "Confirm & Predict":

1. **Save odds** to SQLite + JSON (see Data Storage below)
2. **Run predictions** for each match independently with a progress bar ("Predicting 1/4...")
   - **Each match is predicted independently against the same base historical match list.** Do NOT append all synthetic matches at once — each prediction starts from the same base list with only that one match appended. This prevents synthetic matches (with no scores) from affecting each other's Elo/feature calculations.
   - Each synthetic match dict: `match_id = "manual_{home_team}_{away_team}"`, `season = date.year`, `round_num = 0` (acceptable — `round_number` and `season_progress` features will be 0, which is a minor signal loss but doesn't break predictions)
   - `predict_match()` called with 2026 squad starters (existing logic injects synthetic starters for matches not in player_stats)
3. **Display results** in a combined table:

| Match | Date | Margin | Home Win% | Total | H2H Edge | HC Edge | Total Edge | Value? |
|-------|------|--------|-----------|-------|----------|---------|------------|--------|
| Fever v Lightning | Mar 15 | +3.2 | 58% | 121.4 | +5.2% | -1.1% | +3.8% | H2H, Total |

- **Edge columns** show the best positive edge for that market (home or away side). If neither side has positive edge, show the less-negative edge (still useful context). Dash if no odds were entered for that market.
- **Value column** lists markets with edge above threshold. Empty if none.
- Rows with any value are highlighted green.
- Below the table: individual callout boxes for each value bet with full detail (team, edge %, odds, model fair odds).

Min edge threshold slider stays in the sidebar.

## Data Storage

### SQLite

Each match upserted into `odds_history` via `upsert_odds_extended_batch()`. The SQL template requires all named parameters — pass `None` for unused fields:

```python
{
    "match_id": "manual_{home_team}_{away_team}",
    "source": "bet365_screenshot",
    "home_back_odds": h2h_home,      # from review table
    "away_back_odds": h2h_away,
    "home_lay_odds": None,            # not available from screenshots
    "away_lay_odds": None,
    "home_volume": None,
    "away_volume": None,
    "timestamp": datetime.now().isoformat(),
    "handicap_line": line,
    "handicap_home_odds": hc_home,
    "handicap_away_odds": hc_away,
    "total_line": total,
    "over_odds": over,
    "under_odds": under,
}
```

### JSON export

Written to `data/odds/YYYY-MM-DD.json` (filename = session date, i.e. when the user runs the flow).

```json
{
  "session_date": "2026-03-14",
  "source": "bet365_screenshot",
  "matches": [
    {
      "date": "2026-03-15",
      "home_team": "West Coast Fever",
      "away_team": "Sunshine Coast Lightning",
      "home_odds": 2.05,
      "away_odds": 1.75,
      "handicap_line": 1.5,
      "handicap_home_odds": 1.80,
      "handicap_away_odds": 2.00,
      "total_line": 125.5,
      "over_odds": 1.87,
      "under_odds": 1.87
    }
  ]
}
```

Each match has its own `date` field (editable in review step) since screenshots may span multiple match days. If the file already exists for that session date, the matches array is appended to (not overwritten). **Deduplication:** before appending, remove any existing entries with the same `home_team + away_team + date` combination to avoid duplicates from re-running the flow.

A `st.success("Saved odds to DB + data/odds/2026-03-14.json")` confirms the save.

## Files Changed

- **`app.py`** — Major rewrite of the "Manual entry" mode into the 3-step wizard. Remove old single-match manual entry. Add wizard state management, batch OCR (using `_parse_screenshot_image()` wrapper), `st.data_editor` review table, batch prediction (each match predicted independently), combined results table with value analysis.
- **`components/paste_image/index.html`** — No changes (already built).
- **`src/netball_model/data/bet365_screenshot.py`** — Wrap the body of `parse_screenshot()` in a try/except so it returns `None` (or a partial dict) instead of raising on OCR failures. The caller should handle `None` gracefully.
- **`src/netball_model/model/train.py`** — No changes (missing feature fill already handled).

## No New Files

All changes fit within existing files. The `data/odds/` directory is created on first save if it doesn't exist.
