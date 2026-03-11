# Netball Value Finder — Design Spec

## Purpose

A local Streamlit dashboard that combines the existing Super Netball prediction model with manually entered bookmaker odds to identify value bets. The user sees model predictions alongside market odds and can instantly spot where the model disagrees with the bookmakers.

## Architecture

Single Streamlit app importing existing modules. No new backend or API layer.

```
app.py (Streamlit)
  ├── loads trained model (model.pkl) via @st.cache_resource
  ├── reads matches + player stats from SQLite DB
  ├── builds features via FeatureBuilder (cached)
  ├── predicts via NetballModel
  ├── accepts manual odds input via UI
  └── computes value via ValueDetector
```

### Dependencies

- `streamlit` (new dependency, added to pyproject.toml)
- All existing: `netball_model.model.train`, `netball_model.features.builder`, `netball_model.value.detector`, `netball_model.data.database`

## Page Layout

### Section 1: Match Selection

Two modes:

**DB matches**: Dropdown of upcoming matches from the database (where `home_score IS NULL`). Pre-populated with team names and date.

**Manual entry**: Two dropdowns for home/away team (populated from all teams in the DB), plus a date picker. For ad-hoc predictions when the fixture isn't yet in the database.

A "Predict" button triggers feature building and model prediction.

### Section 2: Model Prediction

Displays after prediction runs:

| Field | Description |
|-------|-------------|
| Predicted margin | Home - away predicted goal difference |
| Home win probability | From calibration model (norm.cdf) |
| Away win probability | 1 - home win prob |
| Fair home odds | 1 / home_win_prob |
| Fair away odds | 1 / away_win_prob |

### Section 3: Bookmaker Odds & Value Table

A dynamic table where the user enters odds from bookmakers:

| Bookmaker | Home Odds | Away Odds | Home Edge | Away Edge | Value? |
|-----------|-----------|-----------|-----------|-----------|--------|
| bet365    | 1.38      | 3.00      | ...       | ...       | ...    |
| Sportsbet | 1.40      | 2.90      | ...       | ...       | ...    |

- Rows added/removed dynamically (default: bet365, Sportsbet, TAB, Ladbrokes)
- Edge = model_prob - implied_prob (where implied_prob = 1/odds)
- "Value?" column: green checkmark when edge >= threshold
- Threshold configurable via sidebar slider (default 5%)
- Best value bet highlighted with a callout: "Back [Team] at [Book] — X.X% edge"

## Changes to Existing Code

### `src/netball_model/value/detector.py`

Rename parameters from Betfair-specific to generic:
- `betfair_home_back` → `home_odds`
- `betfair_away_back` → `away_odds`

The math is identical (decimal odds → implied probability → edge). Only the parameter names change. Update the one call site in `cli.py` and the tests accordingly.

### No other existing code changes

The app imports and uses existing modules (`FeatureBuilder`, `NetballModel`, `Database`, `ValueDetector`) through their public interfaces.

## New Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app (~150-200 lines) |
| `pyproject.toml` | Add `streamlit` dependency |

## Caching Strategy

- **Model**: `@st.cache_resource` — loaded once on app start
- **Database + matches**: `@st.cache_resource` — DB connection and match list loaded once
- **Feature matrix**: `@st.cache_data` — Glicko-2 replay + feature building cached, busted when DB changes
- **Predictions**: Computed on demand per match selection (fast, no caching needed)

## Team Name Handling

Uses canonical Champion Data team names from the database. The app populates team dropdowns from `SELECT DISTINCT home_team FROM matches`. Manual entry uses the same canonical names.

Note: "Melbourne Mavericks" is the 2026 rebrand of "Melbourne Vixens". The DB will contain whichever name Champion Data uses for the current season.

## Out of Scope

- Auto-fetching odds from APIs (OddsPapi has netball but no active fixtures yet)
- Bet tracking or bankroll management
- Historical value analysis or backtesting through the UI
- User accounts or remote deployment
- Kelly criterion or staking suggestions

## Running

```bash
poetry add streamlit
streamlit run app.py
```

Opens at `http://localhost:8501`.
