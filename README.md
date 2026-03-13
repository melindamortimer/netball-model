# Netball Model

Super Netball match prediction model using Glicko-2 ratings (team and individual player), player matchup features, time-weighted training, and Ridge regression. Includes multi-market value detection for h2h, handicap, and total lines.

## Features

- **Team Glicko-2 ratings** with season boundary resets (regression toward mean + uncertainty increase)
- **Individual player Glicko-2 ratings** based on 9 positional matchup pairings per match
- **17 player interaction matchup features** (GS vs GK, GA vs GD, WA vs WD, C vs C, WD vs WA)
- **Contextual features**: rest days, travel distance, recent form, head-to-head, round/season progress
- **Time-weighted training**: exponential decay by season age, adjusted by roster continuity
- **Multi-market value detection**: h2h, handicap (line), and totals (over/under) using normal CDF models
- **Bet365 screenshot OCR**: parse odds from screenshots using EasyOCR
- **2026 squad data** with player movement tracking and roster continuity

## Setup

```bash
poetry install
```

## Usage

### Streamlit App (Value Finder)

```bash
poetry run streamlit run app.py
```

Opens a dashboard where you can select upcoming matches or manually enter teams, view model predictions, upload bet365 screenshots, and compare bookmaker odds across h2h, handicap, and total markets to find value bets.

### CLI

```bash
# Train the model
poetry run netball train --db data/netball.db

# Predict upcoming matches
poetry run netball predict --db data/netball.db
```

### Tests

```bash
poetry run pytest -v  # 153 tests
```

## Architecture

```
src/netball_model/
  data/
    database.py          # SQLite access layer
    champion_data.py     # Champion Data API ingestion
    squads.py            # 2026 SSN squad data
    player_movements.py  # Roster continuity tracking
    bet365.py            # Bet365 web scraper
    bet365_screenshot.py # Bet365 screenshot OCR (EasyOCR)
  features/
    builder.py           # Central feature orchestrator
    elo.py               # Team Glicko-2 ratings
    player_elo.py        # Individual player Glicko-2 ratings
    player_rankings.py   # Player ranking lookups
    contextual.py        # Rest, travel, form, H2H, round
    matchups.py          # Player interaction matchup features
    player_profile.py    # Rolling player profiles
  model/
    train.py             # Ridge regression (margin + total)
    calibration.py       # Residual std calibration
  value/
    detector.py          # Multi-market value detection
app.py                   # Streamlit Value Finder dashboard
```

## Data

- 539 matches (2017-2025), 254 players, 11k+ player stat rows
- SQLite database at `data/netball.db`
- 2026 squad data: 8 teams, 56 players
