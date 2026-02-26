# Netball Betting Model — Design Document

**Date:** 2026-02-26
**Status:** Approved

## Goal

Build a Python betting model for netball that:
1. Predicts match scorelines and margins
2. Derives win probabilities from those predictions
3. Compares model probabilities against bookmaker/exchange odds to find value bets

Target competitions: Suncorp Super Netball (SSN) and international matches involving Australia.

## Modeling Approach

Elo/Glicko-2 rating system + Ridge regression. Start simple, iterate.

- Maintain Glicko-2 ratings per team, updated after each match
- Feed ratings + contextual features into Ridge regression to predict match margin
- Derive win probabilities from a calibrated normal distribution of residuals
- Compare against Betfair exchange odds to flag value bets

## Data Sources

### 1. Champion Data (Match & Player Stats)

- **Source:** Reverse-engineer the JSON API at `mc.championdata.com` (same endpoints used by the `superNetballR` R package)
- **Data:** Match scores by quarter, venue, date, round. Per-player stats: goals, goal attempts, assists, gains, intercepts, deflections, centre pass receives, feeds, turnovers, penalties, rebounds.
- **Coverage:** SSN 2017-present (~500+ matches)

### 2. Betfair Historical Odds

- **Source:** `historicdata.betfair.com`, filter by sport "Netball". TAR/JSON format.
- **Data:** Exchange odds (back/lay), volume, pre-match closing prices.
- **Coverage:** 2016-present for SSN and international netball.

### 3. Betfair Live API (Current Odds)

- **Source:** Betfair Exchange API (requires existing account).
- **Data:** Real-time exchange odds for upcoming matches.
- **Purpose:** Compare model predictions against live odds for value detection.

### 4. TheSportsDB (International Results Fallback)

- **Source:** `thesportsdb.com` free API.
- **Data:** Basic results, fixtures, team info for international netball (World Cup, Commonwealth Games, Constellation Cup, Quad Series).
- **Purpose:** Supplement Champion Data for international matches not covered.

### Storage

SQLite database (`data/netball.db`, gitignored) with tables:
- `matches` — match results, scores, venue, date
- `player_stats` — per-player per-match statistics
- `team_stats` — aggregated team stats per match
- `odds_history` — Betfair historical odds per match
- `elo_ratings` — Glicko-2 rating snapshots per team per match

## Feature Engineering

### Glicko-2 Rating System

- Team-level Glicko-2 ratings tracking rating, deviation, and volatility
- All teams initialized at 1500 with high deviation
- Separate rating pools for SSN and internationals
- International ratings seeded from World Netball official rankings
- K-factor / update sensitivity tuned via backtesting

### Contextual Features

| Feature | Description |
|---------|-------------|
| Home/away | Binary indicator for home court advantage |
| Rest days | Days since last match for each team |
| Travel distance | Approximate km between venue cities |
| Recent form | Rolling 5-match win rate and average margin |
| Head-to-head | Win rate in last N meetings between teams |
| Roster strength | Average Nissan Net Points of starting 7 |
| Key player availability | Binary flags for star player absence |

### Target Variables

- **Match margin** (primary) — Team A score minus Team B score
- **Total goals** (secondary) — for over/under markets
- **Win probability** (derived) — P(margin > 0) from calibrated normal distribution

## Model Pipeline

### Training

1. Backfill Glicko-2 ratings by replaying all historical matches chronologically
2. Build feature matrix — snapshot pre-match ratings + features (no data leakage)
3. Train Ridge regression to predict margin
4. Fit normal distribution to residuals for probability calibration
5. Walk-forward backtest: train on 2017-2023, predict 2024 round by round

### Prediction (match day)

1. Look up current Glicko-2 ratings
2. Compute contextual features
3. Predict margin and total goals
4. Derive win probability from calibrated residual distribution
5. Compare against Betfair exchange odds
6. Flag value bets exceeding configurable edge threshold (default 5%)

### Output

CLI table per round: teams, predicted margin, predicted total, model win%, Betfair odds, implied%, edge%, value flag. Results logged to SQLite.

## Tech Stack

- **Python 3.11+**
- **Poetry** — dependency management + virtualenv
- `httpx` — async HTTP for Champion Data + Betfair APIs
- `pandas` — data manipulation
- `scikit-learn` — Ridge regression, cross-validation
- `glicko2` — Glicko-2 rating implementation
- `sqlite3` — built-in database
- `click` — CLI framework
- `rich` — terminal table formatting

## Project Structure

```
netball-model/
├── src/
│   └── netball_model/
│       ├── __init__.py
│       ├── cli.py                 # Click CLI entry point
│       ├── data/
│       │   ├── champion_data.py   # Champion Data API client
│       │   ├── betfair.py         # Betfair historical + live odds
│       │   ├── thesportsdb.py     # International results fallback
│       │   └── database.py        # SQLite schema + read/write
│       ├── features/
│       │   ├── elo.py             # Glicko-2 rating system
│       │   ├── contextual.py      # Home/away, rest, travel, form
│       │   └── builder.py         # Assemble feature matrix
│       ├── model/
│       │   ├── train.py           # Training + walk-forward validation
│       │   ├── predict.py         # Match-day predictions
│       │   └── calibration.py     # Probability calibration
│       └── value/
│           └── detector.py        # Compare model vs odds, flag edges
├── data/
│   └── netball.db                 # SQLite database (gitignored)
├── notebooks/                     # Exploratory analysis
├── tests/
├── pyproject.toml
└── docs/
    └── plans/
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `netball ingest --season 2024` | Pull match + player data from Champion Data |
| `netball odds --season 2024` | Import Betfair historical odds |
| `netball train` | Backfill Elo, build features, train model |
| `netball predict --round 5` | Predict upcoming round, compare to live odds |
| `netball backtest --seasons 2022-2024` | Walk-forward validation report |

## Key Research References

- **Betfair Historical Data:** https://historicdata.betfair.com/
- **Champion Data:** https://www.championdata.com/
- **superNetballR (API reference):** https://github.com/SteveLane/superNetballR
- **Data Sports Group Netball API:** https://datasportsgroup.com/coverage/netball/
- **TheSportsDB Netball:** https://www.thesportsdb.com/sport/netball
- **World Netball Rankings:** https://netball.sport/events-and-results/world-rankings-hub/current-world-rankings/
- **Existing model (Glicko + DL):** https://github.com/mitch-mooney/netball_prediction
- **OddsMatrix (enterprise odds):** https://oddsmatrix.com/
