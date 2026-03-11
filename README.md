# Netball Model

Super Netball match prediction model using Glicko-2 ratings, player matchup features, and Ridge regression.

## Setup

```bash
poetry install
```

## Usage

### Streamlit App (Value Finder)

```bash
poetry run streamlit run app.py
```

Opens a dashboard where you can select upcoming matches or manually enter teams, view model predictions, and compare bookmaker odds to find value bets.

### CLI

```bash
poetry run netball predict
```

### Tests

```bash
poetry run pytest -v
```
