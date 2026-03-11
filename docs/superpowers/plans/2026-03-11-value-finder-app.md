# Value Finder App Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit dashboard that shows model predictions and identifies value bets from manually entered bookmaker odds.

**Architecture:** Single `app.py` at repo root importing existing model/feature/value modules. One change to existing code: rename ValueDetector params from Betfair-specific to generic. Add `streamlit` dependency.

**Tech Stack:** Python, Streamlit, existing netball_model package (Ridge regression model, Glicko-2 features, SQLite DB)

**Spec:** `docs/superpowers/specs/2026-03-11-value-finder-app-design.md`

---

## Task 1: Generalise ValueDetector parameter names

Rename `betfair_home_back` / `betfair_away_back` to `home_odds` / `away_odds` in the detector, its tests, and the one CLI call site.

**Files:**
- Modify: `src/netball_model/value/detector.py`
- Modify: `tests/value/test_detector.py`
- Modify: `src/netball_model/cli.py:157-161`

- [ ] **Step 1: Update detector.py parameter names**

In `src/netball_model/value/detector.py`, rename the two parameters in `evaluate()`:

```python
def evaluate(
    self,
    home_team: str,
    away_team: str,
    model_win_prob: float,
    home_odds: float | None = None,
    away_odds: float | None = None,
) -> dict:
    model_away_prob = 1.0 - model_win_prob

    home_implied = 1 / home_odds if home_odds else None
    away_implied = 1 / away_odds if away_odds else None

    home_edge = (model_win_prob - home_implied) if home_implied else None
    away_edge = (model_away_prob - away_implied) if away_implied else None

    best_side = "home"
    best_edge = home_edge or 0
    best_model_prob = model_win_prob
    best_implied = home_implied or 0
    best_odds = home_odds

    if away_edge is not None and away_edge > (home_edge or 0):
        best_side = "away"
        best_edge = away_edge
        best_model_prob = model_away_prob
        best_implied = away_implied or 0
        best_odds = away_odds

    return {
        "home_team": home_team,
        "away_team": away_team,
        "bet_side": best_side,
        "model_prob": best_model_prob,
        "implied_prob": best_implied,
        "edge": round(best_edge, 4),
        "odds": best_odds,
        "is_value": best_edge >= self.min_edge,
    }
```

- [ ] **Step 2: Update test parameter names**

In `tests/value/test_detector.py`, rename all `betfair_home_back=` to `home_odds=` and `betfair_away_back=` to `away_odds=`. Three call sites:

```python
# test_detect_value_home
result = detector.evaluate(
    home_team="Firebirds",
    away_team="Swifts",
    model_win_prob=0.65,
    home_odds=1.80,  # implied prob = 55.6%
)

# test_no_value
result = detector.evaluate(
    home_team="Firebirds",
    away_team="Swifts",
    model_win_prob=0.55,
    home_odds=1.75,  # implied prob = 57.1%
)

# test_value_on_away
result = detector.evaluate(
    home_team="Firebirds",
    away_team="Swifts",
    model_win_prob=0.35,
    home_odds=1.60,  # implied home prob = 62.5%
    away_odds=2.50,  # implied away prob = 40%
)
```

- [ ] **Step 3: Run tests to verify rename**

Run: `poetry run pytest tests/value/ -v`
Expected: 3 tests pass

- [ ] **Step 4: Update CLI call site**

In `src/netball_model/cli.py` (the `predict` command, ~line 157), the `evaluate` call currently passes no odds. No actual change needed — it uses positional/default args. But verify nothing breaks.

Run: `poetry run pytest -v`
Expected: All 92 tests pass

---

## Task 2: Add Streamlit dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add streamlit to dependencies**

Run: `poetry add streamlit`

This adds streamlit to `[tool.poetry.dependencies]` and installs it.

- [ ] **Step 2: Verify install**

Run: `poetry run python -c "import streamlit; print(streamlit.__version__)"`
Expected: Prints a version number (e.g. `1.x.x`)

---

## Task 3: Build the Streamlit app

**Files:**
- Create: `app.py` (repo root)

This is the main task. The app has three sections: match selection, model prediction, and bookmaker value table.

- [ ] **Step 1: Create app.py with imports and cached loaders**

```python
"""Netball Value Finder — Streamlit dashboard."""
from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel
from netball_model.value.detector import ValueDetector

DB_PATH = "data/netball.db"
MODEL_PATH = "data/model.pkl"
DEFAULT_BOOKS = ["bet365", "Sportsbet", "TAB", "Ladbrokes"]


@st.cache_resource
def load_model() -> NetballModel:
    return NetballModel.load(MODEL_PATH)


@st.cache_resource
def load_db() -> Database:
    return Database(DB_PATH)


@st.cache_data
def load_matches() -> list[dict]:
    return load_db().get_matches()


@st.cache_data
def load_player_stats(matches: tuple) -> dict[str, list[dict]]:
    """Load starters for all matches. Arg is tuple for hashability."""
    db = load_db()
    stats = {}
    for mid in matches:
        s = db.get_starters_for_match(mid)
        if s:
            stats[mid] = s
    return stats


def get_teams(matches: list[dict]) -> list[str]:
    """Extract sorted unique team names from match history."""
    teams = set()
    for m in matches:
        teams.add(m["home_team"])
        teams.add(m["away_team"])
    return sorted(teams)


def predict_match(
    matches: list[dict],
    match_index: int,
    player_stats: dict | None,
) -> dict:
    """Run model prediction for a single match by index."""
    model = load_model()
    builder = FeatureBuilder(matches, player_stats=player_stats)
    row = builder.build_row(match_index)
    pred = model.predict(pd.DataFrame([row]))
    return {
        "predicted_margin": float(pred["predicted_margin"].iloc[0]),
        "predicted_total": float(pred["predicted_total"].iloc[0]),
        "win_probability": float(pred["win_probability"].iloc[0]),
    }
```

- [ ] **Step 2: Add match selection section**

Append to `app.py`:

```python
def main():
    st.set_page_config(page_title="Netball Value Finder", layout="wide")
    st.title("Netball Value Finder")

    matches = load_matches()
    teams = get_teams(matches)
    match_ids = tuple(m["match_id"] for m in matches)
    player_stats = load_player_stats(match_ids)

    # --- Sidebar config ---
    with st.sidebar:
        st.header("Settings")
        min_edge = st.slider("Min edge threshold", 0.0, 0.20, 0.05, 0.01,
                             format="%.0f%%",
                             help="Minimum model edge to flag as value")

    # --- Section 1: Match Selection ---
    st.header("Match Selection")

    upcoming = [
        m for m in matches if m.get("home_score") is None
    ]

    mode = st.radio("Source", ["Upcoming (from DB)", "Manual entry"],
                    horizontal=True)

    selected_index = None

    if mode == "Upcoming (from DB)":
        if not upcoming:
            st.warning("No upcoming matches in the database. Use `netball ingest` or switch to manual entry.")
        else:
            labels = {
                m["match_id"]: f"{m['home_team']} vs {m['away_team']} — {m.get('date', '?')[:10]}"
                for m in upcoming
            }
            chosen_id = st.selectbox("Match", list(labels.keys()),
                                     format_func=lambda x: labels[x])
            selected_index = next(
                i for i, m in enumerate(matches) if m["match_id"] == chosen_id
            )
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            home_team = st.selectbox("Home team", teams)
        with col2:
            away_team = st.selectbox("Away team",
                                     [t for t in teams if t != home_team])
        with col3:
            match_date = st.date_input("Date", datetime.date.today())

        # For manual entry, append a temporary match to the list and use its index
        if home_team and away_team:
            temp_match = {
                "match_id": f"manual_{home_team}_{away_team}",
                "date": match_date.isoformat(),
                "round_num": 0,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": None,
                "away_score": None,
                "venue": "",
            }
            manual_matches = matches + [temp_match]
            selected_index = len(manual_matches) - 1
```

- [ ] **Step 3: Add prediction and value table sections**

Append to `main()` inside `app.py`:

```python
    # --- Section 2 & 3: Prediction + Value ---
    if st.button("Predict", type="primary"):
        if selected_index is None:
            st.error("Select a match first.")
        else:
            match_list = manual_matches if mode != "Upcoming (from DB)" else matches
            m = match_list[selected_index]

            with st.spinner("Running model..."):
                prediction = predict_match(
                    match_list, selected_index,
                    player_stats if player_stats else None,
                )

            st.session_state["prediction"] = prediction
            st.session_state["match"] = m

    # Display prediction if available
    if "prediction" in st.session_state:
        prediction = st.session_state["prediction"]
        m = st.session_state["match"]
        win_prob = prediction["win_probability"]
        away_prob = 1.0 - win_prob

        st.header("Model Prediction")
        st.markdown(f"### {m['home_team']} vs {m['away_team']}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Predicted Margin", f"{prediction['predicted_margin']:+.1f}")
        col2.metric(f"{m['home_team']} Win%", f"{win_prob:.1%}")
        col3.metric(f"{m['away_team']} Win%", f"{away_prob:.1%}")
        col4.metric("Fair Home Odds", f"{1/win_prob:.2f}" if win_prob > 0 else "-")
        col5.metric("Fair Away Odds", f"{1/away_prob:.2f}" if away_prob > 0 else "-")

        # --- Section 3: Bookmaker Odds ---
        st.header("Bookmaker Odds & Value")

        detector = ValueDetector(min_edge=min_edge)

        # Manage bookmaker list in session state
        if "bookmakers" not in st.session_state:
            st.session_state["bookmakers"] = list(DEFAULT_BOOKS)

        # Add/remove bookmaker controls
        add_col, remove_col = st.columns(2)
        with add_col:
            new_book = st.text_input("Add bookmaker", key="new_book",
                                     placeholder="e.g. PointsBet")
            if st.button("Add") and new_book:
                if new_book not in st.session_state["bookmakers"]:
                    st.session_state["bookmakers"].append(new_book)
                    st.rerun()
        with remove_col:
            if len(st.session_state["bookmakers"]) > 1:
                remove_book = st.selectbox("Remove bookmaker",
                                           st.session_state["bookmakers"],
                                           key="remove_book")
                if st.button("Remove"):
                    st.session_state["bookmakers"].remove(remove_book)
                    st.rerun()

        # Odds input table
        results = []
        for book in st.session_state["bookmakers"]:
            col1, col2 = st.columns(2)
            with col1:
                home_odds = st.number_input(
                    f"{book} — {m['home_team']}",
                    min_value=1.01, max_value=100.0, value=None,
                    step=0.01, format="%.2f",
                    key=f"home_{book}",
                    placeholder="Home odds",
                )
            with col2:
                away_odds = st.number_input(
                    f"{book} — {m['away_team']}",
                    min_value=1.01, max_value=100.0, value=None,
                    step=0.01, format="%.2f",
                    key=f"away_{book}",
                    placeholder="Away odds",
                )

            if home_odds or away_odds:
                result = detector.evaluate(
                    home_team=m["home_team"],
                    away_team=m["away_team"],
                    model_win_prob=win_prob,
                    home_odds=home_odds,
                    away_odds=away_odds,
                )
                result["bookmaker"] = book
                results.append(result)

        # Value summary table
        if results:
            st.subheader("Value Summary")

            table_data = []
            for r in results:
                home_implied = (1 / r["odds"]) if r["bet_side"] == "home" and r["odds"] else None
                away_implied = (1 / r["odds"]) if r["bet_side"] == "away" and r["odds"] else None

                # Compute both edges for display
                h_odds = st.session_state.get(f"home_{r['bookmaker']}")
                a_odds = st.session_state.get(f"away_{r['bookmaker']}")
                h_edge = win_prob - (1 / h_odds) if h_odds else None
                a_edge = away_prob - (1 / a_odds) if a_odds else None

                table_data.append({
                    "Bookmaker": r["bookmaker"],
                    "Home Odds": f"{h_odds:.2f}" if h_odds else "-",
                    "Away Odds": f"{a_odds:.2f}" if a_odds else "-",
                    "Home Edge": f"{h_edge:+.1%}" if h_edge is not None else "-",
                    "Away Edge": f"{a_edge:+.1%}" if a_edge is not None else "-",
                    "Value?": "Yes" if r["is_value"] else "No",
                })

            df = pd.DataFrame(table_data)

            def highlight_value(row):
                if row["Value?"] == "Yes":
                    return ["background-color: #d4edda"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df.style.apply(highlight_value, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # Best value callout
            value_bets = [r for r in results if r["is_value"]]
            for v in value_bets:
                team = m["home_team"] if v["bet_side"] == "home" else m["away_team"]
                st.success(
                    f"Back **{team}** at **{v['bookmaker']}** — "
                    f"{v['edge']:.1%} edge (odds {v['odds']:.2f})"
                )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the app starts**

Run: `poetry run streamlit run app.py --server.headless true`
Expected: App starts on port 8501 without errors. Ctrl-C to stop.

- [ ] **Step 5: Manual smoke test**

Run `poetry run streamlit run app.py` and verify in browser:
1. Page loads with "Netball Value Finder" title
2. Upcoming matches appear in dropdown (or "no upcoming" warning if none)
3. Manual entry shows team dropdowns populated from DB
4. Predict button produces margin/win prob/fair odds
5. Entering bet365 odds 1.38 / 3.00 for a match shows value calculation
6. Green highlight appears when edge >= threshold

---

## Task 4: Run full test suite

**Files:** None (verification only)

- [ ] **Step 1: Run all existing tests**

Run: `poetry run pytest -v`
Expected: All 92 tests pass (the only code change was the parameter rename in Task 1)
