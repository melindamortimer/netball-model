"""Netball Value Finder — Streamlit dashboard."""
from __future__ import annotations

import datetime
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel
from netball_model.value.detector import ValueDetector

_ROOT = Path(__file__).resolve().parent
DB_PATH = _ROOT / "data" / "netball.db"
MODEL_PATH = _ROOT / "data" / "model.pkl"
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


def parse_odds_paste(text: str) -> list[dict] | None:
    """Parse pasted bookmaker odds. Returns list of {bookmaker, home_odds, away_odds}.

    Expects tab-separated rows like:
      Bet365  Start  1.380  3.000  03/10 18:48  ...
              Last   1.380  3.000  03/11 22:12  ...
    Takes the "Start" row odds (opening odds). Falls back to first row with numbers.
    """
    results = []
    current_book = None
    for line in text.strip().splitlines():
        parts = [p.strip() for p in re.split(r"\t+", line) if p.strip()]
        if not parts:
            continue
        # Skip header row
        if any(h in parts for h in ("Bookmaker", "S/K/L", "Full Time Result")):
            continue
        # Detect bookmaker name — first field that isn't Start/Last/Key
        if parts[0] not in ("Start", "Last", "Key"):
            current_book = parts[0]
            parts = parts[1:]
        # Look for "Start" row
        if parts and parts[0] == "Start":
            try:
                home = float(parts[1])
                away = float(parts[2])
                results.append({
                    "bookmaker": current_book or "bet365",
                    "home_odds": home,
                    "away_odds": away,
                })
            except (IndexError, ValueError):
                pass
    return results if results else None


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
        min_edge_pct = st.slider("Min edge threshold", 0, 20, 5, 1,
                                  format="%d%%",
                                  help="Minimum model edge to flag as value")
        min_edge = min_edge_pct / 100.0

    # --- Section 1: Match Selection ---
    st.header("Match Selection")

    upcoming = [
        m for m in matches if m.get("home_score") is None
    ]

    mode = st.radio("Source", ["Upcoming (from DB)", "Manual entry"],
                    horizontal=True)

    selected_index = None
    manual_matches = matches  # default; overwritten for manual entry

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

        # Paste odds from bet365 etc.
        with st.expander("Paste odds from bet365"):
            pasted = st.text_area(
                "Paste the row from bet365 (tab-separated)",
                key="odds_paste",
                height=80,
                placeholder="Bet365\tStart\t1.380\t3.000\t...",
            )
            if st.button("Parse & fill"):
                parsed = parse_odds_paste(pasted) if pasted else None
                if parsed:
                    for p in parsed:
                        book = p["bookmaker"]
                        if book not in st.session_state["bookmakers"]:
                            st.session_state["bookmakers"].append(book)
                        st.session_state[f"home_{book}"] = p["home_odds"]
                        st.session_state[f"away_{book}"] = p["away_odds"]
                    names = ", ".join(p["bookmaker"] for p in parsed)
                    st.success(f"Filled odds for: {names}")
                    st.rerun()
                elif pasted:
                    st.error("Couldn't parse odds. Expected tab-separated rows with 'Start' and two decimal odds.")

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
                    return ["background-color: #1a7a3a; color: white"] * len(row)
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
