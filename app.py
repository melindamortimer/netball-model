"""Netball Value Finder — Streamlit dashboard."""
from __future__ import annotations

import base64
import datetime
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from netball_model.data.database import Database
from netball_model.data.player_movements import _find_player_id_by_name
from netball_model.data.squads import get_all_squads
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel
from netball_model.value.detector import ValueDetector

_ROOT = Path(__file__).resolve().parent
DB_PATH = _ROOT / "data" / "netball.db"
MODEL_PATH = _ROOT / "data" / "model.pkl"

_paste_component = components.declare_component(
    "paste_image",
    path=str(_ROOT / "components" / "paste_image"),
)


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
    teams = set()
    for m in matches:
        teams.add(m["home_team"])
        teams.add(m["away_team"])
    return sorted(teams)


@st.cache_data
def build_squad_starters(matches: tuple) -> dict[str, dict[str, tuple]]:
    """Resolve 2026 squad names to player_ids for upcoming match predictions."""
    db = load_db()
    all_matches = db.get_matches()
    squads = get_all_squads(2026)
    resolved: dict[str, dict[str, tuple[int | None, str]]] = {}
    for team, positions in squads.items():
        resolved[team] = {}
        for pos, name in positions.items():
            pid = _find_player_id_by_name(name, all_matches, db)
            resolved[team][pos] = (pid, name)
    return resolved


def _make_synthetic_starters(
    match_id: str, home_team: str, away_team: str, resolved_squads: dict,
) -> list[dict]:
    starters = []
    for team in (home_team, away_team):
        squad = resolved_squads.get(team, {})
        for pos, (pid, name) in squad.items():
            if pid is None:
                continue
            starters.append({
                "match_id": match_id, "player_id": pid,
                "player_name": name, "team": team, "position": pos,
                "goals": 0, "attempts": 0, "assists": 0, "rebounds": 0,
                "feeds": 0, "turnovers": 0, "gains": 0, "intercepts": 0,
                "deflections": 0, "penalties": 0, "centre_pass_receives": 0,
            })
    return starters


def save_odds_json(
    matches: list[dict],
    output_dir: Path | None = None,
) -> Path:
    """Save parsed odds to a JSON file with append + deduplication.

    Filename is based on today's date (session date).
    If the file exists, appends new matches and deduplicates by
    home_team + away_team + date.
    """
    import json

    if output_dir is None:
        output_dir = _ROOT / "data" / "odds"
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today().isoformat()
    filepath = output_dir / f"{today}.json"

    # Load existing data if file exists
    if filepath.exists():
        existing = json.loads(filepath.read_text())
        existing_matches = existing.get("matches", [])
    else:
        existing_matches = []

    # Dedup key: home_team + away_team + date
    dedup_keys = {
        (m["home_team"], m["away_team"], m.get("date", ""))
        for m in matches
    }
    # Keep existing matches that don't conflict with new ones
    kept = [
        m for m in existing_matches
        if (m["home_team"], m["away_team"], m.get("date", "")) not in dedup_keys
    ]
    all_matches = kept + matches

    data = {
        "session_date": today,
        "source": "bet365_screenshot",
        "matches": all_matches,
    }
    filepath.write_text(json.dumps(data, indent=2))
    return filepath


def predict_match(
    matches: list[dict], match_index: int, player_stats: dict | None,
) -> dict:
    model = load_model()
    m = matches[match_index]
    if player_stats is not None and m["match_id"] not in player_stats:
        match_ids = tuple(mm["match_id"] for mm in matches if mm.get("home_score") is not None)
        resolved = build_squad_starters(match_ids)
        synthetic = _make_synthetic_starters(
            m["match_id"], m["home_team"], m["away_team"], resolved,
        )
        if synthetic:
            player_stats = dict(player_stats)
            player_stats[m["match_id"]] = synthetic

    builder = FeatureBuilder(matches, player_stats=player_stats)
    row = builder.build_row(match_index)
    pred = model.predict(pd.DataFrame([row]))
    return {
        "predicted_margin": float(pred["predicted_margin"].iloc[0]),
        "predicted_total": float(pred["predicted_total"].iloc[0]),
        "win_probability": float(pred["win_probability"].iloc[0]),
    }


def _parse_screenshot_image(image_bytes: bytes) -> dict | None:
    """Run OCR on image bytes. Returns parsed odds dict or None."""
    try:
        from netball_model.data.bet365_screenshot import parse_screenshot
    except ImportError:
        st.error("easyocr not installed. Run `poetry install`.")
        return None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        f.flush()
        temp_path = f.name

    try:
        parsed = parse_screenshot(temp_path)
    except Exception as e:
        st.error(f"Screenshot parse failed: {e}")
        return None
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return parsed


def main():
    st.set_page_config(page_title="Netball Value Finder", layout="wide")
    st.title("Netball Value Finder")

    matches = load_matches()
    teams = get_teams(matches)
    match_ids = tuple(m["match_id"] for m in matches)
    player_stats = load_player_stats(match_ids)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        min_edge_pct = st.slider("Min edge %", 0, 20, 5, 1, format="%d%%")
        min_edge = min_edge_pct / 100.0

    # --- Match Selection ---
    st.header("Match Selection")
    upcoming = [m for m in matches if m.get("home_score") is None]
    mode = st.radio("Source", ["Upcoming (from DB)", "Screenshot round"], horizontal=True)

    if mode == "Upcoming (from DB)":
        if not upcoming:
            st.warning("No upcoming matches in the database. Switch to Screenshot round.")
        else:
            # Build summary table with stored odds
            db = load_db()
            summary_rows = []
            for m in upcoming:
                stored = db.get_odds_for_match(m["match_id"]) or {}
                def _fmt(v):
                    return f"{v:.2f}" if v else "—"
                summary_rows.append({
                    "Date": (m.get("date") or "")[:10],
                    "Home Team": m["home_team"],
                    "Away Team": m["away_team"],
                    "H2H Home": _fmt(stored.get("home_back_odds")),
                    "H2H Away": _fmt(stored.get("away_back_odds")),
                    "Total": f"{stored['total_line']:.1f}" if stored.get("total_line") else "—",
                    "Over": _fmt(stored.get("over_odds")),
                    "Under": _fmt(stored.get("under_odds")),
                    "Line": f"{stored['handicap_line']:.1f}" if stored.get("handicap_line") is not None else "—",
                    "HC Home": _fmt(stored.get("handicap_home_odds")),
                    "HC Away": _fmt(stored.get("handicap_away_odds")),
                })
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )
    if mode == "Screenshot round":
        # Initialize wizard state
        if "wizard_step" not in st.session_state:
            st.session_state["wizard_step"] = "upload"
        if "pasted_images" not in st.session_state:
            st.session_state["pasted_images"] = []
        if "parsed_matches" not in st.session_state:
            st.session_state["parsed_matches"] = []
        if "last_paste_hash" not in st.session_state:
            st.session_state["last_paste_hash"] = None

        if st.session_state["wizard_step"] == "upload":
            st.subheader("Step 1: Upload Screenshots")
            st.caption("Upload bet365 match screenshots — one per match")

            # File uploader (multiple files)
            uploaded_files = st.file_uploader(
                "Drop all screenshots here",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="screenshot_uploader",
            )

            # Paste component
            st.markdown("**Or paste from clipboard:**")
            pasted_data = _paste_component(key="paste_screenshot", default=None)

            # Deduplicated paste append
            if pasted_data:
                paste_hash = hash(pasted_data[:100])
                if paste_hash != st.session_state["last_paste_hash"]:
                    try:
                        _, b64 = pasted_data.split(",", 1)
                        img_bytes = base64.b64decode(b64)
                        st.session_state["pasted_images"].append(img_bytes)
                        st.session_state["last_paste_hash"] = paste_hash
                    except Exception:
                        st.error("Could not decode pasted image.")

            # Show collection summary
            n_uploaded = len(uploaded_files) if uploaded_files else 0
            n_pasted = len(st.session_state["pasted_images"])
            total = n_uploaded + n_pasted
            if total > 0:
                st.info(f"{total} screenshot(s) collected ({n_uploaded} uploaded, {n_pasted} pasted)")

            # Parse all button
            if total > 0 and st.button("Parse all screenshots", type="primary"):
                all_images = []
                if uploaded_files:
                    for f in uploaded_files:
                        all_images.append(f.getvalue())
                all_images.extend(st.session_state["pasted_images"])

                parsed_list = []
                parsed_images = []
                progress = st.progress(0, text="Parsing screenshots...")
                for idx, img_bytes in enumerate(all_images):
                    progress.progress(
                        (idx + 1) / len(all_images),
                        text=f"Parsing {idx + 1}/{len(all_images)}...",
                    )
                    parsed = _parse_screenshot_image(img_bytes)
                    if parsed:
                        from netball_model.data.team_names import normalise_team
                        home = normalise_team(parsed.get("home_team") or "")
                        away = normalise_team(parsed.get("away_team") or "")
                        parsed_list.append({
                            "Date": datetime.date.today(),
                            "Home Team": home if home else teams[0],
                            "Away Team": away if away else (teams[1] if len(teams) > 1 else teams[0]),
                            "H2H Home": parsed.get("home_odds"),
                            "H2H Away": parsed.get("away_odds"),
                            "Total": parsed.get("total_line"),
                            "Over": parsed.get("over_odds"),
                            "Under": parsed.get("under_odds"),
                            "Line": parsed.get("handicap_line"),
                            "HC Home": parsed.get("handicap_home_odds"),
                            "HC Away": parsed.get("handicap_away_odds"),
                        })
                        parsed_images.append(img_bytes)
                    else:
                        st.warning(f"Screenshot {idx + 1} could not be parsed — skipped")

                if parsed_list:
                    st.session_state["parsed_matches"] = parsed_list
                    st.session_state["parsed_images"] = parsed_images
                    st.session_state["wizard_step"] = "review"
                    st.rerun()
                else:
                    st.error("No screenshots could be parsed. Check the images and try again.")

        elif st.session_state["wizard_step"] == "review":
            st.subheader("Step 2: Review & Edit Parsed Odds")
            st.caption("Fix any OCR errors, add missing matches, delete bad rows")

            # Show original screenshots for reference (carousel)
            stored_images = st.session_state.get("parsed_images", [])
            if stored_images:
                if "screenshot_idx" not in st.session_state:
                    st.session_state["screenshot_idx"] = 0
                idx = st.session_state["screenshot_idx"]
                idx = max(0, min(idx, len(stored_images) - 1))

                with st.expander(f"Original screenshots ({len(stored_images)})", expanded=True):
                    nav1, nav2, nav3 = st.columns([1, 6, 1])
                    with nav1:
                        if st.button("\u25c0 Prev", disabled=idx == 0, key="ss_prev"):
                            st.session_state["screenshot_idx"] = idx - 1
                            st.rerun()
                    with nav3:
                        if st.button("Next \u25b6", disabled=idx >= len(stored_images) - 1, key="ss_next"):
                            st.session_state["screenshot_idx"] = idx + 1
                            st.rerun()
                    with nav2:
                        st.markdown(f"**Screenshot {idx + 1} of {len(stored_images)}**")
                    st.image(stored_images[idx], use_container_width=True)

            teams = get_teams(matches)

            # Build DataFrame from parsed matches
            df = pd.DataFrame(st.session_state["parsed_matches"])

            # Configure column types for data_editor
            column_config = {
                "Date": st.column_config.DateColumn("Date", default=datetime.date.today()),
                "Home Team": st.column_config.SelectboxColumn(
                    "Home Team", options=teams, required=True,
                ),
                "Away Team": st.column_config.SelectboxColumn(
                    "Away Team", options=teams, required=True,
                ),
                "H2H Home": st.column_config.NumberColumn("H2H Home", min_value=1.01, max_value=100.0, format="%.2f"),
                "H2H Away": st.column_config.NumberColumn("H2H Away", min_value=1.01, max_value=100.0, format="%.2f"),
                "Total": st.column_config.NumberColumn("Total", min_value=50.0, max_value=200.0, format="%.1f"),
                "Over": st.column_config.NumberColumn("Over", min_value=1.01, max_value=100.0, format="%.2f"),
                "Under": st.column_config.NumberColumn("Under", min_value=1.01, max_value=100.0, format="%.2f"),
                "Line": st.column_config.NumberColumn("Line", min_value=-50.0, max_value=50.0, format="%.1f"),
                "HC Home": st.column_config.NumberColumn("HC Home", min_value=1.01, max_value=100.0, format="%.2f"),
                "HC Away": st.column_config.NumberColumn("HC Away", min_value=1.01, max_value=100.0, format="%.2f"),
            }

            column_order = [
                "Date", "Home Team", "Away Team",
                "H2H Home", "H2H Away",
                "Total", "Over", "Under",
                "Line", "HC Home", "HC Away",
            ]

            edited_df = st.data_editor(
                df,
                column_config=column_config,
                column_order=column_order,
                num_rows="dynamic",  # allows add/delete rows
                use_container_width=True,
                hide_index=True,
                key="odds_editor",
            )

            if st.button("Confirm & Predict", type="primary"):
                if edited_df.empty:
                    st.error("No matches to predict. Add at least one row.")
                else:
                    st.session_state["confirmed_matches"] = edited_df.to_dict("records")
                    st.session_state["wizard_step"] = "predict"
                    st.rerun()

        elif st.session_state["wizard_step"] == "predict":
            st.subheader("Step 3: Predictions & Value")

            confirmed = st.session_state.get("confirmed_matches", [])
            if not confirmed:
                st.warning("No matches confirmed.")
            else:
                model = load_model()
                detector = ValueDetector(min_edge=min_edge)

                # --- Save odds to DB + JSON ---
                db = load_db()
                odds_for_db = []
                odds_for_json = []
                for row in confirmed:
                    match_id = f"manual_{row['Home Team']}_{row['Away Team']}"
                    date_val = row.get("Date")
                    if isinstance(date_val, datetime.date):
                        date_str = date_val.isoformat()
                    else:
                        date_str = str(date_val) if date_val else datetime.date.today().isoformat()

                    odds_for_db.append({
                        "match_id": match_id,
                        "source": "bet365_screenshot",
                        "home_back_odds": row.get("H2H Home"),
                        "away_back_odds": row.get("H2H Away"),
                        "home_lay_odds": None,
                        "away_lay_odds": None,
                        "home_volume": None,
                        "away_volume": None,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "handicap_line": row.get("Line"),
                        "handicap_home_odds": row.get("HC Home"),
                        "handicap_away_odds": row.get("HC Away"),
                        "total_line": row.get("Total"),
                        "over_odds": row.get("Over"),
                        "under_odds": row.get("Under"),
                    })
                    odds_for_json.append({
                        "date": date_str,
                        "home_team": row["Home Team"],
                        "away_team": row["Away Team"],
                        "home_odds": row.get("H2H Home"),
                        "away_odds": row.get("H2H Away"),
                        "handicap_line": row.get("Line"),
                        "handicap_home_odds": row.get("HC Home"),
                        "handicap_away_odds": row.get("HC Away"),
                        "total_line": row.get("Total"),
                        "over_odds": row.get("Over"),
                        "under_odds": row.get("Under"),
                    })

                try:
                    db.upsert_odds_extended_batch(odds_for_db)
                except Exception as e:
                    st.warning(f"Could not save to DB: {e}")

                # --- Save matches to DB so they appear in "Upcoming (from DB)" ---
                matches_for_db = []
                for row in confirmed:
                    date_val = row.get("Date")
                    if isinstance(date_val, datetime.date):
                        date_str = date_val.isoformat()
                        season = date_val.year
                    else:
                        date_str = str(date_val) if date_val else datetime.date.today().isoformat()
                        season = datetime.date.today().year
                    match_id = f"manual_{row['Home Team']}_{row['Away Team']}"
                    matches_for_db.append({
                        "match_id": match_id,
                        "competition_id": 0,
                        "season": season,
                        "round_num": 0,
                        "game_num": 0,
                        "date": date_str,
                        "venue": "",
                        "home_team": row["Home Team"],
                        "away_team": row["Away Team"],
                        "home_score": None,
                        "away_score": None,
                        "home_q1": None, "home_q2": None,
                        "home_q3": None, "home_q4": None,
                        "away_q1": None, "away_q2": None,
                        "away_q3": None, "away_q4": None,
                    })
                try:
                    db.upsert_matches(matches_for_db)
                    load_matches.clear()  # bust cache so Upcoming mode sees them
                except Exception as e:
                    st.warning(f"Could not save matches to DB: {e}")

                json_path = save_odds_json(odds_for_json)
                st.success(f"Saved odds + matches to DB + {json_path.relative_to(_ROOT)}")

                # --- Run predictions ---
                results_rows = []
                progress = st.progress(0, text="Running predictions...")
                for idx, row in enumerate(confirmed):
                    progress.progress(
                        (idx + 1) / len(confirmed),
                        text=f"Predicting {idx + 1}/{len(confirmed)}...",
                    )
                    home = row["Home Team"]
                    away = row["Away Team"]
                    date_val = row.get("Date")
                    if isinstance(date_val, datetime.date):
                        date_str = date_val.isoformat()
                        season = date_val.year
                    else:
                        date_str = str(date_val) if date_val else datetime.date.today().isoformat()
                        season = datetime.date.today().year

                    # Build independent match list (base matches + this one match)
                    temp_match = {
                        "match_id": f"manual_{home}_{away}",
                        "date": date_str,
                        "season": season,
                        "round_num": 0,
                        "home_team": home,
                        "away_team": away,
                        "home_score": None,
                        "away_score": None,
                        "venue": "",
                    }
                    match_list = matches + [temp_match]
                    match_index = len(match_list) - 1

                    prediction = predict_match(
                        match_list, match_index,
                        player_stats if player_stats else None,
                    )

                    # Evaluate value
                    win_prob = prediction["win_probability"]
                    pred_dict = {
                        "margin": prediction["predicted_margin"],
                        "total_goals": prediction["predicted_total"],
                        "win_prob": win_prob,
                        "residual_std": model.residual_std,
                        "total_residual_std": model.total_residual_std,
                    }
                    odds_dict = {}
                    if row.get("H2H Home"):
                        odds_dict["home_odds"] = row["H2H Home"]
                    if row.get("H2H Away"):
                        odds_dict["away_odds"] = row["H2H Away"]
                    if row.get("Line") is not None:
                        odds_dict["handicap_line"] = row["Line"]
                        odds_dict["handicap_home_odds"] = row.get("HC Home")
                        odds_dict["handicap_away_odds"] = row.get("HC Away")
                    if row.get("Total") is not None:
                        odds_dict["total_line"] = row["Total"]
                        odds_dict["over_odds"] = row.get("Over")
                        odds_dict["under_odds"] = row.get("Under")

                    value_results = detector.evaluate(pred_dict, odds_dict) if odds_dict else []

                    # Extract best edge per market
                    def best_edge(market_name):
                        market_results = [v for v in value_results if v["market"] == market_name]
                        if not market_results:
                            return None
                        positive = [v for v in market_results if v["edge"] > 0]
                        if positive:
                            return max(positive, key=lambda v: v["edge"])
                        return max(market_results, key=lambda v: v["edge"])

                    h2h_best = best_edge("h2h")
                    hc_best = best_edge("handicap")
                    tot_best = best_edge("total")

                    def _fmt_odds(v):
                        return f"{v:.2f}" if v else "—"

                    # Per-cell value flags (which specific odds cell to highlight)
                    def _has_value(market, side):
                        return any(
                            v["market"] == market and v["side"] == side and v["edge"] >= min_edge
                            for v in value_results
                        )

                    results_rows.append({
                        "Match": f"{home} v {away}",
                        "Date": date_str[:10],
                        "Margin": f"{prediction['predicted_margin']:+.1f}",
                        "Home Win%": f"{win_prob:.0%}",
                        "Pred Total": f"{prediction['predicted_total']:.1f}",
                        # H2H market
                        "Home": _fmt_odds(row.get("H2H Home")),
                        "Away": _fmt_odds(row.get("H2H Away")),
                        "H2H Edge": f"{h2h_best['edge']:+.1%}" if h2h_best else "—",
                        # Totals market
                        "Total": f"{row['Total']:.1f}" if row.get("Total") else "—",
                        "Over": _fmt_odds(row.get("Over")),
                        "Under": _fmt_odds(row.get("Under")),
                        "Tot Edge": f"{tot_best['edge']:+.1%}" if tot_best else "—",
                        # Handicap market
                        "HC Line": f"{row['Line']:.1f}" if row.get("Line") is not None else "—",
                        "HC Home": _fmt_odds(row.get("HC Home")),
                        "HC Away": _fmt_odds(row.get("HC Away")),
                        "HC Edge": f"{hc_best['edge']:+.1%}" if hc_best else "—",
                        # per-cell flags for highlighting the odds to bet on
                        "_val_home": _has_value("h2h", "home"),
                        "_val_away": _has_value("h2h", "away"),
                        "_val_over": _has_value("total", "over"),
                        "_val_under": _has_value("total", "under"),
                        "_val_hc_home": _has_value("handicap", "home"),
                        "_val_hc_away": _has_value("handicap", "away"),
                        "_value_details": value_results,
                        "_home": home,
                        "_away": away,
                    })

                progress.empty()

                # --- Predictions table ---
                st.markdown("#### Predictions")
                pred_df = pd.DataFrame(results_rows)
                pred_cols = ["Match", "Date", "Margin", "Home Win%", "Pred Total"]
                st.dataframe(
                    pred_df[pred_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                # --- Markets & Value table ---
                st.markdown("#### Markets & Value")
                market_cols = ["Match",
                               "Home", "Away", "H2H Edge",
                               "Total", "Over", "Under", "Tot Edge",
                               "HC Line", "HC Home", "HC Away", "HC Edge"]
                market_df = pred_df[market_cols].copy()

                # Map odds columns to their per-cell value flags
                odds_flag_cols = {
                    "Home": "_val_home", "Away": "_val_away",
                    "Over": "_val_over", "Under": "_val_under",
                    "HC Home": "_val_hc_home", "HC Away": "_val_hc_away",
                }

                def highlight_cells(row):
                    idx = row.name
                    styles = [""] * len(row)
                    green = "background-color: #1a7a3a; color: white; font-weight: bold"
                    for col_name, flag_col in odds_flag_cols.items():
                        if col_name in row.index and pred_df[flag_col].iloc[idx]:
                            col_pos = list(row.index).index(col_name)
                            styles[col_pos] = green
                    return styles

                st.dataframe(
                    market_df.style.apply(highlight_cells, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                # --- Value callouts ---
                for row in results_rows:
                    for v in row["_value_details"]:
                        if v.get("edge", 0) >= min_edge:
                            side = v["side"]
                            market = v["market"]
                            if market == "h2h":
                                label = row["_home"] if side == "home" else row["_away"]
                            elif market == "handicap":
                                label = f"{'Home' if side == 'home' else 'Away'} ({v.get('line', '')})"
                            else:
                                label = f"{side.title()} {v.get('line', '')}"
                            st.success(
                                f"**{row['Match']}** — {market.upper()} — {label} — "
                                f"{v['edge']:.1%} edge (odds {v.get('odds', 0):.2f})"
                            )

                # Reset button
                if st.button("Start new round"):
                    for key in ["wizard_step", "pasted_images", "parsed_matches",
                                "parsed_images", "screenshot_idx", "last_paste_hash",
                                "confirmed_matches"]:
                        st.session_state.pop(key, None)
                    st.rerun()

    # --- Predict all upcoming (from DB mode) ---
    if mode == "Upcoming (from DB)":
        if not upcoming:
            return

        if st.button("Predict all", type="primary"):
            model = load_model()
            detector = ValueDetector(min_edge=min_edge)
            db = load_db()

            results_rows = []
            progress = st.progress(0, text="Running predictions...")
            for idx_u, m in enumerate(upcoming):
                progress.progress(
                    (idx_u + 1) / len(upcoming),
                    text=f"Predicting {idx_u + 1}/{len(upcoming)}...",
                )
                match_index = next(
                    i for i, mm in enumerate(matches) if mm["match_id"] == m["match_id"]
                )
                prediction = predict_match(
                    matches, match_index,
                    player_stats if player_stats else None,
                )

                # Load stored odds from DB
                stored = db.get_odds_for_match(m["match_id"]) or {}
                win_prob = prediction["win_probability"]
                pred_dict = {
                    "margin": prediction["predicted_margin"],
                    "total_goals": prediction["predicted_total"],
                    "win_prob": win_prob,
                    "residual_std": model.residual_std,
                    "total_residual_std": model.total_residual_std,
                }

                odds_dict = {}
                h2h_home = stored.get("home_back_odds")
                h2h_away = stored.get("away_back_odds")
                if h2h_home:
                    odds_dict["home_odds"] = h2h_home
                if h2h_away:
                    odds_dict["away_odds"] = h2h_away
                if stored.get("handicap_line") is not None:
                    odds_dict["handicap_line"] = stored["handicap_line"]
                    odds_dict["handicap_home_odds"] = stored.get("handicap_home_odds")
                    odds_dict["handicap_away_odds"] = stored.get("handicap_away_odds")
                if stored.get("total_line") is not None:
                    odds_dict["total_line"] = stored["total_line"]
                    odds_dict["over_odds"] = stored.get("over_odds")
                    odds_dict["under_odds"] = stored.get("under_odds")

                value_results = detector.evaluate(pred_dict, odds_dict) if odds_dict else []

                def best_edge(market_name):
                    market_results = [v for v in value_results if v["market"] == market_name]
                    if not market_results:
                        return None
                    positive = [v for v in market_results if v["edge"] > 0]
                    if positive:
                        return max(positive, key=lambda v: v["edge"])
                    return max(market_results, key=lambda v: v["edge"])

                h2h_best = best_edge("h2h")
                hc_best = best_edge("handicap")
                tot_best = best_edge("total")

                def _fmt_odds(v):
                    return f"{v:.2f}" if v else "—"

                # Per-cell value flags
                def _has_value(market, side):
                    return any(
                        v["market"] == market and v["side"] == side and v["edge"] >= min_edge
                        for v in value_results
                    )

                results_rows.append({
                    "Match": f"{m['home_team']} v {m['away_team']}",
                    "Date": (m.get("date") or "")[:10],
                    "Margin": f"{prediction['predicted_margin']:+.1f}",
                    "Home Win%": f"{win_prob:.0%}",
                    "Pred Total": f"{prediction['predicted_total']:.1f}",
                    "Home": _fmt_odds(h2h_home),
                    "Away": _fmt_odds(h2h_away),
                    "H2H Edge": f"{h2h_best['edge']:+.1%}" if h2h_best else "—",
                    "Total": f"{stored['total_line']:.1f}" if stored.get("total_line") else "—",
                    "Over": _fmt_odds(stored.get("over_odds")),
                    "Under": _fmt_odds(stored.get("under_odds")),
                    "Tot Edge": f"{tot_best['edge']:+.1%}" if tot_best else "—",
                    "HC Line": f"{stored['handicap_line']:.1f}" if stored.get("handicap_line") is not None else "—",
                    "HC Home": _fmt_odds(stored.get("handicap_home_odds")),
                    "HC Away": _fmt_odds(stored.get("handicap_away_odds")),
                    "HC Edge": f"{hc_best['edge']:+.1%}" if hc_best else "—",
                    "_val_home": _has_value("h2h", "home"),
                    "_val_away": _has_value("h2h", "away"),
                    "_val_over": _has_value("total", "over"),
                    "_val_under": _has_value("total", "under"),
                    "_val_hc_home": _has_value("handicap", "home"),
                    "_val_hc_away": _has_value("handicap", "away"),
                    "_value_details": value_results,
                    "_home": m["home_team"],
                    "_away": m["away_team"],
                })

            progress.empty()
            st.session_state["upcoming_results"] = results_rows

        # --- Display results ---
        if "upcoming_results" not in st.session_state:
            return

        results_rows = st.session_state["upcoming_results"]

        st.markdown("#### Predictions")
        pred_df = pd.DataFrame(results_rows)
        pred_cols = ["Match", "Date", "Margin", "Home Win%", "Pred Total"]
        st.dataframe(pred_df[pred_cols], use_container_width=True, hide_index=True)

        st.markdown("#### Markets & Value")
        market_cols = ["Match",
                       "Home", "Away", "H2H Edge",
                       "Total", "Over", "Under", "Tot Edge",
                       "HC Line", "HC Home", "HC Away", "HC Edge"]
        market_df = pred_df[market_cols].copy()

        odds_flag_cols = {
            "Home": "_val_home", "Away": "_val_away",
            "Over": "_val_over", "Under": "_val_under",
            "HC Home": "_val_hc_home", "HC Away": "_val_hc_away",
        }

        def highlight_cells(row):
            idx = row.name
            styles = [""] * len(row)
            green = "background-color: #1a7a3a; color: white; font-weight: bold"
            for col_name, flag_col in odds_flag_cols.items():
                if col_name in row.index and pred_df[flag_col].iloc[idx]:
                    col_pos = list(row.index).index(col_name)
                    styles[col_pos] = green
            return styles

        st.dataframe(
            market_df.style.apply(highlight_cells, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # --- Value callouts ---
        for row in results_rows:
            for v in row["_value_details"]:
                if v.get("edge", 0) >= min_edge:
                    side = v["side"]
                    market = v["market"]
                    if market == "h2h":
                        label = row["_home"] if side == "home" else row["_away"]
                    elif market == "handicap":
                        label = f"{'Home' if side == 'home' else 'Away'} ({v.get('line', '')})"
                    else:
                        label = f"{side.title()} {v.get('line', '')}"
                    st.success(
                        f"**{row['Match']}** — {market.upper()} — {label} — "
                        f"{v['edge']:.1%} edge (odds {v.get('odds', 0):.2f})"
                    )


if __name__ == "__main__":
    main()
