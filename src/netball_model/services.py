"""Reusable service functions extracted from CLI commands."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from netball_model.data.champion_data import ChampionDataClient
from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel

logger = logging.getLogger(__name__)


async def ingest_season(db: Database, season: int) -> int:
    """Fetch and store a full season from Champion Data. Returns match count."""
    db.initialize()

    client = ChampionDataClient()
    try:
        results = await client.fetch_season(season)

        all_matches = []
        all_stats = []
        for match, players in results:
            all_matches.append(match)
            all_stats.extend(players)

        db.upsert_matches(all_matches)
        db.insert_player_stats_batch(all_stats)

        return len(results)
    finally:
        await client.close()


def train_model(db: Database, output_path: str) -> tuple[NetballModel, float]:
    """Build features, train model, save to disk. Returns (model, training_mae)."""
    matches = db.get_matches()

    if len(matches) < 20:
        raise ValueError(f"Only {len(matches)} matches in DB. Need at least 20 to train.")

    # Load player stats for all matches
    player_stats = {}
    for m in matches:
        starters = db.get_starters_for_match(m["match_id"])
        if starters:
            player_stats[m["match_id"]] = starters

    # Compute roster continuity for all seasons
    from netball_model.data.player_movements import get_team_continuity_all
    roster_continuity = {}
    seasons = sorted(set(m["season"] for m in matches))
    for season in seasons:
        if season == seasons[0]:
            continue  # No previous season to compare
        continuity = get_team_continuity_all(season, db)
        for team, cont in continuity.items():
            roster_continuity[(team, season)] = cont

    builder = FeatureBuilder(
        matches, player_stats=player_stats, roster_continuity=roster_continuity
    )
    df = builder.build_matrix(start_index=1)

    model = NetballModel()
    model.train(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    preds = model.predict(df)
    mae = float((df["margin"] - preds["predicted_margin"].astype(float)).abs().mean())

    return model, mae


def backtest_season(
    db: Database, train_range: tuple[int, int], test_season: int,
    use_player_stats: bool = False,
) -> dict:
    """Walk-forward backtest. Returns dict with matches, accuracy, mae."""
    all_matches = db.get_matches()

    start, end = train_range
    train_matches = [m for m in all_matches if start <= m["season"] <= end]
    test_matches = [m for m in all_matches if m["season"] == test_season]

    if not train_matches or not test_matches:
        raise ValueError("Insufficient data for backtest.")

    # Load player stats if requested
    player_stats = None
    if use_player_stats:
        player_stats = {}
        all_for_backtest = train_matches + test_matches
        for m in all_for_backtest:
            starters = db.get_starters_for_match(m["match_id"])
            if starters:
                player_stats[m["match_id"]] = starters

    builder = FeatureBuilder(train_matches, player_stats=player_stats)
    train_df = builder.build_matrix(start_index=1)

    model = NetballModel()
    model.train(train_df)

    # Walk-forward: for each test match, build features using all prior matches
    all_for_test = train_matches + test_matches
    test_builder = FeatureBuilder(all_for_test, player_stats=player_stats)

    correct = 0
    total = 0
    abs_errors = []

    for i in range(len(train_matches), len(all_for_test)):
        row = test_builder.build_row(i)
        pred = model.predict(pd.DataFrame([row]))

        pred_margin = float(pred["predicted_margin"].iloc[0])
        actual_margin = row["margin"]

        abs_errors.append(abs(pred_margin - actual_margin))
        if (pred_margin > 0 and actual_margin > 0) or (pred_margin < 0 and actual_margin < 0):
            correct += 1
        total += 1

    mae = float(np.mean(abs_errors))
    accuracy = correct / total if total > 0 else 0.0

    return {
        "test_season": test_season,
        "matches": total,
        "accuracy": accuracy,
        "mae": mae,
    }


def import_betsapi_odds(
    db: Database,
    token: str,
    events: list[dict],
    season: int | None = None,
) -> dict[str, int]:
    """Fetch odds from BetsAPI for a list of events and store in DB.

    Each event dict must have: event_id, home_team, away_team.
    If season is provided, only matches DB matches from that season.
    Returns counts: {"matched": N, "unmatched": N, "no_odds": N, "total": N}.
    """
    from netball_model.data.betsapi import BetsApiClient

    db.initialize()
    matches = db.get_matches(season=season) if season else db.get_matches()
    if not matches:
        msg = f"No matches in DB for season {season}." if season else "No matches in DB."
        raise ValueError(f"{msg} Run ingest first.")

    # Build lookup: (home_team, away_team, date) -> match_id
    # Include date to disambiguate the same matchup across seasons.
    match_lookup: dict[tuple[str, str, str], str] = {}
    for m in matches:
        d = (m.get("date") or "")[:10]
        match_lookup[(m["home_team"], m["away_team"], d)] = m["match_id"]

    with BetsApiClient(token) as client:
        events_with_odds = client.fetch_odds_for_events(events)

    matched = 0
    unmatched = 0
    no_odds = 0
    odds_records: list[dict] = []

    for ev in events_with_odds:
        if ev["home_odds"] is None:
            no_odds += 1
            continue

        home = ev["home_team"]
        away = ev["away_team"]
        ev_date = ev.get("date", "")[:10]
        home_odds = ev["home_odds"]
        away_odds = ev["away_odds"]

        match_id = match_lookup.get((home, away, ev_date))
        if not match_id:
            # Try swapped home/away
            match_id = match_lookup.get((away, home, ev_date))
            if match_id:
                home_odds, away_odds = away_odds, home_odds

        if not match_id:
            logger.debug("No DB match for %s vs %s", home, away)
            unmatched += 1
            continue

        odds_records.append({
            "match_id": match_id,
            "source": "betsapi",
            "home_back_odds": home_odds,
            "home_lay_odds": None,
            "away_back_odds": away_odds,
            "away_lay_odds": None,
            "home_volume": 0,
            "away_volume": 0,
            "timestamp": ev.get("date", ""),
        })
        matched += 1

    if odds_records:
        db.upsert_odds_batch(odds_records)

    return {
        "matched": matched,
        "unmatched": unmatched,
        "no_odds": no_odds,
        "total": len(events),
    }
