"""Reusable service functions extracted from CLI commands."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from netball_model.data.champion_data import ChampionDataClient
from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel


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

    builder = FeatureBuilder(matches)
    df = builder.build_matrix(start_index=1)

    model = NetballModel()
    model.train(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    preds = model.predict(df)
    mae = float((df["margin"] - preds["predicted_margin"].astype(float)).abs().mean())

    return model, mae


def backtest_season(
    db: Database, train_range: tuple[int, int], test_season: int
) -> dict:
    """Walk-forward backtest. Returns dict with matches, accuracy, mae."""
    all_matches = db.get_matches()

    start, end = train_range
    train_matches = [m for m in all_matches if start <= m["season"] <= end]
    test_matches = [m for m in all_matches if m["season"] == test_season]

    if not train_matches or not test_matches:
        raise ValueError("Insufficient data for backtest.")

    builder = FeatureBuilder(train_matches)
    train_df = builder.build_matrix(start_index=1)

    model = NetballModel()
    model.train(train_df)

    # Walk-forward: for each test match, build features using all prior matches
    all_for_test = train_matches + test_matches
    test_builder = FeatureBuilder(all_for_test)

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
