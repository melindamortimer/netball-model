import asyncio
from pathlib import Path

import click

from netball_model.data.betfair import BetfairParser
from netball_model.data.champion_data import ChampionDataClient
from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel
from netball_model.model.predict import display_predictions
from netball_model.value.detector import ValueDetector

DEFAULT_DB = "data/netball.db"


@click.group()
def main():
    """Netball betting model CLI."""
    pass


@main.command()
@click.option("--season", required=True, type=int, help="SSN season year (e.g. 2024)")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def ingest(season: int, db: str):
    """Pull match + player data from Champion Data."""
    asyncio.run(_ingest(season, db))


async def _ingest(season: int, db_path: str):
    db = Database(db_path)
    db.initialize()

    client = ChampionDataClient()
    try:
        click.echo(f"Fetching SSN {season} data from Champion Data...")
        results = await client.fetch_season(season)

        for match, players in results:
            db.upsert_match(match)
            for p in players:
                db.insert_player_stats(p)

        click.echo(f"Ingested {len(results)} matches for SSN {season}.")
    finally:
        await client.close()


@main.command()
@click.option("--file", "tar_file", required=True, type=click.Path(exists=True), help="Betfair TAR file path")
@click.option("--db", default=DEFAULT_DB, help="Database path")
def odds(tar_file: str, db: str):
    """Import Betfair historical odds from a TAR file."""
    db_conn = Database(db)
    db_conn.initialize()

    parser = BetfairParser()
    click.echo(f"Parsing {tar_file}...")
    all_odds = parser.parse_tar(tar_file)

    imported = 0
    for record in all_odds:
        odds_record = {
            "match_id": f"bf_{record['home_team']}_{record['event_date'][:10]}",
            "source": "betfair",
            "home_back_odds": record["home_back_odds"],
            "home_lay_odds": record["home_lay_odds"],
            "away_back_odds": record["away_back_odds"],
            "away_lay_odds": record["away_lay_odds"],
            "home_volume": record["home_volume"],
            "away_volume": record["away_volume"],
            "timestamp": record["timestamp"],
        }
        db_conn.upsert_odds(odds_record)
        imported += 1

    click.echo(f"Imported {imported} odds records.")


@main.command()
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--output", default="data/model.pkl", help="Model output path")
def train(db: str, output: str):
    """Backfill Elo, build features, train model."""
    db_conn = Database(db)
    matches = db_conn.get_matches()

    if len(matches) < 20:
        click.echo(f"Only {len(matches)} matches in DB. Need at least 20 to train.")
        return

    click.echo(f"Building features from {len(matches)} matches...")
    builder = FeatureBuilder(matches)
    df = builder.build_matrix(start_index=1)

    click.echo(f"Training on {len(df)} rows, {len(df.columns)} features...")
    model = NetballModel()
    model.train(df)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    click.echo(f"Model saved to {output}")

    # Print training summary
    preds = model.predict(df)
    mae = (df["margin"] - preds["predicted_margin"].astype(float)).abs().mean()
    click.echo(f"Training MAE: {mae:.1f} goals")


@main.command()
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--model-path", default="data/model.pkl", help="Model file path")
@click.option("--min-edge", default=0.05, type=float, help="Minimum edge threshold")
def predict(db: str, model_path: str, min_edge: float):
    """Predict upcoming matches and flag value bets."""
    import pandas as pd

    db_conn = Database(db)
    model = NetballModel.load(model_path)
    detector = ValueDetector(min_edge=min_edge)

    matches = db_conn.get_matches()
    # Find matches without scores (upcoming)
    upcoming = [m for m in matches if m.get("home_score") is None]

    if not upcoming:
        click.echo("No upcoming matches found. Run ingest first.")
        return

    builder = FeatureBuilder(matches)
    results = []

    for i, m in enumerate(matches):
        if m["match_id"] not in {u["match_id"] for u in upcoming}:
            continue

        row = builder.build_row(i)
        pred = model.predict(pd.DataFrame([row]))

        value = detector.evaluate(
            home_team=m["home_team"],
            away_team=m["away_team"],
            model_win_prob=float(pred["win_probability"].iloc[0]),
        )

        results.append({
            **value,
            "predicted_margin": float(pred["predicted_margin"].iloc[0]),
            "predicted_total": float(pred["predicted_total"].iloc[0]),
            "win_probability": float(pred["win_probability"].iloc[0]),
        })

    display_predictions(results)


@main.command()
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--train-seasons", required=True, help="Training seasons range (e.g. 2017-2023)")
@click.option("--test-season", required=True, type=int, help="Test season year")
def backtest(db: str, train_seasons: str, test_season: int):
    """Walk-forward backtest on a held-out season."""
    import numpy as np
    import pandas as pd

    db_conn = Database(db)
    all_matches = db_conn.get_matches()

    start, end = map(int, train_seasons.split("-"))
    train_matches = [m for m in all_matches if start <= m["season"] <= end]
    test_matches = [m for m in all_matches if m["season"] == test_season]

    if not train_matches or not test_matches:
        click.echo("Insufficient data for backtest.")
        return

    click.echo(f"Training on {len(train_matches)} matches ({train_seasons})")
    click.echo(f"Testing on {len(test_matches)} matches ({test_season})")

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

    mae = np.mean(abs_errors)
    accuracy = correct / total if total > 0 else 0

    click.echo(f"\nBacktest Results ({test_season}):")
    click.echo(f"  Matches: {total}")
    click.echo(f"  Win/Loss Accuracy: {accuracy:.1%}")
    click.echo(f"  Mean Absolute Error: {mae:.1f} goals")
