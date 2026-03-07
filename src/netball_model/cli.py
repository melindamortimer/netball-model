import asyncio

import click

from netball_model.data.betfair import BetfairParser
from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel
from netball_model.display import display_predictions
from netball_model.value.detector import ValueDetector
from netball_model.services import ingest_season, train_model, backtest_season

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
    click.echo(f"Fetching SSN {season} data from Champion Data...")
    count = asyncio.run(ingest_season(Database(db), season))
    click.echo(f"Ingested {count} matches for SSN {season}.")


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

    click.echo(f"Building features from {len(matches)} matches...")

    try:
        model, mae = train_model(db_conn, output)
    except ValueError as e:
        click.echo(str(e))
        return

    click.echo(f"Model saved to {output}")
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
    upcoming = [m for m in matches if m.get("home_score") is None]

    if not upcoming:
        click.echo("No upcoming matches found. Run ingest first.")
        return

    builder = FeatureBuilder(matches)
    upcoming_ids = {u["match_id"] for u in upcoming}
    results = []

    for i, m in enumerate(matches):
        if m["match_id"] not in upcoming_ids:
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
    db_conn = Database(db)

    start, end = map(int, train_seasons.split("-"))

    click.echo(f"Training on seasons {train_seasons}, testing on {test_season}...")

    try:
        results = backtest_season(db_conn, (start, end), test_season)
    except ValueError as e:
        click.echo(str(e))
        return

    click.echo(f"\nBacktest Results ({test_season}):")
    click.echo(f"  Matches: {results['matches']}")
    click.echo(f"  Win/Loss Accuracy: {results['accuracy']:.1%}")
    click.echo(f"  Mean Absolute Error: {results['mae']:.1f} goals")
