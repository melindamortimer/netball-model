import asyncio
import os

import click
from dotenv import load_dotenv

load_dotenv()

from netball_model.data.betfair import BetfairParser
from netball_model.data.database import Database
from netball_model.features.builder import FeatureBuilder
from netball_model.model.train import NetballModel
from netball_model.display import display_predictions
from netball_model.value.detector import ValueDetector
from netball_model.services import ingest_season, train_model, backtest_season, import_betsapi_odds

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


@main.command("fetch-odds")
@click.option("--season", default=None, type=int, help="SSN season year (e.g. 2024). If omitted, matches all seasons.")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--token", envvar="BETSAPI_TOKEN", default=None, help="BetsAPI token (or set BETSAPI_TOKEN env var)")
def fetch_odds(season: int | None, db: str, token: str | None):
    """Fetch pre-match odds from BetsAPI for hardcoded event IDs.

    Use scripts/fetch_odds.py for the full standalone workflow.
    """
    if not token:
        click.echo("Error: BetsAPI token required. Pass --token or set BETSAPI_TOKEN.")
        raise SystemExit(1)

    # Import hardcoded events
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ssn_events", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "ssn_events.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SSN_EVENTS = mod.SSN_EVENTS

    db_conn = Database(db)
    label = f"SSN {season}" if season else "all seasons"
    click.echo(f"Fetching BetsAPI odds for {len(SSN_EVENTS)} events ({label})...")

    try:
        counts = import_betsapi_odds(db_conn, token, SSN_EVENTS, season)
    except ValueError as e:
        click.echo(str(e))
        return

    click.echo(
        f"Done: {counts['total']} events, "
        f"{counts['matched']} matched, "
        f"{counts['unmatched']} unmatched, "
        f"{counts['no_odds']} without odds."
    )


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


@main.command("scrape-odds")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--headless", is_flag=True, help="Run browser in headless mode")
def scrape_odds(db: str, headless: bool):
    """Scrape pre-match odds from bet365 for upcoming SSN matches."""
    from datetime import datetime, timezone
    from netball_model.data.bet365 import Bet365Scraper
    from netball_model.data.team_names import normalise_team

    db_conn = Database(db)
    db_conn.initialize()

    matches = db_conn.get_matches()
    upcoming = [m for m in matches if m.get("home_score") is None]

    if not upcoming:
        click.echo("No upcoming matches in DB. Run `netball ingest` first.")
        return

    # Build lookup: (home_team, away_team, date[:10]) -> match_id
    match_lookup: dict[tuple[str, str, str], str] = {}
    for m in upcoming:
        d = (m.get("date") or "")[:10]
        match_lookup[(m["home_team"], m["away_team"], d)] = m["match_id"]

    click.echo("Launching bet365 scraper...")
    scraper = Bet365Scraper(headless=headless)
    scraped = scraper.scrape_ssn_odds()

    if not scraped:
        click.echo("No matches scraped. Check the browser output for errors.")
        return

    click.echo(f"Scraped {len(scraped)} matches from bet365.")

    matched = 0
    unmatched = 0
    now = datetime.now(timezone.utc).isoformat()

    for s in scraped:
        home = s.get("home_team") or ""
        away = s.get("away_team") or ""
        date = s.get("match_date", "")

        match_id = match_lookup.get((home, away, date))
        if not match_id:
            # Try swapped
            match_id = match_lookup.get((away, home, date))
            if match_id:
                s["home_odds"], s["away_odds"] = s.get("away_odds"), s.get("home_odds")
                s["handicap_home_odds"], s["handicap_away_odds"] = s.get("handicap_away_odds"), s.get("handicap_home_odds")
                if s.get("handicap_line") is not None:
                    s["handicap_line"] = -s["handicap_line"]

        if not match_id:
            click.echo(f"  No DB match for {home} vs {away} ({date})")
            unmatched += 1
            continue

        odds_record = {
            "match_id": match_id,
            "source": "bet365",
            "home_back_odds": s.get("home_odds"),
            "home_lay_odds": None,
            "away_back_odds": s.get("away_odds"),
            "away_lay_odds": None,
            "home_volume": 0,
            "away_volume": 0,
            "timestamp": now,
            "handicap_home_odds": s.get("handicap_home_odds"),
            "handicap_line": s.get("handicap_line"),
            "handicap_away_odds": s.get("handicap_away_odds"),
            "total_line": s.get("total_line"),
            "over_odds": s.get("over_odds"),
            "under_odds": s.get("under_odds"),
        }
        db_conn.upsert_odds_extended(odds_record)
        matched += 1

    click.echo(f"Done: {matched} matched, {unmatched} unmatched.")


@main.command()
@click.option("--season", required=True, type=int, help="SSN season year")
@click.option("--db", default=DEFAULT_DB, help="Database path")
@click.option("--output", default="data/model.pkl", help="Model output path")
def update(season: int, db: str, output: str):
    """Ingest latest results from Champion Data and retrain model."""
    db_conn = Database(db)

    click.echo(f"Ingesting SSN {season} data...")
    count = asyncio.run(ingest_season(db_conn, season))
    click.echo(f"Ingested {count} matches.")

    click.echo("Retraining model...")
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

        # Load stored odds from DB if available
        stored_odds = db_conn.get_odds_for_match(m["match_id"])
        home_odds = stored_odds["home_back_odds"] if stored_odds else None
        away_odds = stored_odds["away_back_odds"] if stored_odds else None

        value = detector.evaluate(
            home_team=m["home_team"],
            away_team=m["away_team"],
            model_win_prob=float(pred["win_probability"].iloc[0]),
            home_odds=home_odds,
            away_odds=away_odds,
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
