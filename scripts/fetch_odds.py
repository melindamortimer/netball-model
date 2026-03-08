#!/usr/bin/env python
"""Fetch SSN odds from BetsAPI for hardcoded event IDs.

Event IDs are extracted from https://betsapi.com/netball/l/23816/super-netball
(copy the HTML table manually since the site blocks automated scraping).

For each event, fetches odds via:
    https://api.b365api.com/v2/event/odds?token=TOKEN&event_id=ID

Usage:
    poetry run python scripts/fetch_odds.py
    poetry run python scripts/fetch_odds.py --dry-run
    poetry run python scripts/fetch_odds.py --season 2025
    poetry run python scripts/fetch_odds.py --from-cache data/betsapi_raw
"""
import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from netball_model.data.betsapi import BetsApiClient, _extract_best_closing_odds
from netball_model.data.database import Database

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from ssn_events import SSN_EVENTS


def _load_from_cache(cache_dir: str, events: list[dict]) -> list[dict]:
    """Read odds from previously saved JSON API responses."""
    results = []
    for ev in events:
        eid = ev["event_id"]
        path = os.path.join(cache_dir, f"{eid}.json")
        if not os.path.exists(path):
            logger.warning("No cache file for event %s", eid)
            results.append({**ev, "home_odds": None, "away_odds": None, "bookmaker": None})
            continue

        with open(path) as f:
            data = json.load(f)

        odds_data = data.get("results", {}).get("odds", {})
        best = None
        for market_key in ("147_1", "147_2", "1_1", "1_2", "1_3"):
            market = odds_data.get(market_key)
            if not market:
                continue
            best = _extract_best_closing_odds(market)
            if best:
                break

        results.append({
            **ev,
            "home_odds": best["home_odds"] if best else None,
            "away_odds": best["away_odds"] if best else None,
            "bookmaker": best["bookmaker"] if best else None,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch SSN odds from BetsAPI")
    parser.add_argument("--db", default="data/netball.db", help="Database path")
    parser.add_argument("--season", type=int, default=None, help="Filter DB matches to this season")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't store in DB")
    parser.add_argument("--from-cache", default=None, metavar="DIR",
                        help="Read from cached JSON files instead of hitting the API (e.g. data/betsapi_raw)")
    parser.add_argument("--save-cache", default=None, metavar="DIR",
                        help="Save raw API responses to DIR (e.g. data/betsapi_raw)")
    args = parser.parse_args()

    db = Database(args.db)
    db.initialize()

    matches = db.get_matches(season=args.season) if args.season else db.get_matches()
    if not matches:
        label = f"season {args.season}" if args.season else "any season"
        logger.error("No matches in DB for %s. Run ingest first.", label)
        sys.exit(1)

    logger.info("Loaded %d matches from DB", len(matches))

    # Build match lookup: (home_team, away_team, date) -> match_id
    # Include date to disambiguate the same matchup across seasons.
    match_lookup: dict[tuple[str, str, str], str] = {}
    for m in matches:
        d = (m.get("date") or "")[:10]
        match_lookup[(m["home_team"], m["away_team"], d)] = m["match_id"]

    if args.from_cache:
        # Read from cached JSON files
        logger.info("Reading odds from cache dir: %s", args.from_cache)
        events_with_odds = _load_from_cache(args.from_cache, SSN_EVENTS)
    else:
        token = os.environ.get("BETSAPI_TOKEN", "")
        if not token:
            logger.error("BETSAPI_TOKEN not set in .env or environment")
            sys.exit(1)

        cache_dir = args.save_cache
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Fetch odds, saving raw responses and skipping already-cached events
        logger.info("Fetching odds for %d events...", len(SSN_EVENTS))
        events_to_fetch = []
        cached_events = []
        for ev in SSN_EVENTS:
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"{ev['event_id']}.json")
                if os.path.exists(cache_path):
                    cached_events.append(ev)
                    continue
            events_to_fetch.append(ev)

        if cached_events:
            logger.info("Skipping %d already-cached events", len(cached_events))

        # Fetch uncached events with raw response saving
        events_with_odds = _load_from_cache(cache_dir, cached_events) if cached_events and cache_dir else []

        if events_to_fetch:
            with BetsApiClient(token) as client:
                for i, ev in enumerate(events_to_fetch):
                    eid = ev["event_id"]
                    try:
                        raw_data = client._get("/v2/event/odds", event_id=eid)
                    except Exception as e:
                        logger.warning("Failed to fetch event %s: %s", eid, e)
                        events_with_odds.append({**ev, "home_odds": None, "away_odds": None, "bookmaker": None})
                        continue

                    if cache_dir:
                        with open(os.path.join(cache_dir, f"{eid}.json"), "w") as f:
                            json.dump(raw_data, f)

                    # Extract odds from raw response
                    odds_data = raw_data.get("results", {}).get("odds", {})
                    best = None
                    for market_key in ("147_1", "147_2", "1_1", "1_2", "1_3"):
                        market = odds_data.get(market_key)
                        if not market:
                            continue
                        best = _extract_best_closing_odds(market)
                        if best:
                            break

                    events_with_odds.append({
                        **ev,
                        "home_odds": best["home_odds"] if best else None,
                        "away_odds": best["away_odds"] if best else None,
                        "bookmaker": best["bookmaker"] if best else None,
                    })

                    if (i + 1) % 20 == 0:
                        logger.info("Fetched %d/%d events", i + 1, len(events_to_fetch))
                    import time
                    time.sleep(0.5)

            logger.info("Fetched %d events total", len(events_with_odds))

    # Match to DB and store
    matched = 0
    unmatched = 0
    no_odds = 0
    odds_records: list[dict] = []

    for ev in events_with_odds:
        if ev["home_odds"] is None:
            no_odds += 1
            logger.debug("No odds: event %s (%s v %s)", ev["event_id"], ev["home_team"], ev["away_team"])
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
            unmatched += 1
            logger.warning("No DB match: %s vs %s (event %s)", home, away, ev["event_id"])
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
        logger.info("  %s vs %s -> %.2f / %.2f", home, away, home_odds, away_odds)

    # Summary
    print()
    print("=" * 60)
    print("FETCH ODDS SUMMARY")
    print("=" * 60)
    print(f"  Events total:      {len(SSN_EVENTS)}")
    print(f"  With odds:         {len(SSN_EVENTS) - no_odds}")
    print(f"  Matched to DB:     {matched}")
    print(f"  Unmatched:         {unmatched}")
    print(f"  No odds available: {no_odds}")

    if args.dry_run:
        print("\n  [DRY RUN] No data stored.")
    elif odds_records:
        db.upsert_odds_batch(odds_records)
        print(f"\n  Stored {len(odds_records)} odds records in {args.db}")
    else:
        print("\n  No odds to store.")


if __name__ == "__main__":
    main()
