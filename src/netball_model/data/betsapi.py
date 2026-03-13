"""BetsAPI client for fetching pre-match odds via the b365api.com REST API.

Endpoint used:
- GET /v2/event/odds — pre-match odds per event
"""
from __future__ import annotations

import logging
import os
import ssl
import time

import certifi
import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.b365api.com"

# Super Netball league_id on BetsAPI.
SSN_LEAGUE_ID = 23816

from netball_model.data.team_names import TEAM_NAME_MAP, normalise_team  # noqa: F401


class BetsApiClient:
    """Client for the b365api.com REST API."""

    def __init__(self, token: str, *, timeout: float = 30.0):
        self.token = token
        ca_bundle = os.environ.get("SSL_CERT_FILE") or certifi.where()
        ssl_ctx = ssl.create_default_context(cafile=ca_bundle)
        self._client = httpx.Client(
            base_url=API_BASE_URL,
            timeout=timeout,
            params={"token": token},
            verify=ssl_ctx,
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _get(self, path: str, **params) -> dict:
        """Make a GET request with retry on 429/5xx errors."""
        for attempt in range(3):
            resp = self._client.get(path, params=params)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning("Rate limited, sleeping %ds...", wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 2 * (attempt + 1)
                logger.warning("Server error %d, retrying in %ds...", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def fetch_event_odds(self, event_id: str | int) -> dict | None:
        """Fetch pre-match odds for a single event.

        Returns a dict with home/away decimal odds from available bookmakers,
        or None if no odds are available.
        """
        data = self._get("/v2/event/odds", event_id=event_id)
        if data.get("success") != 1:
            return None

        results = data.get("results", {})
        odds_data = results.get("odds", {})

        # BetsAPI netball market keys:
        #   147_1 = match winner (head-to-head)
        #   147_2 = handicap/line
        #   147_3 = totals (over_od/under_od — different fields, skip)
        # Also try legacy 1_1, 1_2, 1_3 keys as fallback.
        for market_key in ("147_1", "147_2", "1_1", "1_2", "1_3"):
            market = odds_data.get(market_key)
            if not market:
                continue
            best = _extract_best_closing_odds(market)
            if best:
                return best

        return None

    def fetch_odds_for_events(
        self,
        events: list[dict],
        *,
        delay: float = 0.5,
    ) -> list[dict]:
        """Fetch odds for a list of events.

        Each event dict must have: event_id, home_team, away_team.
        Returns the same dicts enriched with: home_odds, away_odds, bookmaker.
        """
        results: list[dict] = []
        for i, ev in enumerate(events):
            event_id = ev["event_id"]
            odds = self.fetch_event_odds(event_id)

            results.append({
                **ev,
                "home_odds": odds["home_odds"] if odds else None,
                "away_odds": odds["away_odds"] if odds else None,
                "bookmaker": odds["bookmaker"] if odds else None,
            })

            if (i + 1) % 20 == 0:
                logger.info("Fetched odds for %d/%d events", i + 1, len(events))
            if delay > 0:
                time.sleep(delay)

        logger.info("Fetched odds for %d events", len(results))
        return results


def _extract_best_closing_odds(market: list[dict]) -> dict | None:
    """From a list of bookmaker odds snapshots, pick the closing pre-match odds.

    BetsAPI returns odds entries with:
        home_od, away_od, add_time, bookmaker_id, ss (live score if in-play)
    We filter to pre-match only (ss is absent or falsy) and pick the latest
    snapshot (highest add_time) as the closing pre-match price.
    """
    if not market:
        return None

    best: dict | None = None
    best_time = 0

    for entry in market:
        # Skip live/in-play entries (have a score in 'ss')
        if entry.get("ss"):
            continue
        t = int(entry.get("add_time", 0))
        home_od = entry.get("home_od")
        away_od = entry.get("away_od")
        if home_od is None or away_od is None:
            continue
        try:
            home_f = float(home_od)
            away_f = float(away_od)
        except (ValueError, TypeError):
            continue
        if home_f <= 1.0 or away_f <= 1.0:
            continue
        if t >= best_time:
            best_time = t
            best = {
                "home_odds": home_f,
                "away_odds": away_f,
                "bookmaker": entry.get("bookmaker_id", ""),
            }

    return best
