"""Tests for bet365 scraper module — import and structure only (no live scraping)."""
from netball_model.data.bet365 import Bet365Scraper


def test_scraper_instantiates():
    scraper = Bet365Scraper(headless=True)
    assert scraper.headless is True
    assert scraper.timeout == 15000


def test_scraper_custom_timeout():
    scraper = Bet365Scraper(headless=True, timeout=30000)
    assert scraper.timeout == 30000
