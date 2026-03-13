"""Playwright-based scraper for bet365 Australia Super Netball odds.

Navigates bet365.com.au, finds the Super Netball section, and extracts
head-to-head, handicap, and total goals odds from individual match pages.

Usage:
    scraper = Bet365Scraper(headless=False)
    odds = scraper.scrape_ssn_odds()
"""
from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime

from playwright.sync_api import Page, sync_playwright

from netball_model.data.team_names import normalise_team

logger = logging.getLogger(__name__)

# bet365 Australia URL.
BET365_URL = "https://www.bet365.com.au"

# Realistic browser viewport.
VIEWPORT = {"width": 1280, "height": 800}


def _random_delay(lo: float = 2.0, hi: float = 5.0) -> None:
    """Sleep for a random duration between *lo* and *hi* seconds."""
    time.sleep(random.uniform(lo, hi))


def _parse_odds(text: str | None) -> float | None:
    """Parse a decimal odds string like '1.38' into a float, or None."""
    if not text:
        return None
    text = text.strip()
    try:
        value = float(text)
        # Sanity check — decimal odds must be > 1.0.
        return value if value > 1.0 else None
    except (ValueError, TypeError):
        return None


def _parse_line(text: str | None) -> float | None:
    """Parse a handicap/total line string like '-4.5' or '125.5' into a float."""
    if not text:
        return None
    text = text.strip()
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _parse_section_text(text: str) -> dict:
    """Parse the inner text of a bet365 table section into odds values.

    Expects text with lines like:
        To Win    1.30    3.50
        Total     O 125.5 1.87    U 125.5 1.87
        Line      -5.5 1.85       +5.5 1.95

    Returns dict with 8 keys, all defaulting to None if not found:
        home_odds, away_odds, total_line, over_odds, under_odds,
        handicap_line, handicap_home_odds, handicap_away_odds
    """
    result: dict = {
        "home_odds": None,
        "away_odds": None,
        "total_line": None,
        "over_odds": None,
        "under_odds": None,
        "handicap_line": None,
        "handicap_home_odds": None,
        "handicap_away_odds": None,
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # --- To Win row ---
        if line.startswith("To Win"):
            odds = re.findall(r"(\d+\.\d{2})", line)
            if len(odds) >= 2:
                result["home_odds"] = float(odds[0])
                result["away_odds"] = float(odds[1])

        # --- Total row ---
        elif line.startswith("Total"):
            over_m = re.search(r"O\s+(\d+\.?\d*)\s+(\d+\.\d{2})", line)
            under_m = re.search(r"U\s+(\d+\.?\d*)\s+(\d+\.\d{2})", line)
            if over_m:
                result["total_line"] = float(over_m.group(1))
                result["over_odds"] = float(over_m.group(2))
            if under_m:
                result["under_odds"] = float(under_m.group(2))

        # --- Line row (but NOT "Match Lines" header) ---
        elif re.match(r"^Line\b", line):
            signed = re.findall(r"([+-]\d+\.?\d*)\s+(\d+\.\d{2})", line)
            if len(signed) >= 2:
                result["handicap_line"] = float(signed[0][0])
                result["handicap_home_odds"] = float(signed[0][1])
                result["handicap_away_odds"] = float(signed[1][1])
            elif len(signed) == 1:
                result["handicap_line"] = float(signed[0][0])
                result["handicap_home_odds"] = float(signed[0][1])

    return result


class Bet365Scraper:
    """Scrape Super Netball odds from bet365 Australia.

    Uses Playwright with stealth mode to navigate the live website.
    The DOM structure uses obfuscated class names, so selectors rely
    on text content matching where possible. Expect to tune selectors
    after the first real run — bet365 changes its markup frequently.

    Args:
        headless: Run the browser without a visible window. Default False
            so you can watch the navigation and debug selectors.
        timeout: Default timeout in milliseconds for waiting on elements.
    """

    def __init__(self, *, headless: bool = False, timeout: int = 15_000):
        self.headless = headless
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_ssn_odds(self) -> list[dict]:
        """Main entry point. Launch browser, navigate to SSN, scrape odds.

        Returns:
            List of match odds dicts with full-match, 1st half, and 1st quarter odds.
        """
        results: list[dict] = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless)
            context = browser.new_context(
                viewport=VIEWPORT,
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                locale="en-AU",
                timezone_id="Australia/Melbourne",
            )
            page = context.new_page()
            page.set_default_timeout(self.timeout)

            # Apply stealth patches.
            try:
                from playwright_stealth import stealth_sync  # type: ignore[import-untyped]

                stealth_sync(page)
            except ImportError:
                logger.warning(
                    "playwright_stealth not installed — running without stealth"
                )

            try:
                self._navigate_to_ssn(page)
                match_labels = self._collect_match_links(page)
                logger.info("Found %d match(es) on the SSN page", len(match_labels))

                for idx, label in enumerate(match_labels):
                    logger.info(
                        "Scraping match %d/%d: %s", idx + 1, len(match_labels), label
                    )
                    try:
                        match_data = self._scrape_single_match(page, label)
                        if match_data:
                            results.append(match_data)
                    except Exception:
                        logger.warning(
                            "Failed to scrape match %s — skipping",
                            label,
                            exc_info=True,
                        )
                    _random_delay(2.0, 4.0)
            finally:
                browser.close()

        logger.info("Scraped %d match(es) total", len(results))
        return results

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _ensure_ssn_expanded(self, page: Page) -> None:
        """Expand the Super Netball dropdown and Matches sub-section if collapsed.

        Called during initial navigation and after each page.go_back()
        since the dropdown may re-collapse. Safe to call when already expanded.
        """
        # Check "Super Netball" is visible.
        ssn_header = page.get_by_text("Super Netball", exact=True).first
        if not ssn_header or not ssn_header.is_visible(timeout=3000):
            raise RuntimeError(
                "Could not find 'Super Netball' on the Netball page. "
                "The site structure may have changed."
            )

        # Check if the SSN dropdown is expanded by looking for the "Matches"
        # sub-section text. If not visible, click SSN header to expand.
        matches_link = page.get_by_text("Matches", exact=True).first
        if not matches_link or not matches_link.is_visible(timeout=2000):
            ssn_header.click()
            _random_delay(1.5, 3.0)

        # Now check if the match list is visible (contains "Team v Team" text).
        # If not, click "Matches" to expand it. If match links are already
        # visible, skip the click to avoid toggling it closed.
        match_text = page.locator("text=/\\w.+ v \\w.+/i").first
        already_has_matches = False
        try:
            already_has_matches = match_text.is_visible(timeout=1500)
        except Exception:
            pass

        if not already_has_matches:
            matches_link = page.get_by_text("Matches", exact=True).first
            if not matches_link or not matches_link.is_visible(timeout=3000):
                raise RuntimeError(
                    "Could not find 'Matches' sub-section under Super Netball."
                )
            matches_link.click()
            _random_delay(1.5, 3.0)

    def _navigate_to_ssn(self, page: Page) -> None:
        """Navigate from the bet365 homepage to the Super Netball section."""

        # 1. Load the homepage.
        logger.info("Loading bet365.com.au ...")
        page.goto(BET365_URL, wait_until="domcontentloaded")
        _random_delay(3.0, 5.0)
        self._dismiss_popups(page)

        # 2. Click Netball in the left sidebar.
        logger.info("Looking for 'Netball' in the sports menu ...")
        netball_link = page.get_by_text("Netball", exact=True).first
        netball_link.click()
        _random_delay(25.0, 30.0)

        # 3. Expand the Super Netball dropdown and Matches section.
        self._ensure_ssn_expanded(page)

    def _dismiss_popups(self, page: Page) -> None:
        """Attempt to close common bet365 popups/overlays.

        bet365 may show location verification, cookie consent, or
        promotional overlays. We try a few common dismiss patterns.
        """
        # Close/Accept buttons that may appear in overlays.
        for selector in [
            "text='Accept All'",
            "text='Accept'",
            "text='OK'",
            "text='Continue'",
            "text='Close'",
            "[aria-label='Close']",
        ]:
            try:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=1000):
                    btn.click()
                    _random_delay(0.5, 1.0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Match discovery
    # ------------------------------------------------------------------

    def _collect_match_links(self, page: Page) -> list[str]:
        """Find match labels in the expanded Super Netball section.

        Returns a list of match label strings like "Adelaide Thunderbirds v NSW Swifts".
        These are used to re-locate clickable elements after page navigation.
        """
        page.wait_for_timeout(2000)

        # Find all text elements matching "Team A v Team B" pattern.
        match_elements = page.locator("text=/\\w.+\\s+v\\s+\\w.+/i")
        labels: list[str] = []
        seen: set[str] = set()

        count = match_elements.count()
        for i in range(count):
            try:
                text = (match_elements.nth(i).inner_text(timeout=1000) or "").strip()
            except Exception:
                continue

            # Must contain " v " and both sides should normalise to known teams.
            if " v " not in text:
                continue

            parts = text.split(" v ", 1)
            home = normalise_team(parts[0].strip())
            away = normalise_team(parts[1].strip())
            if home and away and text not in seen:
                seen.add(text)
                labels.append(text)

        return labels

    # ------------------------------------------------------------------
    # Single match scraping
    # ------------------------------------------------------------------

    def _scrape_single_match(self, page: Page, label: str) -> dict | None:
        """Click into a match page, extract odds, then go back.

        Args:
            page: The active Playwright page on the SSN coupon.
            label: Match label text like "Adelaide Thunderbirds v NSW Swifts".

        Returns:
            Odds dict or None if extraction failed entirely.
        """
        # Locate and click the match link by its text label.
        match_link = page.get_by_text(label, exact=True).first
        match_link.click()
        _random_delay(2.0, 4.0)

        try:
            match_data = self._extract_match_odds(page)
        except Exception:
            logger.warning("Odds extraction failed for %s", label, exc_info=True)
            match_data = None

        # Navigate back to the SSN coupon.
        page.go_back()
        _random_delay(1.5, 3.0)
        self._dismiss_popups(page)

        # Re-expand the SSN dropdown (may have collapsed after go_back).
        try:
            self._ensure_ssn_expanded(page)
        except Exception:
            logger.warning("Failed to re-expand SSN section after going back", exc_info=True)

        return match_data

    def _find_table_section(self, page: Page, header_text: str):
        """Find a table section by its header text and return the container locator.

        Walks up ancestor divs from the header to find the one containing
        odds-like decimal numbers.

        Returns:
            Playwright locator for the section container, or None.
        """
        try:
            header = page.get_by_text(header_text, exact=True).first
            if not header or not header.is_visible(timeout=2000):
                return None
            for depth in range(2, 6):
                try:
                    container = header.locator(f"xpath=ancestor::div[{depth}]").first
                    container_text = container.inner_text(timeout=1000) or ""
                    if re.search(r"\b\d+\.\d{2}\b", container_text):
                        return container
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _extract_match_odds(self, page: Page) -> dict | None:
        """Extract odds from all three table sections on the match page.

        Finds "Match Lines", "1st Half", and "1st Quarter" sections,
        parses each, and merges into a single result dict with prefix mapping.

        Returns:
            26-key odds dict, or None if team names couldn't be identified.
        """
        page.wait_for_timeout(3000)

        home_team, away_team = self._extract_team_names(page)
        if not home_team or not away_team:
            logger.warning("Could not identify team names on match page")
            return None

        result: dict = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": self._extract_match_date(page),
        }

        # Sections to extract: (header_text, key_prefix)
        sections = [
            ("Match Lines", ""),
            ("1st Half", "h1_"),
            ("1st Quarter", "q1_"),
        ]

        for header, prefix in sections:
            section = self._find_table_section(page, header)
            if section:
                try:
                    text = section.inner_text(timeout=3000) or ""
                    parsed = _parse_section_text(text)
                    result.update({f"{prefix}{k}": v for k, v in parsed.items()})
                except Exception:
                    logger.debug("Failed to parse section '%s'", header, exc_info=True)
                    # Fill with None for this section.
                    for k in _parse_section_text(""):
                        result.setdefault(f"{prefix}{k}", None)
            else:
                logger.debug("Section '%s' not found on match page", header)
                for k in _parse_section_text(""):
                    result.setdefault(f"{prefix}{k}", None)

        return result

    # ------------------------------------------------------------------
    # Team names and date
    # ------------------------------------------------------------------

    def _extract_team_names(self, page: Page) -> tuple[str | None, str | None]:
        """Extract and normalise home/away team names from the match header.

        Looks for "Team A v Team B" text in the page header.
        """
        try:
            header = page.locator("text=/\\w.+\\s+v\\s+\\w.+/i").first
            header_text = header.inner_text(timeout=3000)
            if " v " in header_text:
                parts = header_text.split(" v ", 1)
                home = normalise_team(parts[0].strip())
                away = normalise_team(parts[1].strip())
                if home and away:
                    return home, away
        except Exception:
            pass
        return None, None

    def _extract_match_date(self, page: Page) -> str | None:
        """Try to extract the match date from the page.

        bet365 shows dates in various formats near the match header.
        Returns ISO date string (YYYY-MM-DD) or None.
        """
        # Look for date-like text near the top of the page.
        # Common formats: "Friday 14 March 2026", "14/03/2026", "14 Mar".
        try:
            body_text = page.locator("body").inner_text(timeout=3000)
            # Try DD/MM/YYYY.
            m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", body_text)
            if m:
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return f"{year:04d}-{month:02d}-{day:02d}"

            # Try "DD Month YYYY" or "Day DD Month YYYY".
            m = re.search(
                r"(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
                r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
                r"Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})",
                body_text,
                re.IGNORECASE,
            )
            if m:
                day_str, month_str, year_str = m.group(1), m.group(2), m.group(3)
                dt = datetime.strptime(f"{day_str} {month_str} {year_str}", "%d %B %Y")
                return dt.strftime("%Y-%m-%d")

            # Try abbreviated month: "14 Mar"
            m = re.search(
                r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
                body_text,
                re.IGNORECASE,
            )
            if m:
                day_str, month_str = m.group(1), m.group(2)
                # Assume current year.
                year = datetime.now().year
                dt = datetime.strptime(f"{day_str} {month_str} {year}", "%d %b %Y")
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

        return None
