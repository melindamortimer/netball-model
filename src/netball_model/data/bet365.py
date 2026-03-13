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
            List of match odds dicts (see module docstring for schema).
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

            # Apply stealth patches to avoid automation detection.
            try:
                from playwright_stealth import stealth_sync  # type: ignore[import-untyped]

                stealth_sync(page)
            except ImportError:
                logger.warning(
                    "playwright_stealth not installed — running without stealth"
                )

            try:
                self._navigate_to_ssn(page)
                match_links = self._collect_match_links(page)
                logger.info("Found %d match link(s) on the SSN page", len(match_links))

                for idx, link_info in enumerate(match_links):
                    logger.info(
                        "Scraping match %d/%d: %s",
                        idx + 1,
                        len(match_links),
                        link_info.get("label", "?"),
                    )
                    try:
                        match_data = self._scrape_single_match(page, link_info)
                        if match_data:
                            results.append(match_data)
                    except Exception:
                        logger.warning(
                            "Failed to scrape match %s — skipping",
                            link_info.get("label", "?"),
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

    def _navigate_to_ssn(self, page: Page) -> None:
        """Navigate from the bet365 homepage to the Super Netball coupon."""

        # 1. Load the homepage.
        logger.info("Loading bet365.com.au ...")
        page.goto(BET365_URL, wait_until="domcontentloaded")
        _random_delay(3.0, 5.0)

        # Handle potential welcome/location popup overlays.
        self._dismiss_popups(page)

        # 2. Click into the Netball sport category.
        # bet365 uses a left-hand sport menu with text labels.
        logger.info("Looking for 'Netball' in the sports menu ...")
        netball_link = page.get_by_text("Netball", exact=True).first
        netball_link.click()
        _random_delay(2.0, 4.0)

        # 3. Click into the Super Netball competition.
        # The competition name varies — try common variants.
        logger.info("Looking for Super Netball competition ...")
        ssn_clicked = False
        for label in [
            "Suncorp Super Netball",
            "Super Netball",
            "SSN",
            "Australian Super Netball",
        ]:
            ssn_link = page.get_by_text(label, exact=False).first
            if ssn_link and ssn_link.is_visible():
                ssn_link.click()
                ssn_clicked = True
                logger.info("Clicked competition: %s", label)
                break

        if not ssn_clicked:
            raise RuntimeError(
                "Could not find Super Netball competition link on the Netball page. "
                "The site structure may have changed."
            )

        _random_delay(2.0, 4.0)
        self._dismiss_popups(page)

    def _dismiss_popups(self, page: Page) -> None:
        """Attempt to close common bet365 popups/overlays.

        bet365 may show location verification, cookie consent, or
        promotional overlays. We try a few common dismiss patterns.
        """
        # Close/Accept buttons that may appear in overlays.
        for selector in [
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

    def _collect_match_links(self, page: Page) -> list[dict]:
        """Identify clickable match entries on the SSN coupon page.

        Returns a list of dicts with:
            - locator: a Playwright locator for clicking into the match
            - label: a human-readable description for logging

        Strategy:
            bet365 lists matches as rows/containers. Each container has
            two team names and sometimes a date/time. We look for
            participant containers that hold two team-name elements.
        """
        # Wait for match content to render.
        page.wait_for_timeout(3000)

        # Approach: find all elements that look like match event containers.
        # bet365 typically wraps each fixture in a clickable div/a with
        # participant names inside. The outermost clickable ancestor is
        # what we need to click.
        #
        # We search for elements whose text matches known SSN team names.
        # Then group them into pairs (home, away) by shared parent.

        # Collect all visible text nodes that match SSN team names.
        team_elements = page.locator(
            # Participant name elements — bet365 often uses <span> or <div>
            # inside a participant row. We look broadly and filter.
            "div, span"
        )

        match_links: list[dict] = []
        seen_labels: set[str] = set()

        # Scan the page for known team names to find match containers.
        count = team_elements.count()
        for i in range(count):
            el = team_elements.nth(i)
            try:
                text = (el.inner_text(timeout=500) or "").strip()
            except Exception:
                continue

            normalised = normalise_team(text)
            if not normalised:
                continue

            # Walk up to find the match container (a clickable ancestor
            # that contains *two* team names). We use the participant's
            # grandparent or great-grandparent as a heuristic.
            # This is fragile — adjust depth as needed.
            for ancestor_css in [
                "xpath=ancestor::div[3]",
                "xpath=ancestor::div[4]",
                "xpath=ancestor::div[5]",
            ]:
                try:
                    container = el.locator(ancestor_css).first
                    container_text = (container.inner_text(timeout=500) or "")
                    # Check the container has at least two different team names.
                    teams_in_container = [
                        normalise_team(word)
                        for word in re.split(r"\n|\s{2,}", container_text)
                        if normalise_team(word)
                    ]
                    unique_teams = list(dict.fromkeys(teams_in_container))
                    if len(unique_teams) >= 2:
                        label = f"{unique_teams[0]} v {unique_teams[1]}"
                        if label not in seen_labels:
                            seen_labels.add(label)
                            match_links.append(
                                {"locator": container, "label": label}
                            )
                        break
                except Exception:
                    continue

        return match_links

    # ------------------------------------------------------------------
    # Single match scraping
    # ------------------------------------------------------------------

    def _scrape_single_match(self, page: Page, link_info: dict) -> dict | None:
        """Click into a match page, extract odds, then go back.

        Args:
            page: The active Playwright page on the SSN coupon.
            link_info: Dict with 'locator' (Playwright Locator) and 'label'.

        Returns:
            Odds dict or None if extraction failed entirely.
        """
        # Click into the match detail page.
        link_info["locator"].click()
        _random_delay(2.0, 4.0)

        try:
            match_data = self._extract_match_odds(page)
        except Exception:
            logger.warning("Odds extraction failed for %s", link_info["label"], exc_info=True)
            match_data = None

        # Navigate back to the SSN coupon.
        page.go_back()
        _random_delay(1.5, 3.0)
        self._dismiss_popups(page)

        return match_data

    def _extract_match_odds(self, page: Page) -> dict | None:
        """Extract H2H, handicap, and totals from the current match page.

        bet365 match pages show markets in collapsible sections. Each
        section has a header ("Full Time Result", "Handicap", etc.) and
        rows with team names + odds buttons.

        Returns:
            Odds dict matching the project schema, or None.
        """
        # Wait for the match page to load odds content.
        page.wait_for_timeout(3000)

        # --- Team names ---
        # The match header typically shows "Team A v Team B" or lists
        # both participant names at the top of the page.
        home_team, away_team = self._extract_team_names(page)
        if not home_team or not away_team:
            logger.warning("Could not identify team names on match page")
            return None

        result: dict = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": self._extract_match_date(page),
            "home_odds": None,
            "away_odds": None,
            "handicap_home_odds": None,
            "handicap_line": None,
            "handicap_away_odds": None,
            "total_line": None,
            "over_odds": None,
            "under_odds": None,
        }

        # --- Head-to-head (Full Time Result / Match Winner) ---
        h2h = self._extract_h2h(page)
        if h2h:
            result["home_odds"] = h2h.get("home_odds")
            result["away_odds"] = h2h.get("away_odds")

        # --- Handicap ---
        handicap = self._extract_handicap(page)
        if handicap:
            result["handicap_line"] = handicap.get("line")
            result["handicap_home_odds"] = handicap.get("home_odds")
            result["handicap_away_odds"] = handicap.get("away_odds")

        # --- Totals (Total Goals / Total Points) ---
        totals = self._extract_totals(page)
        if totals:
            result["total_line"] = totals.get("line")
            result["over_odds"] = totals.get("over_odds")
            result["under_odds"] = totals.get("under_odds")

        return result

    # ------------------------------------------------------------------
    # Team names and date
    # ------------------------------------------------------------------

    def _extract_team_names(self, page: Page) -> tuple[str | None, str | None]:
        """Extract and normalise home/away team names from the match header.

        Tries several approaches:
        1. Look for a header with "v" separating two names.
        2. Look for participant name elements at the top of the page.
        """
        # Approach 1: header text like "Melbourne Vixens v NSW Swifts".
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

        # Approach 2: scan top-of-page elements for two distinct team names.
        try:
            # bet365 often puts participant names in dedicated elements
            # near the top. We look at the first ~50 spans/divs.
            elements = page.locator("div, span")
            found_teams: list[str] = []
            limit = min(elements.count(), 80)
            for i in range(limit):
                try:
                    text = (elements.nth(i).inner_text(timeout=300) or "").strip()
                except Exception:
                    continue
                norm = normalise_team(text)
                if norm and norm not in found_teams:
                    found_teams.append(norm)
                if len(found_teams) == 2:
                    return found_teams[0], found_teams[1]
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

    # ------------------------------------------------------------------
    # Market extraction helpers
    # ------------------------------------------------------------------

    def _extract_h2h(self, page: Page) -> dict | None:
        """Extract head-to-head (Full Time Result / Match Winner) odds.

        Looks for a market section headed by one of the common H2H labels,
        then extracts two odds values (home, away) from odds buttons.

        Returns:
            {"home_odds": float, "away_odds": float} or None.
        """
        # Identify the H2H market section by its header text.
        for label in ["Full Time Result", "Match Winner", "Match Result", "Winner"]:
            section = self._find_market_section(page, label)
            if section:
                break
        else:
            # Fallback: grab the first two odds-like values on the page.
            # The first market on a bet365 match page is typically H2H.
            logger.debug("No H2H header found — trying first visible odds")
            section = None

        odds_values = self._extract_odds_from_section(page, section, expected_count=2)
        if odds_values and len(odds_values) >= 2:
            return {"home_odds": odds_values[0], "away_odds": odds_values[1]}

        return None

    def _extract_handicap(self, page: Page) -> dict | None:
        """Extract handicap/line market odds.

        The handicap section has a line value (e.g. -4.5) plus two odds.

        Returns:
            {"line": float, "home_odds": float, "away_odds": float} or None.
        """
        for label in ["Handicap", "Line", "Handicap Result", "Spread"]:
            section = self._find_market_section(page, label)
            if section:
                break
        else:
            return None

        # Extract the handicap line value.
        line = self._extract_line_from_section(section)

        # Extract odds (home handicap odds, away handicap odds).
        odds_values = self._extract_odds_from_section(page, section, expected_count=2)
        if odds_values and len(odds_values) >= 2:
            return {
                "line": line,
                "home_odds": odds_values[0],
                "away_odds": odds_values[1],
            }

        return None

    def _extract_totals(self, page: Page) -> dict | None:
        """Extract total goals/points market odds.

        The totals section has a line (e.g. 125.5) plus over/under odds.

        Returns:
            {"line": float, "over_odds": float, "under_odds": float} or None.
        """
        for label in [
            "Total Goals",
            "Total Points",
            "Total Match Goals",
            "Totals",
            "Over/Under",
        ]:
            section = self._find_market_section(page, label)
            if section:
                break
        else:
            return None

        line = self._extract_line_from_section(section)
        odds_values = self._extract_odds_from_section(page, section, expected_count=2)
        if odds_values and len(odds_values) >= 2:
            return {
                "line": line,
                "over_odds": odds_values[0],
                "under_odds": odds_values[1],
            }

        return None

    # ------------------------------------------------------------------
    # Low-level DOM helpers
    # ------------------------------------------------------------------

    def _find_market_section(self, page: Page, header_text: str):
        """Find a market section by its header text.

        Returns the locator for the section container, or None if not found.
        bet365 wraps each market in a container div; the header is a child
        element with the market name. We find the header then go to its
        closest ancestor that wraps the entire market (header + odds rows).
        """
        try:
            header = page.get_by_text(header_text, exact=False).first
            if not header or not header.is_visible(timeout=2000):
                return None
            # Walk up to the market container. Typically 2-4 levels above
            # the text element. We try a few ancestor depths.
            for depth in range(2, 6):
                xpath = f"xpath=ancestor::div[{depth}]"
                try:
                    container = header.locator(xpath).first
                    container_text = container.inner_text(timeout=1000) or ""
                    # A valid market section should contain odds-like numbers
                    # (decimal values > 1.0) in addition to the header.
                    if re.search(r"\b[1-9]\d*\.\d{2}\b", container_text):
                        return container
                except Exception:
                    continue
        except Exception:
            pass

        return None

    def _extract_odds_from_section(
        self,
        page: Page,
        section,
        *,
        expected_count: int = 2,
    ) -> list[float] | None:
        """Extract decimal odds values from a market section (or page).

        Scans for text nodes matching the pattern of decimal odds (e.g.
        "1.38", "2.50"). Returns up to *expected_count* valid odds.

        If *section* is None, scans the full page (fallback).
        """
        target = section if section else page
        try:
            # Odds on bet365 are typically in <span> elements, sometimes
            # inside clickable buttons/divs. They appear as decimal numbers
            # with exactly 2 decimal places.
            all_text = target.inner_text(timeout=3000) or ""

            # Find all decimal-odds-like values.
            # Pattern: 1.01 to 999.99, must be on a word boundary.
            candidates = re.findall(r"\b(\d{1,3}\.\d{2})\b", all_text)

            odds: list[float] = []
            for c in candidates:
                v = float(c)
                # Valid decimal odds range for sports betting.
                if 1.01 <= v <= 200.0:
                    odds.append(v)
                if len(odds) >= expected_count:
                    break

            return odds if odds else None
        except Exception:
            return None

    def _extract_line_from_section(self, section) -> float | None:
        """Extract a handicap or total line value from a market section.

        Lines look like "-4.5", "+4.5", or "125.5". They are usually
        displayed near the odds but are not odds themselves (they can
        be negative or much larger than typical odds).
        """
        if not section:
            return None
        try:
            text = section.inner_text(timeout=2000) or ""
            # Look for lines: signed or unsigned decimals with .5.
            # Handicap: -4.5, +4.5
            # Totals: 125.5, 130.5, etc.
            # We look for .5 values since lines almost always end in .5.
            matches = re.findall(r"([+-]?\d+\.5)\b", text)
            if matches:
                return _parse_line(matches[0])

            # Fallback: any decimal that is not a valid odds value.
            matches = re.findall(r"([+-]?\d+\.\d+)", text)
            for m in matches:
                v = float(m)
                # Lines are typically outside normal odds range, or negative.
                if v < 0 or v > 50.0:
                    return v
        except Exception:
            pass

        return None
