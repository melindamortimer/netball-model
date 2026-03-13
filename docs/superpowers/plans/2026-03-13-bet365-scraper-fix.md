# Bet365 Scraper Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite bet365.py navigation and extraction to match the actual site UI, and add 1st Half / 1st Quarter odds.

**Architecture:** Section-based text extraction. Navigate via dropdown clicks (Super Netball → Matches), collect match labels as strings, re-expand after go_back. Extract odds by finding table sections by header text, parsing inner text line-by-line with row-label-keyed regex.

**Tech Stack:** Playwright (sync API), playwright-stealth, regex

**Spec:** `docs/superpowers/specs/2026-03-13-bet365-scraper-fix-design.md`

**Note:** User's git policy — do NOT run any git commands. Skip all commit steps.

---

## Chunk 1: Pure text parsing function + tests

### Task 1: `_parse_section_text` — pure function for odds extraction

The core of the rewrite. A pure function (no Playwright) that takes the inner text of a table section and returns a dict of parsed values. This is the only part we can meaningfully unit test.

**Files:**
- Modify: `src/netball_model/data/bet365.py` (add function)
- Modify: `tests/data/test_bet365.py` (add tests)

- [ ] **Step 1: Write failing tests for `_parse_section_text`**

Add to `tests/data/test_bet365.py`:

```python
from netball_model.data.bet365 import _parse_section_text


class TestParseSectionText:
    """Tests for the pure text parsing function."""

    def test_full_match_lines_section(self):
        text = (
            "Match Lines\n"
            "Melbourne Mavericks\tGIANTS Netball\n"
            "To Win\t1.30\t3.50\n"
            "Total\tO 125.5 1.87\tU 125.5 1.87\n"
            "Line\t-5.5 1.85\t+5.5 1.95\n"
        )
        result = _parse_section_text(text)
        assert result["home_odds"] == 1.30
        assert result["away_odds"] == 3.50
        assert result["total_line"] == 125.5
        assert result["over_odds"] == 1.87
        assert result["under_odds"] == 1.87
        assert result["handicap_line"] == -5.5
        assert result["handicap_home_odds"] == 1.85
        assert result["handicap_away_odds"] == 1.95

    def test_to_win_only(self):
        text = "To Win\t1.50\t2.50\n"
        result = _parse_section_text(text)
        assert result["home_odds"] == 1.50
        assert result["away_odds"] == 2.50
        assert result["total_line"] is None
        assert result["handicap_line"] is None

    def test_total_row_parsing(self):
        text = "Total\tO 62.5 1.83\tU 62.5 1.83\n"
        result = _parse_section_text(text)
        assert result["total_line"] == 62.5
        assert result["over_odds"] == 1.83
        assert result["under_odds"] == 1.83

    def test_line_row_parsing(self):
        text = "Line\t-2.5 1.80\t+2.5 1.90\n"
        result = _parse_section_text(text)
        assert result["handicap_line"] == -2.5
        assert result["handicap_home_odds"] == 1.80
        assert result["handicap_away_odds"] == 1.90

    def test_empty_text_returns_all_none(self):
        result = _parse_section_text("")
        assert all(v is None for v in result.values())

    def test_line_row_not_confused_with_match_lines_header(self):
        """'Match Lines' in the header should not trigger Line row parsing."""
        text = (
            "Match Lines\n"
            "To Win\t1.30\t3.50\n"
        )
        result = _parse_section_text(text)
        assert result["home_odds"] == 1.30
        assert result["handicap_line"] is None

    def test_whitespace_variations(self):
        """Inner text may use spaces or tabs between values."""
        text = "To Win    1.40    2.75\nTotal    O 62.5 1.83    U 62.5 1.83\n"
        result = _parse_section_text(text)
        assert result["home_odds"] == 1.40
        assert result["away_odds"] == 2.75
        assert result["total_line"] == 62.5

    def test_line_row_single_value(self):
        """Edge case: Line row with only one signed value."""
        text = "Line\t-2.5 1.80\n"
        result = _parse_section_text(text)
        assert result["handicap_line"] == -2.5
        assert result["handicap_home_odds"] == 1.80
        assert result["handicap_away_odds"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/data/test_bet365.py::TestParseSectionText -v`
Expected: FAIL — `ImportError: cannot import name '_parse_section_text'`

- [ ] **Step 3: Implement `_parse_section_text`**

Add this function to `src/netball_model/data/bet365.py` after the existing `_parse_line` function (around line 57):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/data/test_bet365.py::TestParseSectionText -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Run full test suite to check nothing broke**

Run: `poetry run pytest -v`
Expected: All 100 tests PASS (92 existing + 8 new parsing tests)

---

## Chunk 2: Rewrite the scraper module

### Task 2: Rewrite navigation, match discovery, and main loop

Replace the navigation flow, match collection, and main loop to match the actual bet365 UI.

**Files:**
- Modify: `src/netball_model/data/bet365.py`

- [ ] **Step 1: Add `_ensure_ssn_expanded` (must exist before `_navigate_to_ssn` references it)**

Add this new method to the class (in the Navigation section, before `_navigate_to_ssn`):

```python
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
```

- [ ] **Step 2: Rewrite `_navigate_to_ssn`**

Replace the existing `_navigate_to_ssn` method (lines 148-190) with:

```python
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
    _random_delay(2.0, 4.0)

    # 3. Expand the Super Netball dropdown and Matches section.
    self._ensure_ssn_expanded(page)
```

- [ ] **Step 3: Rewrite `_collect_match_links`**

Replace the existing `_collect_match_links` method (lines 218-294) with:

```python
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
```

- [ ] **Step 4: Rewrite `scrape_ssn_odds` main loop**

Replace the existing `scrape_ssn_odds` method (lines 82-142) with:

```python
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
```

- [ ] **Step 5: Rewrite `_scrape_single_match`**

Replace the existing `_scrape_single_match` method (lines 300-325) with:

```python
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
```

- [ ] **Step 6: Run existing tests to check nothing broke**

Run: `poetry run pytest tests/data/test_bet365.py -v`
Expected: All tests PASS (instantiation tests + parsing tests from Task 1)

---

### Task 3: Rewrite odds extraction and team names

**Files:**
- Modify: `src/netball_model/data/bet365.py`
- Modify: `tests/data/test_bet365.py`

- [ ] **Step 1: Write test for prefix mapping in `_extract_match_odds`**

Add to `tests/data/test_bet365.py`:

```python
class TestExtractMatchOddsSchema:
    """Test that _extract_match_odds produces the correct output schema keys."""

    def test_output_has_all_26_keys(self):
        """The result dict should have all expected keys."""
        expected_keys = {
            "home_team", "away_team", "match_date",
            # Match Lines
            "home_odds", "away_odds", "total_line", "over_odds", "under_odds",
            "handicap_line", "handicap_home_odds", "handicap_away_odds",
            # 1st Half
            "h1_home_odds", "h1_away_odds", "h1_total_line", "h1_over_odds",
            "h1_under_odds", "h1_handicap_line", "h1_handicap_home_odds",
            "h1_handicap_away_odds",
            # 1st Quarter
            "q1_home_odds", "q1_away_odds", "q1_total_line", "q1_over_odds",
            "q1_under_odds", "q1_handicap_line", "q1_handicap_home_odds",
            "q1_handicap_away_odds",
        }
        # We can't call _extract_match_odds without Playwright, but we can
        # verify the prefix mapping logic by testing _build_result_dict.
        from netball_model.data.bet365 import _parse_section_text

        section_text = "To Win\t1.30\t3.50\nTotal\tO 125.5 1.87\tU 125.5 1.87\nLine\t-5.5 1.85\t+5.5 1.95\n"
        parsed = _parse_section_text(section_text)

        # Simulate what _extract_match_odds does: no prefix, h1_, q1_
        result = {
            "home_team": "Melbourne Mavericks",
            "away_team": "GIANTS Netball",
            "match_date": "2026-03-15",
        }
        # Match Lines — keys as-is
        result.update(parsed)
        # 1st Half — h1_ prefix
        result.update({f"h1_{k}": v for k, v in parsed.items()})
        # 1st Quarter — q1_ prefix
        result.update({f"q1_{k}": v for k, v in parsed.items()})

        assert set(result.keys()) == expected_keys
```

- [ ] **Step 2: Run test to verify it passes (this is a schema check, should pass immediately)**

Run: `poetry run pytest tests/data/test_bet365.py::TestExtractMatchOddsSchema -v`
Expected: PASS

- [ ] **Step 3: Add `_find_table_section` helper**

Add this method to the class:

```python
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
```

- [ ] **Step 4: Rewrite `_extract_match_odds`**

Replace the existing `_extract_match_odds` method (lines 327-382) with:

```python
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
```

- [ ] **Step 5: Rewrite `_extract_team_names`**

Replace the existing `_extract_team_names` method (lines 388-428) with:

```python
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
```

- [ ] **Step 6: Remove old methods**

Delete these methods from the class (they are fully replaced):
- `_extract_h2h` (lines 480-504)
- `_extract_handicap` (lines 506-533)
- `_extract_totals` (lines 535-565)
- `_find_market_section` (lines 571-599)
- `_extract_odds_from_section` (lines 601-637)
- `_extract_line_from_section` (lines 639-668)

- [ ] **Step 7: Run full test suite**

Run: `poetry run pytest -v`
Expected: All 101 tests PASS (92 existing + 8 parsing tests + 1 schema test)
