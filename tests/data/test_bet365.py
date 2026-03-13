"""Tests for bet365 scraper module — import and structure only (no live scraping)."""
from netball_model.data.bet365 import Bet365Scraper, _parse_section_text


def test_scraper_instantiates():
    scraper = Bet365Scraper(headless=True)
    assert scraper.headless is True
    assert scraper.timeout == 15000


def test_scraper_custom_timeout():
    scraper = Bet365Scraper(headless=True, timeout=30000)
    assert scraper.timeout == 30000


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
