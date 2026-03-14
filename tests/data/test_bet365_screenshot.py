import pytest
from unittest.mock import patch, MagicMock
from netball_model.data.bet365_screenshot import _parse_section_text, parse_screenshot


class TestParseSectionText:
    def test_to_win_row(self):
        text = "To Win\n1.30\n3.50"
        result = _parse_section_text(text)
        assert result["home_odds"] == 1.30
        assert result["away_odds"] == 3.50

    def test_total_row(self):
        text = "Total\nO 125.5 1.87\nU 125.5 1.87"
        result = _parse_section_text(text)
        assert result["total_line"] == 125.5
        assert result["over_odds"] == 1.87
        assert result["under_odds"] == 1.87

    def test_line_row(self):
        text = "Line\n-5.5 1.85\n+5.5 1.95"
        result = _parse_section_text(text)
        assert result["handicap_line"] == -5.5
        assert result["handicap_home_odds"] == 1.85
        assert result["handicap_away_odds"] == 1.95

    def test_missing_row_returns_none(self):
        text = "To Win\n1.30\n3.50"
        result = _parse_section_text(text)
        assert result.get("total_line") is None
        assert result.get("handicap_line") is None

    def test_full_section(self):
        text = (
            "Match Lines\n"
            "Melbourne Mavericks    GIANTS Netball\n"
            "To Win\n1.30\n3.50\n"
            "Total\nO 125.5 1.87\nU 125.5 1.87\n"
            "Line\n-5.5 1.85\n+5.5 1.95"
        )
        result = _parse_section_text(text)
        assert result["home_odds"] == 1.30
        assert result["away_odds"] == 3.50
        assert result["total_line"] == 125.5
        assert result["handicap_line"] == -5.5


class TestParseScreenshot:
    def test_full_parse_mocked_ocr(self, tmp_path):
        """Test full pipeline with mocked EasyOCR output."""
        from PIL import Image
        img = Image.new("RGB", (1200, 800), color=(30, 30, 30))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        mock_results = [
            # Match title
            ([0, 0, 400, 30], "Super Netball", 0.9),
            ([0, 30, 600, 60], "Melbourne Mavericks v GIANTS Netball", 0.95),
            # Match Lines section
            ([0, 100, 200, 130], "Match Lines", 0.9),
            ([0, 140, 100, 160], "To Win", 0.9),
            ([200, 140, 300, 160], "1.30", 0.95),
            ([500, 140, 600, 160], "3.50", 0.95),
            ([0, 170, 100, 190], "Total", 0.9),
            ([200, 170, 400, 190], "O 125.5 1.87", 0.95),
            ([500, 170, 700, 190], "U 125.5 1.87", 0.95),
            ([0, 200, 100, 220], "Line", 0.9),
            ([200, 200, 400, 220], "-5.5 1.85", 0.95),
            ([500, 200, 700, 220], "+5.5 1.95", 0.95),
        ]

        with patch("netball_model.data.bet365_screenshot.easyocr") as mock_easyocr:
            reader = MagicMock()
            reader.readtext.return_value = mock_results
            mock_easyocr.Reader.return_value = reader

            result = parse_screenshot(str(img_path))

        assert result["home_team"] == "Melbourne Mavericks"
        assert result["away_team"] == "GIANTS Netball"
        assert result["home_odds"] == 1.30
        assert result["away_odds"] == 3.50
        assert result["total_line"] == 125.5
        assert result["handicap_line"] == -5.5

    def test_missing_sections_return_none(self, tmp_path):
        """When only Match Lines is visible, h1_ and q1_ keys are None."""
        from PIL import Image
        img = Image.new("RGB", (1200, 400), color=(30, 30, 30))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        mock_results = [
            ([0, 30, 600, 60], "Melbourne Mavericks v GIANTS Netball", 0.95),
            ([0, 100, 200, 130], "Match Lines", 0.9),
            ([0, 140, 100, 160], "To Win", 0.9),
            ([200, 140, 300, 160], "1.30", 0.95),
            ([500, 140, 600, 160], "3.50", 0.95),
        ]

        with patch("netball_model.data.bet365_screenshot.easyocr") as mock_easyocr:
            reader = MagicMock()
            reader.readtext.return_value = mock_results
            mock_easyocr.Reader.return_value = reader

            result = parse_screenshot(str(img_path))

        assert result["home_odds"] == 1.30
        assert result["h1_home_odds"] is None
        assert result["q1_home_odds"] is None

    def test_empty_ocr_returns_none(self, tmp_path):
        """parse_screenshot returns None when OCR produces no results."""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        img_path = tmp_path / "empty.png"
        img.save(img_path)

        with patch("netball_model.data.bet365_screenshot.easyocr") as mock_easyocr:
            reader = MagicMock()
            reader.readtext.return_value = []  # empty OCR
            mock_easyocr.Reader.return_value = reader

            result = parse_screenshot(str(img_path))

        assert result is None

    def test_invalid_image_returns_none(self, tmp_path):
        bad_path = tmp_path / "nonexistent.png"
        result = parse_screenshot(str(bad_path))
        assert result is None
