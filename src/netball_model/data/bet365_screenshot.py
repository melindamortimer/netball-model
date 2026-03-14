"""Parse bet365 screenshots into structured odds data using OCR."""
from __future__ import annotations

import re
from pathlib import Path

import easyocr
from PIL import Image

TO_WIN_RE = re.compile(r"(\d+\.\d{2})")
TOTAL_RE = re.compile(r"([OU])\s+(\d+\.?\d*)\s+(\d+\.\d{2})")
LINE_RE = re.compile(r"([+-]\d+\.?\d*)\s+(\d+\.\d{2})")
MATCH_TITLE_RE = re.compile(r"(.+?)\s+v\s+(.+)")


def _parse_section_text(text: str) -> dict:
    """Parse raw text from one section into odds dict."""
    result = {}
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    to_win_odds = []
    total_matches = []
    line_matches = []

    i = 0
    while i < len(lines):
        stripped = lines[i]
        if stripped.startswith("To Win") or stripped == "To Win":
            # Collect the next odds values
            remaining = " ".join(lines[i:i+3])
            odds = TO_WIN_RE.findall(remaining)
            # Filter out any that are part of Total/Line patterns
            to_win_odds = [float(o) for o in odds[:2]]
            i += 1
            continue

        if stripped.startswith("Total") or stripped == "Total":
            remaining = " ".join(lines[i:i+3])
            total_matches = TOTAL_RE.findall(remaining)
            i += 1
            continue

        if stripped == "Line" or (stripped.startswith("Line") and "Match" not in stripped):
            remaining = " ".join(lines[i:i+3])
            line_matches = LINE_RE.findall(remaining)
            i += 1
            continue

        i += 1

    if len(to_win_odds) >= 2:
        result["home_odds"] = to_win_odds[0]
        result["away_odds"] = to_win_odds[1]

    for m in total_matches:
        direction, line_val, odds_val = m
        if direction == "O":
            result["total_line"] = float(line_val)
            result["over_odds"] = float(odds_val)
        elif direction == "U":
            result["under_odds"] = float(odds_val)

    if len(line_matches) >= 2:
        result["handicap_line"] = float(line_matches[0][0])
        result["handicap_home_odds"] = float(line_matches[0][1])
        result["handicap_away_odds"] = float(line_matches[1][1])

    return result


_SECTION_HEADERS = ["Match Lines", "1st Half", "1st Quarter"]
_SECTION_PREFIXES = {"Match Lines": "", "1st Half": "h1_", "1st Quarter": "q1_"}

_ALL_KEYS = [
    "home_team", "away_team", "match_date",
    "home_odds", "away_odds", "handicap_line", "handicap_home_odds",
    "handicap_away_odds", "total_line", "over_odds", "under_odds",
    "h1_home_odds", "h1_away_odds", "h1_handicap_line", "h1_handicap_home_odds",
    "h1_handicap_away_odds", "h1_total_line", "h1_over_odds", "h1_under_odds",
    "q1_home_odds", "q1_away_odds", "q1_handicap_line", "q1_handicap_home_odds",
    "q1_handicap_away_odds", "q1_total_line", "q1_over_odds", "q1_under_odds",
]


def parse_screenshot(image_path: str | Path) -> dict | None:
    """Parse a bet365 screenshot and return structured odds data.

    Returns None if the image cannot be parsed (missing file, OCR failure, etc.).
    """
    path = Path(image_path)
    if not path.exists():
        return None

    try:
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(str(path))
    except Exception:
        return None

    if not results:
        return None

    # Extract all text with positions: (bbox, text, confidence)
    texts = [(r[0], r[1], r[2]) for r in results]

    # Find match title
    home_team = None
    away_team = None
    for bbox, text, conf in texts:
        m = MATCH_TITLE_RE.search(text)
        if m and " v " in text:
            from netball_model.data.team_names import normalise_team
            home_team = normalise_team(m.group(1).strip())
            away_team = normalise_team(m.group(2).strip())
            break

    # Group text into sections by vertical position
    sections = _group_into_sections(texts)

    # Parse each section
    output = {k: None for k in _ALL_KEYS}
    output["home_team"] = home_team
    output["away_team"] = away_team
    output["match_date"] = None

    for section_name, section_text in sections.items():
        prefix = _SECTION_PREFIXES.get(section_name, "")
        parsed = _parse_section_text(section_text)
        for key, value in parsed.items():
            output[f"{prefix}{key}"] = value

    return output


def _group_into_sections(texts: list) -> dict[str, str]:
    """Group OCR text elements into sections based on header text."""
    # Find section header positions
    section_starts = []
    for bbox, text, conf in texts:
        for header in _SECTION_HEADERS:
            if header.lower() in text.lower():
                y_pos = bbox[0][1] if isinstance(bbox[0], list) else bbox[1]
                section_starts.append((y_pos, header))
                break

    if not section_starts:
        # No sections found, treat all as Match Lines
        all_text = "\n".join(t[1] for t in texts)
        return {"Match Lines": all_text}

    section_starts.sort(key=lambda x: x[0])

    # Assign text to sections by vertical position
    sections = {}
    for i, (y_start, name) in enumerate(section_starts):
        y_end = section_starts[i + 1][0] if i + 1 < len(section_starts) else float("inf")
        section_texts = []
        for bbox, text, conf in texts:
            text_y = bbox[0][1] if isinstance(bbox[0], list) else bbox[1]
            if y_start <= text_y < y_end:
                section_texts.append(text)
        sections[name] = "\n".join(section_texts)

    return sections
