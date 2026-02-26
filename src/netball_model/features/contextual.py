from __future__ import annotations

import math
from datetime import datetime

# Approximate lat/lon for SSN venue cities
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Brisbane": (-27.47, 153.03),
    "Melbourne": (-37.81, 144.96),
    "Sydney": (-33.87, 151.21),
    "Perth": (-31.95, 115.86),
    "Adelaide": (-34.93, 138.60),
    "Launceston": (-41.45, 147.14),
    "Gold Coast": (-28.02, 153.40),
    "Canberra": (-35.28, 149.13),
    "Hobart": (-42.88, 147.33),
}

# Map venue names to cities
VENUE_TO_CITY: dict[str, str] = {
    "Brisbane Entertainment Centre": "Brisbane",
    "Nissan Arena": "Brisbane",
    "Queensland State Netball Centre": "Brisbane",
    "John Cain Arena": "Melbourne",
    "Melbourne Arena": "Melbourne",
    "State Netball Hockey Centre": "Melbourne",
    "RAC Arena": "Perth",
    "Perth Arena": "Perth",
    "Ken Rosewall Arena": "Sydney",
    "Qudos Bank Arena": "Sydney",
    "Sydney Olympic Park": "Sydney",
    "Adelaide Entertainment Centre": "Adelaide",
    "Priceline Stadium": "Adelaide",
    "USC Stadium": "Gold Coast",
    "Gold Coast Convention Centre": "Gold Coast",
    "Silverdome": "Launceston",
    "MyState Bank Arena": "Launceston",
    "AIS Arena": "Canberra",
}

# Map team names to home cities
TEAM_HOME_CITY: dict[str, str] = {
    "Queensland Firebirds": "Brisbane",
    "Melbourne Vixens": "Melbourne",
    "NSW Swifts": "Sydney",
    "GIANTS Netball": "Sydney",
    "West Coast Fever": "Perth",
    "Adelaide Thunderbirds": "Adelaide",
    "Collingwood Magpies": "Melbourne",
    "Sunshine Coast Lightning": "Gold Coast",
    "Melbourne Mavericks": "Melbourne",
}


class ContextualFeatures:
    """Contextual features derived from match history.

    Provides rest days between matches, recent form (rolling win rate and
    average margin), head-to-head records, travel distance via Haversine
    formula, and home/away detection.
    """

    def __init__(self, matches: list[dict]) -> None:
        self.matches = matches

    def rest_days(self, team: str, match_index: int) -> int | None:
        """Calculate rest days since the team's previous match.

        Returns None when no prior match exists for the team.
        """
        current_date = self.matches[match_index].get("date", "")
        if not current_date:
            return None

        for i in range(match_index - 1, -1, -1):
            m = self.matches[i]
            if team in (m.get("home_team"), m.get("away_team")):
                prev_date = m.get("date", "")
                if prev_date:
                    d1 = datetime.fromisoformat(current_date[:10])
                    d2 = datetime.fromisoformat(prev_date[:10])
                    return (d1 - d2).days
        return None

    def recent_form(
        self, team: str, match_index: int, window: int = 5
    ) -> tuple[float, float]:
        """Calculate recent form as (win_rate, average_margin).

        Looks at up to ``window`` matches before ``match_index``.
        Returns (0.5, 0.0) when no prior matches are found.
        """
        wins = 0
        margins: list[int] = []

        for i in range(match_index - 1, -1, -1):
            if len(margins) >= window:
                break
            m = self.matches[i]
            home = m.get("home_team")
            away = m.get("away_team")
            hs = m.get("home_score", 0)
            as_ = m.get("away_score", 0)

            if team == home:
                margin = hs - as_
            elif team == away:
                margin = as_ - hs
            else:
                continue

            margins.append(margin)
            if margin > 0:
                wins += 1

        if not margins:
            return 0.5, 0.0

        return wins / len(margins), sum(margins) / len(margins)

    def head_to_head(
        self, team_a: str, team_b: str, match_index: int, window: int = 10
    ) -> float:
        """Calculate head-to-head win rate for *team_a* against *team_b*.

        Looks at up to ``window`` prior encounters before ``match_index``.
        Draws do not count as wins.  Returns 0.5 when no prior H2H found.
        """
        wins = 0
        total = 0

        for i in range(match_index - 1, -1, -1):
            if total >= window:
                break
            m = self.matches[i]
            home = m.get("home_team")
            away = m.get("away_team")
            hs = m.get("home_score", 0)
            as_ = m.get("away_score", 0)

            if {home, away} != {team_a, team_b}:
                continue

            total += 1
            if team_a == home and hs > as_:
                wins += 1
            elif team_a == away and as_ > hs:
                wins += 1

        if total == 0:
            return 0.5

        return wins / total

    @staticmethod
    def travel_distance(city_a: str, city_b: str) -> float:
        """Calculate great-circle distance between two cities in kilometres.

        Uses the Haversine formula with coordinates from ``CITY_COORDS``.
        Returns 0.0 when both cities are the same or coordinates are unknown.
        """
        if city_a == city_b:
            return 0.0

        coords_a = CITY_COORDS.get(city_a)
        coords_b = CITY_COORDS.get(city_b)
        if not coords_a or not coords_b:
            return 0.0

        lat1, lon1 = math.radians(coords_a[0]), math.radians(coords_a[1])
        lat2, lon2 = math.radians(coords_b[0]), math.radians(coords_b[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth radius in km
        return r * c

    @staticmethod
    def is_home(team: str, match: dict) -> bool:
        """Return True if *team* is the home team in *match*."""
        return match.get("home_team") == team
