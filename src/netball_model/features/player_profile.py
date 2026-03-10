"""Rolling player profile computation from historical stat rows."""
from __future__ import annotations

RAW_STATS = [
    "goals", "attempts", "assists", "rebounds", "feeds",
    "turnovers", "gains", "intercepts", "deflections",
    "penalties", "centre_pass_receives",
]


class PlayerProfiler:
    """Computes rolling average profiles from a player's recent stat rows."""

    def compute_profile(
        self, stat_rows: list[dict], position: str
    ) -> dict | None:
        """Compute a profile from up to N recent stat rows.

        Returns None if stat_rows is empty.
        """
        if not stat_rows:
            return None

        n = len(stat_rows)
        profile: dict = {"matches_used": n}

        for stat in RAW_STATS:
            total = sum(row.get(stat, 0) or 0 for row in stat_rows)
            profile[stat] = total / n

        # Derived ratios
        if position in ("GS", "GA"):
            profile["shooting_pct"] = (
                profile["goals"] / profile["attempts"]
                if profile["attempts"] > 0 else 0.0
            )
        if position in ("GD", "GK", "WD"):
            profile["clean_steal_rate"] = (
                profile["intercepts"] / profile["gains"]
                if profile["gains"] > 0 else 0.0
            )
        if position in ("WA", "C", "GA"):
            profile["delivery_efficiency"] = (
                profile["feeds"] / profile["turnovers"]
                if profile["turnovers"] > 0 else 0.0
            )

        return profile
