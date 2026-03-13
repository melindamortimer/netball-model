from __future__ import annotations

import math
from dataclasses import dataclass

from glicko2 import Player


@dataclass
class TeamRating:
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06


class GlickoSystem:
    """Glicko-2 rating system with margin-of-victory scaling and separate pools."""

    def __init__(self):
        self._ratings: dict[str, dict[str, TeamRating]] = {}

    def _key(self, team: str, pool: str) -> TeamRating:
        """Get or create the TeamRating for a team in a given pool."""
        if pool not in self._ratings:
            self._ratings[pool] = {}
        if team not in self._ratings[pool]:
            self._ratings[pool][team] = TeamRating()
        return self._ratings[pool][team]

    def get_rating(self, team: str, pool: str = "ssn") -> dict:
        """Return the current rating dict for a team in a pool."""
        tr = self._key(team, pool)
        return {"rating": tr.rating, "rd": tr.rd, "vol": tr.vol}

    def update(
        self,
        home_team: str,
        away_team: str,
        winner: str,
        margin: int = 0,
        pool: str = "ssn",
    ):
        """Update ratings after a match result.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            winner: One of "home", "away", or "draw".
            margin: Absolute goal/point margin of the match.
            pool: Rating pool (e.g. "ssn", "international").
        """
        home_tr = self._key(home_team, pool)
        away_tr = self._key(away_team, pool)

        home_player = Player(rating=home_tr.rating, rd=home_tr.rd, vol=home_tr.vol)
        away_player = Player(rating=away_tr.rating, rd=away_tr.rd, vol=away_tr.vol)

        if winner == "home":
            home_score, away_score = 1.0, 0.0
        elif winner == "away":
            home_score, away_score = 0.0, 1.0
        else:
            home_score, away_score = 0.5, 0.5

        # Margin-of-victory multiplier using log scaling.
        mov_mult = math.log(max(abs(margin), 1) + 1)

        # Save original opponent values before any mutation. The glicko2 library
        # mutates Player objects in place, so we snapshot the ratings first to
        # ensure each player is updated against the *pre-match* opponent values.
        orig_home_rating = home_tr.rating
        orig_home_rd = home_tr.rd
        orig_away_rating = away_tr.rating
        orig_away_rd = away_tr.rd

        home_player.update_player(
            [orig_away_rating], [orig_away_rd], [home_score]
        )
        away_player.update_player(
            [orig_home_rating], [orig_home_rd], [away_score]
        )

        # Apply margin-of-victory scaling to the rating delta only.
        home_delta = home_player.rating - orig_home_rating
        away_delta = away_player.rating - orig_away_rating

        home_tr.rating = orig_home_rating + home_delta * mov_mult
        home_tr.rd = home_player.rd
        home_tr.vol = home_player.vol

        away_tr.rating = orig_away_rating + away_delta * mov_mult
        away_tr.rd = away_player.rd
        away_tr.vol = away_player.vol

    def predict_win_prob(
        self, home_team: str, away_team: str, pool: str = "ssn"
    ) -> float:
        """Predict the probability that the home team wins.

        Uses the Glicko-2 expected score formula, accounting for
        both teams' rating deviations.
        """
        home = self._key(home_team, pool)
        away = self._key(away_team, pool)

        q = math.log(10) / 400
        combined_rd = math.sqrt(home.rd**2 + away.rd**2)
        g = 1 / math.sqrt(1 + 3 * q**2 * combined_rd**2 / math.pi**2)
        expected = 1 / (1 + 10 ** (-g * (home.rating - away.rating) / 400))
        return expected

    def get_all_ratings(self, pool: str = "ssn") -> dict[str, dict]:
        """Return all team ratings in a given pool."""
        if pool not in self._ratings:
            return {}
        return {
            team: {"rating": tr.rating, "rd": tr.rd, "vol": tr.vol}
            for team, tr in self._ratings[pool].items()
        }

    def regress_ratings(self, factor: float = 0.2, mean: float = 1500.0, pool: str = "ssn"):
        """Regress all team ratings toward the mean by factor."""
        if pool not in self._ratings:
            return
        for tr in self._ratings[pool].values():
            tr.rating = tr.rating * (1 - factor) + mean * factor

    def increase_rd(self, amount: float = 30.0, pool: str = "ssn"):
        """Increase rating deviation for all teams (off-season uncertainty)."""
        if pool not in self._ratings:
            return
        for tr in self._ratings[pool].values():
            tr.rd = min(tr.rd + amount, 350.0)  # cap at initial RD
