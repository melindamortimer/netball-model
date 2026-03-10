from __future__ import annotations

import pandas as pd

from netball_model.features.contextual import (
    ContextualFeatures,
    TEAM_HOME_CITY,
    VENUE_TO_CITY,
)
from netball_model.features.elo import GlickoSystem
from netball_model.features.matchups import MatchupFeatures
from netball_model.features.player_profile import PlayerProfiler
from netball_model.match_utils import determine_winner


class FeatureBuilder:
    """Combines Glicko-2 ratings and contextual features into a feature matrix.

    Processes a chronological list of matches, replaying Glicko-2 updates
    incrementally so that each row's ratings reflect only information
    available *before* that match was played.
    """

    def __init__(self, matches: list[dict], pool: str = "ssn",
                 player_stats: dict[str, list[dict]] | None = None):
        self.matches = matches
        self.pool = pool
        self.glicko = GlickoSystem()
        self.ctx = ContextualFeatures(matches)
        self._elo_computed_up_to = -1
        self._player_stats = player_stats  # match_id -> [starters]
        self._profiler = PlayerProfiler() if player_stats else None
        self._matchups = MatchupFeatures() if player_stats else None

    def _ensure_elo_up_to(self, match_index: int):
        """Replay matches to update Elo up to (but not including) match_index."""
        for i in range(self._elo_computed_up_to + 1, match_index):
            m = self.matches[i]
            hs = m.get("home_score", 0)
            as_ = m.get("away_score", 0)
            self.glicko.update(
                m["home_team"], m["away_team"],
                winner=determine_winner(hs, as_), margin=hs - as_, pool=self.pool,
            )
        self._elo_computed_up_to = match_index - 1

    def _build_matchup_features(self, match_index: int) -> dict[str, float]:
        """Build position-pair difference features for the match at match_index."""
        m = self.matches[match_index]
        match_id = m["match_id"]
        home_team = m["home_team"]
        away_team = m["away_team"]

        starters = self._player_stats.get(match_id, [])
        if not starters:
            return {}

        home_profiles: dict[str, dict] = {}
        away_profiles: dict[str, dict] = {}

        for starter in starters:
            pos = starter["position"]
            if pos == "-":
                continue
            pid = starter["player_id"]
            team = starter["team"]

            history = self._get_player_history(pid, match_index)
            profile = self._profiler.compute_profile(history, pos)

            if profile is None:
                continue

            if team == home_team:
                home_profiles[pos] = profile
            elif team == away_team:
                away_profiles[pos] = profile

        return self._matchups.compute_features(home_profiles, away_profiles)

    def _get_player_history(self, player_id: int, before_index: int) -> list[dict]:
        """Get a player's stat rows from matches before before_index (up to 5)."""
        history = []
        for i in range(before_index - 1, -1, -1):
            if len(history) >= 5:
                break
            mid = self.matches[i]["match_id"]
            match_starters = self._player_stats.get(mid, [])
            for s in match_starters:
                if s["player_id"] == player_id:
                    history.append(s)
                    break
        return history

    def build_row(self, match_index: int) -> dict:
        """Build a single feature row for the match at *match_index*.

        Ratings are updated through all matches *before* match_index so
        the features represent pre-match knowledge only.
        """
        self._ensure_elo_up_to(match_index)

        m = self.matches[match_index]
        home = m["home_team"]
        away = m["away_team"]

        home_elo = self.glicko.get_rating(home, self.pool)
        away_elo = self.glicko.get_rating(away, self.pool)

        home_rest = self.ctx.rest_days(home, match_index)
        away_rest = self.ctx.rest_days(away, match_index)

        home_form_wr, home_form_margin = self.ctx.recent_form(home, match_index)
        away_form_wr, away_form_margin = self.ctx.recent_form(away, match_index)

        h2h = self.ctx.head_to_head(home, away, match_index)

        venue = m.get("venue", "")
        venue_city = VENUE_TO_CITY.get(venue, "")
        home_city = TEAM_HOME_CITY.get(home, "")
        away_city = TEAM_HOME_CITY.get(away, "")
        home_travel = self.ctx.travel_distance(home_city, venue_city) if home_city and venue_city else 0
        away_travel = self.ctx.travel_distance(away_city, venue_city) if away_city and venue_city else 0

        win_prob = self.glicko.predict_win_prob(home, away, self.pool)

        hs = m.get("home_score", 0)
        as_ = m.get("away_score", 0)

        row = {
            "match_id": m["match_id"],
            "home_team": home,
            "away_team": away,
            "home_elo": home_elo["rating"],
            "away_elo": away_elo["rating"],
            "home_elo_rd": home_elo["rd"],
            "away_elo_rd": away_elo["rd"],
            "elo_diff": home_elo["rating"] - away_elo["rating"],
            "elo_win_prob": win_prob,
            "home_rest_days": home_rest if home_rest is not None else 7,
            "away_rest_days": away_rest if away_rest is not None else 7,
            "rest_diff": (home_rest or 7) - (away_rest or 7),
            "home_form_win_rate": home_form_wr,
            "away_form_win_rate": away_form_wr,
            "home_form_avg_margin": home_form_margin,
            "away_form_avg_margin": away_form_margin,
            "h2h_home_win_rate": h2h,
            "home_travel_km": home_travel,
            "away_travel_km": away_travel,
            "travel_diff": away_travel - home_travel,
            "margin": hs - as_,
            "total_goals": hs + as_,
        }

        if self._player_stats is not None:
            row.update(self._build_matchup_features(match_index))

        return row

    def build_matrix(self, start_index: int = 1) -> pd.DataFrame:
        """Build the full feature matrix from *start_index* to the end.

        Typically ``start_index=1`` so the first match is used only to
        seed the rating system and contextual lookups.
        """
        rows = []
        for i in range(start_index, len(self.matches)):
            rows.append(self.build_row(i))
        return pd.DataFrame(rows)
