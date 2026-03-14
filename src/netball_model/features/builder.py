from __future__ import annotations

import pandas as pd

from netball_model.features.contextual import (
    ContextualFeatures,
    TEAM_HOME_CITY,
    VENUE_TO_CITY,
)
from netball_model.features.elo import GlickoSystem
from netball_model.features.matchups import MatchupFeatures
from netball_model.features.player_elo import PlayerGlicko2
from netball_model.features.player_profile import PlayerProfiler
from netball_model.match_utils import determine_winner


class FeatureBuilder:
    """Combines Glicko-2 ratings and contextual features into a feature matrix.

    Processes a chronological list of matches, replaying Glicko-2 updates
    incrementally so that each row's ratings reflect only information
    available *before* that match was played.
    """

    def __init__(self, matches: list[dict], pool: str = "ssn",
                 player_stats: dict[str, list[dict]] | None = None,
                 roster_continuity: dict[tuple[str, int], float] | None = None):
        self.matches = matches
        self.pool = pool
        self.glicko = GlickoSystem()
        self.ctx = ContextualFeatures(matches)
        self._elo_computed_up_to = -1
        self._player_stats = player_stats  # match_id -> [starters]
        self._profiler = PlayerProfiler() if player_stats else None
        self._matchups = MatchupFeatures() if player_stats else None
        self._player_glicko = PlayerGlicko2() if player_stats else None
        self._player_elo_computed_up_to = -1
        self._roster_continuity = roster_continuity or {}

    def _ensure_elo_up_to(self, match_index: int):
        """Replay matches to update Elo up to (but not including) match_index."""
        start = self._elo_computed_up_to + 1
        prev_season = self.matches[start - 1].get("season") if start > 0 else None

        for i in range(start, match_index):
            m = self.matches[i]
            m_season = m.get("season")

            # Season boundary detection
            if prev_season is not None and m_season is not None and m_season != prev_season:
                self.glicko.regress_ratings(factor=0.2, mean=1500.0, pool=self.pool)
                self.glicko.increase_rd(amount=30.0, pool=self.pool)
                if self._player_glicko:
                    self._player_glicko.regress_ratings(factor=0.2, mean=1500.0)
                    self._player_glicko.increase_rd(amount=15.0)

            # Team Glicko update — skip unscored (upcoming) matches
            hs = m.get("home_score")
            as_ = m.get("away_score")
            if hs is not None and as_ is not None:
                self.glicko.update(
                    m["home_team"], m["away_team"],
                    winner=determine_winner(hs, as_), margin=hs - as_, pool=self.pool,
                )

                # Player Glicko update
                if self._player_glicko and self._player_stats:
                    starters = self._player_stats.get(m["match_id"], [])
                    home_starters = [s for s in starters if s["team"] == m["home_team"] and s["position"] != "-"]
                    away_starters = [s for s in starters if s["team"] == m["away_team"] and s["position"] != "-"]
                    if home_starters and away_starters:
                        self._player_glicko.process_match(m, home_starters, away_starters)

            prev_season = m_season

        self._elo_computed_up_to = match_index - 1
        if self._player_glicko:
            self._player_elo_computed_up_to = match_index - 1

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

    def _position_stats(self) -> dict[str, tuple[float, float]]:
        """Return {position: (mean, std)} from current player ratings."""
        from collections import defaultdict
        by_pos: dict[str, list[float]] = defaultdict(list)
        for (_, pos), pr in self._player_glicko._ratings.items():
            by_pos[pos].append(pr.rating)
        stats = {}
        for pos, ratings in by_pos.items():
            mean = sum(ratings) / len(ratings)
            var = sum((r - mean) ** 2 for r in ratings) / len(ratings)
            stats[pos] = (mean, max(var ** 0.5, 1.0))
        return stats

    def _build_player_elo_features(self, match_index: int) -> dict:
        """Compute z-scored, RD-weighted positional elo diff features."""
        m = self.matches[match_index]
        starters = self._player_stats.get(m["match_id"], [])
        home_starters = [s for s in starters if s["team"] == m["home_team"] and s["position"] != "-"]
        away_starters = [s for s in starters if s["team"] == m["away_team"] and s["position"] != "-"]

        pos_stats = self._position_stats()

        features = {}
        pos_pairs = [("GS", "GK"), ("GA", "GD"), ("WA", "WD"), ("C", "C"), ("WD", "WA")]
        home_by_pos = {s["position"]: s for s in home_starters}
        away_by_pos = {s["position"]: s for s in away_starters}

        for att_pos, def_pos in pos_pairs:
            key = f"{att_pos.lower()}_{def_pos.lower()}_player_elo_diff"
            h = home_by_pos.get(att_pos)
            a = away_by_pos.get(def_pos)
            if h and a:
                h_r = self._player_glicko.get_rating(h["player_id"], att_pos)
                a_r = self._player_glicko.get_rating(a["player_id"], def_pos)
                # Z-score each rating within its position pool
                h_mean, h_std = pos_stats.get(att_pos, (1500.0, 1.0))
                a_mean, a_std = pos_stats.get(def_pos, (1500.0, 1.0))
                z_diff = (h_r.rating - h_mean) / h_std - (a_r.rating - a_mean) / a_std
                # Dampen by average confidence (low RD = more trustworthy)
                confidence = 1.0 - (h_r.rd + a_r.rd) / 700.0
                features[key] = z_diff * max(confidence, 0.0)
            else:
                features[key] = 0.0

        return features

    def _compute_sample_weight(self, match_index: int) -> float:
        """Compute time-decay * roster-continuity weight for a match."""
        import math
        m = self.matches[match_index]
        current_season = self.matches[-1].get("season", m.get("season", 2025))
        match_season = m.get("season", current_season)
        years_ago = current_season - match_season

        base_weight = math.exp(-0.5 * years_ago)

        # Roster continuity adjustment (average of home and away)
        home_cont = self._roster_continuity.get((m["home_team"], match_season), 1.0)
        away_cont = self._roster_continuity.get((m["away_team"], match_season), 1.0)
        avg_continuity = (home_cont + away_cont) / 2
        weight = base_weight * (0.5 + 0.5 * avg_continuity)

        return max(weight, 0.01)  # floor to avoid zero weights

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

        hs = m.get("home_score") or 0
        as_ = m.get("away_score") or 0

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

        # Player Elo features (5 z-scored, RD-weighted positional diffs)
        if self._player_glicko and self._player_stats:
            player_elo_features = self._build_player_elo_features(match_index)
            row.update(player_elo_features)

        # Round features (2 features)
        round_feats = self.ctx.round_features(m)
        row.update(round_feats)

        # Sample weight (excluded from model features via NON_FEATURE_COLUMNS)
        row["_sample_weight"] = self._compute_sample_weight(match_index)

        return row

    def build_matrix(self, start_index: int = 1) -> pd.DataFrame:
        """Build the full feature matrix from *start_index* to the end.

        Typically ``start_index=1`` so the first match is used only to
        seed the rating system and contextual lookups.
        """
        rows = []
        for i in range(start_index, len(self.matches)):
            rows.append(self.build_row(i))
        df = pd.DataFrame(rows)
        # Fill NaN matchup/player elo features (matches without player stats) with 0.0
        df = df.fillna(0.0)
        return df
