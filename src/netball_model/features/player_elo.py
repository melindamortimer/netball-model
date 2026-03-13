"""Individual player Glicko-2 ratings based on positional matchups."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import glicko2

from netball_model.features.player_profile import PlayerProfiler
from netball_model.features.matchups import (
    _gs_vs_gk, _ga_vs_gd, _wa_vs_wd, _c_vs_c, _wd_vs_wa,
)


@dataclass
class PlayerRating:
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06


# 9 bidirectional pairings: (attacker_pos, defender_pos, matchup_fn, win_keys)
_PAIRINGS = [
    # Home attacking -> Away defending
    ("GS", "GK", _gs_vs_gk, ["gs_vs_gk_shooting_pressure"]),
    ("GA", "GD", _ga_vs_gd, ["ga_vs_gd_shooting_pressure", "ga_vs_gd_feed_vs_intercept"]),
    ("WA", "WD", _wa_vs_wd, ["wa_vs_wd_delivery_vs_disruption", "wa_vs_wd_supply_line"]),
    # Midcourt
    ("C", "C", _c_vs_c, ["c_vs_c_distribution_battle", "c_vs_c_disruption_battle"]),
    ("WD", "WA", _wd_vs_wa, ["wd_vs_wa_pressure_effectiveness", "wd_vs_wa_aerial_battle"]),
]


class PlayerGlicko2:
    def __init__(self):
        self._ratings: dict[tuple[int, str], PlayerRating] = {}

    def get_rating(self, player_id: int, position: str) -> PlayerRating:
        key = (player_id, position)
        if key not in self._ratings:
            self._ratings[key] = PlayerRating()
        return self._ratings[key]

    def process_match(
        self, match: dict, home_starters: list[dict], away_starters: list[dict]
    ) -> None:
        """Update player ratings based on positional matchup outcomes."""
        profiler = PlayerProfiler()

        # Build profiles: {position: profile_dict}
        home_profiles = self._build_profiles(home_starters, profiler)
        away_profiles = self._build_profiles(away_starters, profiler)

        # Build player_id lookup: {position: player_id}
        home_ids = {s["position"]: s["player_id"] for s in home_starters}
        away_ids = {s["position"]: s["player_id"] for s in away_starters}

        # Forward pairings: home attacks, away defends
        for att_pos, def_pos, fn, keys in _PAIRINGS:
            if att_pos not in home_profiles or def_pos not in away_profiles:
                continue
            features = fn(home_profiles[att_pos], away_profiles[def_pos])
            win_val = sum(features.get(k, 0) for k in keys) / len(keys)
            mov_scale = math.log(abs(win_val) + 1)
            att_id, def_id = home_ids.get(att_pos), away_ids.get(def_pos)
            if att_id is not None and def_id is not None:
                self._update_pair(att_id, att_pos, def_id, def_pos, win_val > 0, mov_scale)

        # Reverse pairings: away attacks, home defends
        for att_pos, def_pos, fn, keys in _PAIRINGS:
            if att_pos == "C" and def_pos == "C":
                continue  # C vs C already handled (symmetric)
            if att_pos == "WD" and def_pos == "WA":
                continue  # WD vs WA already covered in forward pass
            if att_pos not in away_profiles or def_pos not in home_profiles:
                continue
            features = fn(away_profiles[att_pos], home_profiles[def_pos])
            win_val = sum(features.get(k, 0) for k in keys) / len(keys)
            mov_scale = math.log(abs(win_val) + 1)
            att_id, def_id = away_ids.get(att_pos), home_ids.get(def_pos)
            if att_id is not None and def_id is not None:
                self._update_pair(att_id, att_pos, def_id, def_pos, win_val > 0, mov_scale)

    def _build_profiles(self, starters: list[dict], profiler: PlayerProfiler) -> dict:
        """Build {position: profile_dict} from starter stat rows."""
        profiles = {}
        for s in starters:
            pos = s["position"]
            if pos == "-":
                continue
            profiles[pos] = profiler.compute_profile([s], pos)
        return profiles

    def _update_pair(
        self, att_id: int, att_pos: str, def_id: int, def_pos: str,
        att_wins: bool, mov_scale: float
    ) -> None:
        """Apply Glicko-2 update to a matched pair of players."""
        att_r = self.get_rating(att_id, att_pos)
        def_r = self.get_rating(def_id, def_pos)

        att_g2 = glicko2.Player(rating=att_r.rating, rd=att_r.rd, vol=att_r.vol)
        def_g2 = glicko2.Player(rating=def_r.rating, rd=def_r.rd, vol=def_r.vol)

        att_score = 1.0 if att_wins else 0.0
        def_score = 1.0 - att_score

        # Scale the rating change by margin of victory
        att_g2.update_player([def_r.rating], [def_r.rd], [att_score])
        def_g2.update_player([att_r.rating], [att_r.rd], [def_score])

        # Apply MOV scaling to the rating delta
        att_delta = (att_g2.rating - att_r.rating) * max(mov_scale, 0.5)
        def_delta = (def_g2.rating - def_r.rating) * max(mov_scale, 0.5)

        att_r.rating = att_r.rating + att_delta
        att_r.rd = att_g2.rd
        att_r.vol = att_g2.vol
        def_r.rating = def_r.rating + def_delta
        def_r.rd = def_g2.rd
        def_r.vol = def_g2.vol

    def get_matchup_prediction(
        self, player_a_id: int, pos_a: str, player_b_id: int, pos_b: str
    ) -> float:
        """Predict win probability for player A vs player B."""
        a = self.get_rating(player_a_id, pos_a)
        b = self.get_rating(player_b_id, pos_b)
        expected = 1 / (1 + 10 ** ((b.rating - a.rating) / 400))
        return expected

    def regress_ratings(self, factor: float = 0.2, mean: float = 1500.0) -> None:
        """Regress all player ratings toward the mean by factor (season reset)."""
        for pr in self._ratings.values():
            pr.rating = pr.rating * (1 - factor) + mean * factor

    def increase_rd(self, amount: float = 30.0) -> None:
        """Increase RD for all players (off-season uncertainty)."""
        for pr in self._ratings.values():
            pr.rd = min(pr.rd + amount, 350.0)
