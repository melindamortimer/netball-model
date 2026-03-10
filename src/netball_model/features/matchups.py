"""Interaction-based matchup features from player profiles.

Instead of simple stat diffs (e.g. GS goals - GK goals), these features
capture cross-stat interactions: how one player's attacking strengths
match up against the opposing player's defensive strengths.
"""
from __future__ import annotations


def _safe_get(profile: dict | None, stat: str) -> float:
    if profile is None:
        return 0.0
    return profile.get(stat, 0.0)


def _gs_vs_gk(gs: dict | None, gk: dict | None) -> dict[str, float]:
    """Shooting circle: primary shooter vs last line of defense."""
    gs_shooting_pct = _safe_get(gs, "shooting_pct")
    gs_attempts = _safe_get(gs, "attempts")
    gs_rebounds = _safe_get(gs, "rebounds")
    gk_deflections = _safe_get(gk, "deflections")
    gk_rebounds = _safe_get(gk, "rebounds")
    gk_intercepts = _safe_get(gk, "intercepts")

    deflection_rate = gk_deflections / gs_attempts if gs_attempts > 0 else 0.0

    return {
        "gs_vs_gk_shooting_pressure": gs_shooting_pct - deflection_rate,
        "gs_vs_gk_rebounding_battle": gs_rebounds - gk_rebounds,
        "gs_vs_gk_shot_volume_vs_disruption": gs_attempts - (gk_intercepts + gk_deflections),
    }


def _ga_vs_gd(ga: dict | None, gd: dict | None) -> dict[str, float]:
    """Shooting circle: second shooter vs primary defender."""
    ga_shooting_pct = _safe_get(ga, "shooting_pct")
    ga_attempts = _safe_get(ga, "attempts")
    ga_feeds = _safe_get(ga, "feeds")
    ga_assists = _safe_get(ga, "assists")
    gd_deflections = _safe_get(gd, "deflections")
    gd_intercepts = _safe_get(gd, "intercepts")
    gd_gains = _safe_get(gd, "gains")

    deflection_rate = gd_deflections / ga_attempts if ga_attempts > 0 else 0.0

    return {
        "ga_vs_gd_shooting_pressure": ga_shooting_pct - deflection_rate,
        "ga_vs_gd_feed_vs_intercept": ga_feeds - gd_intercepts,
        "ga_vs_gd_creativity_vs_discipline": ga_assists - gd_gains,
    }


def _wa_vs_wd(wa: dict | None, wd: dict | None) -> dict[str, float]:
    """Mid-court: primary feeder vs wing defender."""
    wa_feeds = _safe_get(wa, "feeds")
    wa_turnovers = _safe_get(wa, "turnovers")
    wa_cpr = _safe_get(wa, "centre_pass_receives")
    wd_intercepts = _safe_get(wd, "intercepts")
    wd_deflections = _safe_get(wd, "deflections")
    wd_gains = _safe_get(wd, "gains")

    return {
        "wa_vs_wd_delivery_vs_disruption": wa_feeds - (wd_intercepts + wd_deflections),
        "wa_vs_wd_turnover_vulnerability": wa_turnovers - wd_gains,
        "wa_vs_wd_supply_line": wa_cpr - wd_deflections,
    }


def _c_vs_c(home_c: dict | None, away_c: dict | None) -> dict[str, float]:
    """Centre court: engine room battle."""
    h_feeds = _safe_get(home_c, "feeds")
    h_assists = _safe_get(home_c, "assists")
    h_gains = _safe_get(home_c, "gains")
    h_intercepts = _safe_get(home_c, "intercepts")
    h_turnovers = _safe_get(home_c, "turnovers")
    a_feeds = _safe_get(away_c, "feeds")
    a_assists = _safe_get(away_c, "assists")
    a_gains = _safe_get(away_c, "gains")
    a_intercepts = _safe_get(away_c, "intercepts")
    a_turnovers = _safe_get(away_c, "turnovers")

    return {
        "c_vs_c_distribution_battle": (h_feeds + h_assists) - (a_feeds + a_assists),
        "c_vs_c_disruption_battle": (h_gains + h_intercepts) - (a_gains + a_intercepts),
        "c_vs_c_turnover_differential": a_turnovers - h_turnovers,
    }


def _wd_vs_wa(wd: dict | None, wa: dict | None) -> dict[str, float]:
    """Reverse mid-court: defender pressuring feeder."""
    wd_gains = _safe_get(wd, "gains")
    wd_intercepts = _safe_get(wd, "intercepts")
    wd_deflections = _safe_get(wd, "deflections")
    wa_turnovers = _safe_get(wa, "turnovers")
    wa_feeds = _safe_get(wa, "feeds")

    return {
        "wd_vs_wa_pressure_effectiveness": (wd_gains + wd_intercepts) - wa_turnovers,
        "wd_vs_wa_aerial_battle": wd_deflections - wa_feeds,
    }


def _mean_of(features: dict[str, float], keys: list[str]) -> float:
    vals = [features[k] for k in keys]
    return sum(vals) / len(vals) if vals else 0.0


class MatchupFeatures:
    """Compute interaction-based matchup features from player profiles."""

    _ATTACK_KEYS = [
        "gs_vs_gk_shooting_pressure",
        "ga_vs_gd_shooting_pressure",
        "wa_vs_wd_delivery_vs_disruption",
    ]
    _DEFENCE_KEYS = [
        "wd_vs_wa_pressure_effectiveness",
        "wd_vs_wa_aerial_battle",
    ]
    _MIDCOURT_KEYS = [
        "c_vs_c_distribution_battle",
        "c_vs_c_disruption_battle",
        "c_vs_c_turnover_differential",
        "wa_vs_wd_delivery_vs_disruption",
        "wa_vs_wd_turnover_vulnerability",
        "wa_vs_wd_supply_line",
    ]

    def compute_features(
        self,
        home_profiles: dict[str, dict],
        away_profiles: dict[str, dict],
    ) -> dict[str, float]:
        """Compute interaction features for all matchup pairs.

        Each profile dict is keyed by position (e.g. "GS") with values
        from PlayerProfiler.compute_profile().

        Missing positions produce 0.0 for all their features.
        """
        features: dict[str, float] = {}

        features.update(_gs_vs_gk(home_profiles.get("GS"), away_profiles.get("GK")))
        features.update(_ga_vs_gd(home_profiles.get("GA"), away_profiles.get("GD")))
        features.update(_wa_vs_wd(home_profiles.get("WA"), away_profiles.get("WD")))
        features.update(_c_vs_c(home_profiles.get("C"), away_profiles.get("C")))
        features.update(_wd_vs_wa(home_profiles.get("WD"), away_profiles.get("WA")))

        features["attack_matchup"] = _mean_of(features, self._ATTACK_KEYS)
        features["defence_matchup"] = _mean_of(features, self._DEFENCE_KEYS)
        features["midcourt_matchup"] = _mean_of(features, self._MIDCOURT_KEYS)

        return features
