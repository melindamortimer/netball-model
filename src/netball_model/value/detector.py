from __future__ import annotations

from scipy.stats import norm


class ValueDetector:
    def __init__(self, min_edge: float = 0.05):
        self.min_edge = min_edge

    def evaluate(
        self,
        prediction: dict,
        odds: dict,
        threshold: float | None = None,
    ) -> list[dict]:
        """Evaluate all available markets for value.

        Args:
            prediction: {"margin", "total_goals", "win_prob",
                         "residual_std", "total_residual_std"}
            odds: {"home_odds", "away_odds", optionally "handicap_line",
                   "handicap_home_odds", "handicap_away_odds",
                   "total_line", "over_odds", "under_odds"}
            threshold: Override min_edge for this call
        """
        edge_threshold = threshold if threshold is not None else self.min_edge
        results = []

        # H2H
        results.extend(self._evaluate_h2h(prediction, odds, edge_threshold))

        # Handicap
        if odds.get("handicap_line") is not None and odds.get("handicap_home_odds"):
            results.extend(self._evaluate_handicap(prediction, odds, edge_threshold))

        # Totals
        if odds.get("total_line") is not None and odds.get("over_odds"):
            results.extend(self._evaluate_total(prediction, odds, edge_threshold))

        return results

    def _evaluate_h2h(self, prediction: dict, odds: dict, threshold: float) -> list[dict]:
        results = []
        home_odds = odds.get("home_odds")
        away_odds = odds.get("away_odds")
        win_prob = prediction["win_prob"]

        if home_odds:
            implied = 1 / home_odds
            edge = win_prob - implied
            results.append({
                "market": "h2h", "side": "home",
                "model_prob": round(win_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": home_odds, "line": None,
            })

        if away_odds:
            away_prob = 1 - win_prob
            implied = 1 / away_odds
            edge = away_prob - implied
            results.append({
                "market": "h2h", "side": "away",
                "model_prob": round(away_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": away_odds, "line": None,
            })

        return results

    def _evaluate_handicap(self, prediction: dict, odds: dict, threshold: float) -> list[dict]:
        results = []
        margin = prediction["margin"]
        sigma = prediction["residual_std"]
        line = odds["handicap_line"]

        # P(home covers) = 1 - Phi((L - M) / sigma)
        home_prob = 1 - norm.cdf((line - margin) / sigma)
        away_prob = 1 - home_prob

        home_odds = odds.get("handicap_home_odds")
        if home_odds:
            implied = 1 / home_odds
            edge = home_prob - implied
            results.append({
                "market": "handicap", "side": "home",
                "model_prob": round(home_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": home_odds, "line": line,
            })

        away_odds = odds.get("handicap_away_odds")
        if away_odds:
            implied = 1 / away_odds
            edge = away_prob - implied
            results.append({
                "market": "handicap", "side": "away",
                "model_prob": round(away_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": away_odds, "line": -line,
            })

        return results

    def _evaluate_total(self, prediction: dict, odds: dict, threshold: float) -> list[dict]:
        results = []
        total = prediction["total_goals"]
        sigma = prediction["total_residual_std"]
        line = odds["total_line"]

        # P(over) = 1 - Phi((L - T) / sigma)
        over_prob = 1 - norm.cdf((line - total) / sigma)
        under_prob = 1 - over_prob

        over_odds = odds.get("over_odds")
        if over_odds:
            implied = 1 / over_odds
            edge = over_prob - implied
            results.append({
                "market": "total", "side": "over",
                "model_prob": round(over_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": over_odds, "line": line,
            })

        under_odds = odds.get("under_odds")
        if under_odds:
            implied = 1 / under_odds
            edge = under_prob - implied
            results.append({
                "market": "total", "side": "under",
                "model_prob": round(under_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "odds": under_odds, "line": line,
            })

        return results
