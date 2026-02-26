from __future__ import annotations


class ValueDetector:
    def __init__(self, min_edge: float = 0.05):
        self.min_edge = min_edge

    def evaluate(
        self,
        home_team: str,
        away_team: str,
        model_win_prob: float,
        betfair_home_back: float | None = None,
        betfair_away_back: float | None = None,
    ) -> dict:
        model_away_prob = 1.0 - model_win_prob

        home_implied = 1 / betfair_home_back if betfair_home_back else None
        away_implied = 1 / betfair_away_back if betfair_away_back else None

        home_edge = (model_win_prob - home_implied) if home_implied else None
        away_edge = (model_away_prob - away_implied) if away_implied else None

        best_side = "home"
        best_edge = home_edge or 0
        best_model_prob = model_win_prob
        best_implied = home_implied or 0
        best_odds = betfair_home_back

        if away_edge is not None and away_edge > (home_edge or 0):
            best_side = "away"
            best_edge = away_edge
            best_model_prob = model_away_prob
            best_implied = away_implied or 0
            best_odds = betfair_away_back

        return {
            "home_team": home_team,
            "away_team": away_team,
            "bet_side": best_side,
            "model_prob": best_model_prob,
            "implied_prob": best_implied,
            "edge": round(best_edge, 4),
            "odds": best_odds,
            "is_value": best_edge >= self.min_edge,
        }
