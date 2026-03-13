import pytest
from netball_model.value.detector import ValueDetector


class TestH2HValue:
    def test_home_value(self):
        detector = ValueDetector(min_edge=0.05)
        results = detector.evaluate(
            prediction={"margin": 10.0, "total_goals": 120.0, "win_prob": 0.65,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.80, "away_odds": 2.10},
        )
        h2h = [r for r in results if r["market"] == "h2h"]
        assert len(h2h) >= 1
        best = max(h2h, key=lambda r: r["edge"])
        assert best["side"] == "home"
        assert best["edge"] > 0.05

    def test_no_value(self):
        detector = ValueDetector(min_edge=0.05)
        results = detector.evaluate(
            prediction={"margin": 2.0, "total_goals": 120.0, "win_prob": 0.55,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.75, "away_odds": 2.20},
        )
        # No bet should have edge > 5%
        h2h_value = [r for r in results if r["market"] == "h2h" and r["edge"] >= 0.05]
        assert len(h2h_value) == 0

    def test_away_value(self):
        detector = ValueDetector(min_edge=0.05)
        results = detector.evaluate(
            prediction={"margin": -5.0, "total_goals": 120.0, "win_prob": 0.35,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.60, "away_odds": 2.50},
        )
        h2h = [r for r in results if r["market"] == "h2h" and r["edge"] >= 0.05]
        assert any(r["side"] == "away" for r in h2h)


class TestHandicapValue:
    def test_home_covers_line(self):
        detector = ValueDetector(min_edge=0.0)
        results = detector.evaluate(
            prediction={"margin": 8.5, "total_goals": 120.0, "win_prob": 0.7,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.30, "away_odds": 3.50,
                  "handicap_line": -5.5, "handicap_home_odds": 1.85,
                  "handicap_away_odds": 1.95},
        )
        handicap = [r for r in results if r["market"] == "handicap"]
        assert len(handicap) >= 1
        home_hc = [r for r in handicap if r["side"] == "home"]
        assert len(home_hc) == 1
        # P(margin > 5.5) when predicted margin = 8.5, std = 12
        # = 1 - Phi((-5.5 - 8.5)/12) = 1 - Phi(-1.17) ≈ 0.879
        assert home_hc[0]["model_prob"] > 0.8

    def test_no_handicap_when_missing(self):
        detector = ValueDetector(min_edge=0.0)
        results = detector.evaluate(
            prediction={"margin": 8.5, "total_goals": 120.0, "win_prob": 0.7,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.30, "away_odds": 3.50},
        )
        handicap = [r for r in results if r["market"] == "handicap"]
        assert len(handicap) == 0


class TestTotalValue:
    def test_over_value(self):
        detector = ValueDetector(min_edge=0.0)
        results = detector.evaluate(
            prediction={"margin": 5.0, "total_goals": 130.0, "win_prob": 0.6,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.50, "away_odds": 2.60,
                  "total_line": 125.5, "over_odds": 1.87, "under_odds": 1.87},
        )
        totals = [r for r in results if r["market"] == "total"]
        assert len(totals) >= 1
        over = [r for r in totals if r["side"] == "over"]
        assert len(over) == 1
        # P(total > 125.5) when predicted = 130, std = 15
        # = 1 - Phi((125.5-130)/15) = 1 - Phi(-0.3) ≈ 0.618
        assert 0.55 < over[0]["model_prob"] < 0.70

    def test_no_total_when_missing(self):
        detector = ValueDetector(min_edge=0.0)
        results = detector.evaluate(
            prediction={"margin": 5.0, "total_goals": 130.0, "win_prob": 0.6,
                        "residual_std": 12.0, "total_residual_std": 15.0},
            odds={"home_odds": 1.50, "away_odds": 2.60},
        )
        totals = [r for r in results if r["market"] == "total"]
        assert len(totals) == 0


class TestCalibrationModel:
    def test_total_residual_std(self):
        import numpy as np
        from netball_model.model.calibration import CalibrationModel
        cal = CalibrationModel()
        margin_residuals = np.array([1.0, -2.0, 3.0, -1.0])
        total_residuals = np.array([5.0, -3.0, 4.0, -6.0])
        cal.fit(margin_residuals, total_residuals=total_residuals)
        assert cal.residual_std > 0
        assert cal.total_residual_std > 0
        assert abs(cal.total_residual_std - np.std(total_residuals)) < 0.01

    def test_backward_compat_no_total(self):
        import numpy as np
        from netball_model.model.calibration import CalibrationModel
        cal = CalibrationModel()
        cal.fit(np.array([1.0, -2.0, 3.0]))
        assert cal.residual_std > 0
        assert cal.total_residual_std == 10.0  # default
