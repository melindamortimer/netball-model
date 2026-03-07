import numpy as np
import pandas as pd
import pytest

from netball_model.data.database import Database


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def seed_matches():
    """Return a function that seeds a database with synthetic matches."""

    def _seed(db: Database, n: int = 50):
        teams = [
            "Queensland Firebirds", "NSW Swifts", "Melbourne Vixens",
            "West Coast Fever", "Adelaide Thunderbirds", "GIANTS Netball",
            "Collingwood Magpies", "Sunshine Coast Lightning",
        ]
        rng = np.random.default_rng(42)
        matches = []
        for i in range(n):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            home_score = int(rng.integers(45, 70))
            away_score = int(rng.integers(45, 70))
            match = {
                "match_id": f"test_{i:03d}",
                "competition_id": 99999,
                "season": 2024,
                "round_num": (i // 4) + 1,
                "game_num": (i % 4) + 1,
                "date": f"2024-{((i // 4) % 10) + 3:02d}-{(i % 28) + 1:02d}",
                "venue": "Brisbane Entertainment Centre",
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "home_q1": home_score // 4,
                "home_q2": home_score // 4,
                "home_q3": home_score // 4,
                "home_q4": home_score - 3 * (home_score // 4),
                "away_q1": away_score // 4,
                "away_q2": away_score // 4,
                "away_q3": away_score // 4,
                "away_q4": away_score - 3 * (away_score // 4),
            }
            db.upsert_match(match)
            matches.append(match)
        return matches

    return _seed


@pytest.fixture
def dummy_feature_df():
    """Return a function that generates a dummy feature DataFrame."""

    def _make(n=100):
        rng = np.random.default_rng(42)
        elo_diff = rng.normal(0, 100, n)
        noise = rng.normal(0, 8, n)
        margin = 0.05 * elo_diff + noise
        return pd.DataFrame({
            "match_id": [f"m{i}" for i in range(n)],
            "home_team": ["A"] * n,
            "away_team": ["B"] * n,
            "elo_diff": elo_diff,
            "home_elo": 1500 + elo_diff / 2,
            "away_elo": 1500 - elo_diff / 2,
            "home_elo_rd": [200.0] * n,
            "away_elo_rd": [200.0] * n,
            "elo_win_prob": [0.5] * n,
            "home_rest_days": rng.integers(5, 10, n),
            "away_rest_days": rng.integers(5, 10, n),
            "rest_diff": [0] * n,
            "home_form_win_rate": rng.uniform(0.3, 0.7, n),
            "away_form_win_rate": rng.uniform(0.3, 0.7, n),
            "home_form_avg_margin": rng.normal(0, 5, n),
            "away_form_avg_margin": rng.normal(0, 5, n),
            "h2h_home_win_rate": [0.5] * n,
            "home_travel_km": [0.0] * n,
            "away_travel_km": rng.uniform(0, 3000, n),
            "travel_diff": rng.uniform(0, 3000, n),
            "margin": margin,
            "total_goals": rng.integers(90, 130, n).astype(float),
        })

    return _make
