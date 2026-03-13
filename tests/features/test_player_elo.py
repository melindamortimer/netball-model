import pytest
import math
from netball_model.features.player_elo import PlayerGlicko2, PlayerRating


class TestPlayerGlicko2Init:
    def test_new_player_defaults(self):
        pg = PlayerGlicko2()
        r = pg.get_rating(999, "GS")
        assert r.rating == 1500.0
        assert r.rd == 350.0
        assert abs(r.vol - 0.06) < 0.001

    def test_get_matchup_prediction_equal_ratings(self):
        pg = PlayerGlicko2()
        prob = pg.get_matchup_prediction(1, "GS", 2, "GK")
        assert abs(prob - 0.5) < 0.01


class TestProcessMatch:
    def test_updates_all_pairings(self):
        pg = PlayerGlicko2()
        home_starters = [
            {"player_id": i, "player_name": f"H{i}", "team": "A", "position": pos,
             "goals": 10, "attempts": 15, "assists": 5, "rebounds": 3, "feeds": 8,
             "turnovers": 2, "gains": 3, "intercepts": 2, "deflections": 2,
             "penalties": 1, "centre_pass_receives": 4, "net_points": 10}
            for i, pos in enumerate(["GS", "GA", "WA", "C", "WD", "GD", "GK"], 1)
        ]
        away_starters = [
            {"player_id": i, "player_name": f"A{i}", "team": "B", "position": pos,
             "goals": 5, "attempts": 10, "assists": 3, "rebounds": 2, "feeds": 5,
             "turnovers": 4, "gains": 1, "intercepts": 1, "deflections": 1,
             "penalties": 2, "centre_pass_receives": 3, "net_points": 5}
            for i, pos in enumerate(["GS", "GA", "WA", "C", "WD", "GD", "GK"], 8)
        ]
        match = {"match_id": "m1", "season": 2024}

        pg.process_match(match, home_starters, away_starters)

        # All 14 players should have ratings updated from default
        for pid in range(1, 15):
            pos = ["GS", "GA", "WA", "C", "WD", "GD", "GK"][(pid - 1) % 7]
            r = pg.get_rating(pid, pos)
            # RD should decrease from 350 (uncertainty reduced after match)
            assert r.rd < 350.0

    def test_winner_rating_increases(self):
        pg = PlayerGlicko2()
        # Create starters where home GS dominates (high goals, low turnovers)
        home_starters = [
            {"player_id": 1, "player_name": "H1", "team": "A", "position": "GS",
             "goals": 30, "attempts": 35, "assists": 0, "rebounds": 5, "feeds": 2,
             "turnovers": 0, "gains": 0, "intercepts": 0, "deflections": 0,
             "penalties": 0, "centre_pass_receives": 0, "net_points": 30},
        ] + [
            {"player_id": i, "player_name": f"H{i}", "team": "A", "position": pos,
             "goals": 0, "attempts": 0, "assists": 0, "rebounds": 0, "feeds": 0,
             "turnovers": 0, "gains": 0, "intercepts": 0, "deflections": 0,
             "penalties": 0, "centre_pass_receives": 0, "net_points": 0}
            for i, pos in enumerate(["GA", "WA", "C", "WD", "GD", "GK"], 2)
        ]
        away_starters = [
            {"player_id": i, "player_name": f"A{i}", "team": "B", "position": pos,
             "goals": 0, "attempts": 0, "assists": 0, "rebounds": 0, "feeds": 0,
             "turnovers": 5, "gains": 0, "intercepts": 0, "deflections": 0,
             "penalties": 5, "centre_pass_receives": 0, "net_points": 0}
            for i, pos in enumerate(["GS", "GA", "WA", "C", "WD", "GD", "GK"], 8)
        ]
        pg.process_match({"match_id": "m1", "season": 2024}, home_starters, away_starters)

        # Home GS should gain rating (dominated the GS vs GK matchup)
        assert pg.get_rating(1, "GS").rating > 1500.0


class TestSeasonReset:
    def test_regress_ratings(self):
        pg = PlayerGlicko2()
        pg._ratings[(1, "GS")] = PlayerRating(rating=1700, rd=50, vol=0.06)
        pg.regress_ratings(factor=0.2, mean=1500.0)
        expected = 1700 * 0.8 + 1500 * 0.2
        assert abs(pg.get_rating(1, "GS").rating - expected) < 0.1

    def test_increase_rd(self):
        pg = PlayerGlicko2()
        pg._ratings[(1, "GS")] = PlayerRating(rating=1600, rd=50, vol=0.06)
        pg.increase_rd(amount=30)
        assert abs(pg.get_rating(1, "GS").rd - 80) < 0.1

    def test_increase_rd_capped(self):
        pg = PlayerGlicko2()
        pg._ratings[(1, "GS")] = PlayerRating(rating=1600, rd=340, vol=0.06)
        pg.increase_rd(amount=30)
        assert pg.get_rating(1, "GS").rd == 350.0
