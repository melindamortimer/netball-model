import pytest
from netball_model.features.player_rankings import (
    PlayerRankings,
    get_position_rankings,
    get_matchup_prediction,
    get_team_matchup_report,
)
from netball_model.features.player_elo import PlayerGlicko2, PlayerRating


class TestPlayerRankings:
    def _make_glicko(self):
        pg = PlayerGlicko2()
        pg._ratings[(1, "GS")] = PlayerRating(rating=1650, rd=45, vol=0.06)
        pg._ratings[(2, "GS")] = PlayerRating(rating=1550, rd=60, vol=0.06)
        pg._ratings[(3, "GK")] = PlayerRating(rating=1600, rd=50, vol=0.06)
        pg._ratings[(4, "GK")] = PlayerRating(rating=1480, rd=70, vol=0.06)
        return pg

    def _make_player_map(self):
        return {
            1: {"player_name": "Sophie Garbin", "team": "Melbourne Vixens"},
            2: {"player_name": "Jhaniele Fowler", "team": "West Coast Fever"},
            3: {"player_name": "Courtney Bruce", "team": "West Coast Fever"},
            4: {"player_name": "Geva Mentor", "team": "Melbourne Vixens"},
        }

    def test_get_position_rankings(self):
        pg = self._make_glicko()
        player_map = self._make_player_map()
        rankings = PlayerRankings(pg, player_map)
        result = rankings.get_position_rankings("GS")
        assert len(result) == 2
        assert result[0]["rank"] == 1
        assert result[0]["player_name"] == "Sophie Garbin"
        assert result[0]["rating"] == 1650
        assert result[1]["rank"] == 2

    def test_get_matchup_prediction(self):
        pg = self._make_glicko()
        player_map = self._make_player_map()
        rankings = PlayerRankings(pg, player_map)
        result = rankings.get_matchup_prediction(
            player_a_id=1, pos_a="GS", player_b_id=3, pos_b="GK"
        )
        assert result["player_a"] == "Sophie Garbin"
        assert result["player_b"] == "Courtney Bruce"
        assert result["a_win_prob"] > 0.5  # Higher rated player
        assert result["rating_diff"] == 50  # 1650 - 1600

    def test_get_team_matchup_report(self):
        pg = PlayerGlicko2()
        # Set up ratings for a full 5-position matchup
        positions_home = {"GS": 10, "GA": 11, "WA": 12, "C": 13, "WD": 14}
        positions_away = {"GK": 20, "GD": 21, "WD": 22, "C": 23, "WA": 24}
        for pos, pid in positions_home.items():
            pg._ratings[(pid, pos)] = PlayerRating(rating=1550, rd=60, vol=0.06)
        for pos, pid in positions_away.items():
            pg._ratings[(pid, pos)] = PlayerRating(rating=1500, rd=60, vol=0.06)

        home_squad = {pos: pid for pos, pid in positions_home.items()}
        away_squad = {pos: pid for pos, pid in positions_away.items()}

        player_map = {}
        for pos, pid in positions_home.items():
            player_map[pid] = {"player_name": f"Home_{pos}", "team": "Team A"}
        for pos, pid in positions_away.items():
            player_map[pid] = {"player_name": f"Away_{pos}", "team": "Team B"}

        rankings = PlayerRankings(pg, player_map)
        report = rankings.get_team_matchup_report(
            "Team A", "Team B", home_squad, away_squad
        )
        assert len(report) == 5
        for entry in report:
            assert "pair" in entry
            assert "home_player" in entry
            assert "away_player" in entry
            assert "home_win_prob" in entry


class TestModuleLevelFunctions:
    def test_get_position_rankings_no_data(self):
        pg = PlayerGlicko2()
        result = get_position_rankings(pg, {}, "GS")
        assert result == []

    def test_get_matchup_prediction_default_ratings(self):
        pg = PlayerGlicko2()
        player_map = {
            1: {"player_name": "Player A", "team": "Team A"},
            2: {"player_name": "Player B", "team": "Team B"},
        }
        result = get_matchup_prediction(pg, player_map, 1, "GS", 2, "GK")
        assert abs(result["a_win_prob"] - 0.5) < 0.01
