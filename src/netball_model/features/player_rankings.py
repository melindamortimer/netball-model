"""Named matchup rankings built on top of PlayerGlicko2 ratings."""
from __future__ import annotations

from netball_model.features.player_elo import PlayerGlicko2

# The 5 positional pairings for team matchup reports
_TEAM_PAIRINGS = [
    ("GS", "GK", "GS vs GK"),
    ("GA", "GD", "GA vs GD"),
    ("WA", "WD", "WA vs WD"),
    ("C", "C", "C vs C"),
    ("WD", "WA", "WD vs WA"),
]


class PlayerRankings:
    """Wraps PlayerGlicko2 with named player lookup for rankings and reports."""

    def __init__(self, glicko: PlayerGlicko2, player_map: dict[int, dict]):
        """
        Args:
            glicko: The PlayerGlicko2 instance with computed ratings.
            player_map: {player_id: {"player_name": ..., "team": ...}}
        """
        self._glicko = glicko
        self._player_map = player_map

    def get_position_rankings(self, position: str) -> list[dict]:
        """Return ranked list of players at a position, sorted by rating descending."""
        entries = []
        for (pid, pos), pr in self._glicko._ratings.items():
            if pos != position:
                continue
            info = self._player_map.get(pid, {})
            entries.append({
                "player_id": pid,
                "player_name": info.get("player_name", f"Player {pid}"),
                "team": info.get("team", "Unknown"),
                "rating": pr.rating,
                "rd": pr.rd,
            })
        entries.sort(key=lambda x: x["rating"], reverse=True)
        for i, e in enumerate(entries):
            e["rank"] = i + 1
        return entries

    def get_matchup_prediction(
        self, player_a_id: int, pos_a: str, player_b_id: int, pos_b: str
    ) -> dict:
        """Predict outcome of a named matchup between two players."""
        a_info = self._player_map.get(player_a_id, {})
        b_info = self._player_map.get(player_b_id, {})
        a_rating = self._glicko.get_rating(player_a_id, pos_a)
        b_rating = self._glicko.get_rating(player_b_id, pos_b)
        win_prob = self._glicko.get_matchup_prediction(
            player_a_id, pos_a, player_b_id, pos_b
        )
        return {
            "player_a": a_info.get("player_name", f"Player {player_a_id}"),
            "player_b": b_info.get("player_name", f"Player {player_b_id}"),
            "a_win_prob": win_prob,
            "rating_diff": a_rating.rating - b_rating.rating,
        }

    def get_team_matchup_report(
        self,
        home_team: str,
        away_team: str,
        home_squad: dict[str, int],
        away_squad: dict[str, int],
    ) -> list[dict]:
        """Full matchup report for all 5 positional pairs between two teams.

        Args:
            home_squad: {position: player_id} for home team
            away_squad: {position: player_id} for away team
        """
        report = []
        for home_pos, away_pos, label in _TEAM_PAIRINGS:
            home_pid = home_squad.get(home_pos)
            away_pid = away_squad.get(away_pos)
            if home_pid is None or away_pid is None:
                continue
            home_info = self._player_map.get(home_pid, {})
            away_info = self._player_map.get(away_pid, {})
            win_prob = self._glicko.get_matchup_prediction(
                home_pid, home_pos, away_pid, away_pos
            )
            home_r = self._glicko.get_rating(home_pid, home_pos)
            away_r = self._glicko.get_rating(away_pid, away_pos)
            report.append({
                "pair": label,
                "home_player": home_info.get("player_name", f"Player {home_pid}"),
                "away_player": away_info.get("player_name", f"Player {away_pid}"),
                "home_rating": home_r.rating,
                "away_rating": away_r.rating,
                "home_win_prob": win_prob,
                "rating_diff": home_r.rating - away_r.rating,
            })
        return report


# Module-level convenience functions
def get_position_rankings(
    glicko: PlayerGlicko2, player_map: dict[int, dict], position: str
) -> list[dict]:
    """Return ranked list of players at a position."""
    rankings = PlayerRankings(glicko, player_map)
    return rankings.get_position_rankings(position)


def get_matchup_prediction(
    glicko: PlayerGlicko2, player_map: dict[int, dict],
    player_a_id: int, pos_a: str, player_b_id: int, pos_b: str
) -> dict:
    """Predict outcome of a named matchup."""
    rankings = PlayerRankings(glicko, player_map)
    return rankings.get_matchup_prediction(player_a_id, pos_a, player_b_id, pos_b)


def get_team_matchup_report(
    glicko: PlayerGlicko2, player_map: dict[int, dict],
    home_team: str, away_team: str,
    home_squad: dict[str, int], away_squad: dict[str, int]
) -> list[dict]:
    """Full matchup report for all 5 positional pairs between two teams."""
    rankings = PlayerRankings(glicko, player_map)
    return rankings.get_team_matchup_report(home_team, away_team, home_squad, away_squad)
