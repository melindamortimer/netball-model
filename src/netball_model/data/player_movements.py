"""Track player movements between seasons for roster continuity."""
from __future__ import annotations

import difflib
import sys

from netball_model.data.database import Database


def get_player_movements(season: int, db: Database) -> list[dict]:
    """Detect player movements into a given season."""
    prev_season = season - 1
    current_matches = db.get_matches(season=season)
    prev_matches = db.get_matches(season=prev_season)

    # Build player->team maps for each season
    prev_teams = _build_player_team_map(prev_matches, db)
    curr_teams = _build_player_team_map(current_matches, db)

    # For 2026, supplement with hardcoded squad data
    if season == 2026 and not curr_teams:
        from netball_model.data.squads import get_all_squads
        squads = get_all_squads(2026)
        for team, positions in squads.items():
            for pos, name in positions.items():
                pid = _fuzzy_match_player_id(name, prev_matches, db)
                if pid is not None:
                    curr_teams[pid] = team
                # Skip players with no DB match — they are new to the league

    movements = []
    all_players = set(prev_teams.keys()) | set(curr_teams.keys())

    for player in all_players:
        prev_team = prev_teams.get(player)
        curr_team = curr_teams.get(player)

        if prev_team and curr_team:
            movement_type = "stayed" if prev_team == curr_team else "moved"
        elif curr_team and not prev_team:
            movement_type = "new"
        else:
            movement_type = "retired"

        movements.append({
            "player_name": str(player),
            "player_id": player if isinstance(player, int) else None,
            "from_team": prev_team,
            "to_team": curr_team,
            "movement_type": movement_type,
        })

    return movements


def get_roster_continuity(team: str, season: int, db: Database) -> float:
    """Fraction of current season's starters who played for same team last season."""
    prev_season = season - 1
    current_matches = db.get_matches(season=season)
    prev_matches = db.get_matches(season=prev_season)

    curr_players = set()
    for m in current_matches:
        starters = db.get_starters_for_match(m["match_id"])
        for s in starters:
            if s["team"] == team:
                curr_players.add(s["player_id"])

    # For 2026 with no matches yet, use squad data
    if not curr_players and season == 2026:
        from netball_model.data.squads import get_squad
        try:
            squad = get_squad(team, 2026)
            # Map names to IDs from previous season
            for name in squad.values():
                pid = _find_player_id_by_name(name, prev_matches, db)
                if pid:
                    curr_players.add(pid)
        except (KeyError, ValueError):
            pass

    if not curr_players:
        return 0.0

    prev_players_on_team = set()
    for m in prev_matches:
        starters = db.get_starters_for_match(m["match_id"])
        for s in starters:
            if s["team"] == team:
                prev_players_on_team.add(s["player_id"])

    retained = curr_players & prev_players_on_team
    return len(retained) / len(curr_players) if curr_players else 0.0


def get_team_continuity_all(season: int, db: Database) -> dict[str, float]:
    """Return roster continuity for all teams in a season."""
    matches = db.get_matches(season=season)
    teams = set()
    for m in matches:
        teams.add(m["home_team"])
        teams.add(m["away_team"])

    # For 2026, also include teams from squad data
    if season == 2026:
        from netball_model.data.squads import get_all_squads
        teams.update(get_all_squads(2026).keys())

    return {team: get_roster_continuity(team, season, db) for team in teams}


def _build_player_team_map(matches: list[dict], db: Database) -> dict[int, str]:
    """Map player_id -> most recent team from match list."""
    player_teams: dict[int, str] = {}
    for m in matches:
        starters = db.get_starters_for_match(m["match_id"])
        for s in starters:
            player_teams[s["player_id"]] = s["team"]
    return player_teams


def _find_player_id_by_name(name: str, matches: list[dict], db: Database) -> int | None:
    """Find player_id by fuzzy name match against player_stats."""
    best_id = None
    best_ratio = 0.0
    seen = set()

    for m in matches:
        stats = db.get_starters_for_match(m["match_id"])
        for s in stats:
            if s["player_id"] in seen:
                continue
            seen.add(s["player_id"])
            ratio = difflib.SequenceMatcher(None, name.lower(), s["player_name"].lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = s["player_id"]

    if best_ratio < 0.85:
        print(f"Warning: No match for '{name}' (best: {best_ratio:.2f})", file=sys.stderr)
        return None

    return best_id


def _fuzzy_match_player_id(name: str, prev_matches: list[dict], db: Database) -> int | None:
    """Try to match a player name to an existing player_id via fuzzy search."""
    return _find_player_id_by_name(name, prev_matches, db)
