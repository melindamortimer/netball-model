"""Backfill empty player_name values in player_stats from Champion Data API playerInfo."""
from __future__ import annotations

import asyncio
import sys

import httpx

from netball_model.data.champion_data import COMPETITION_IDS
from netball_model.data.database import Database

BASE_URL = "https://mc.championdata.com/data"


async def backfill(db_path: str = "data/netball.db"):
    db = Database(db_path)

    # Get player_ids missing names
    with db.connection() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT player_id FROM player_stats WHERE player_name = '' OR player_name IS NULL"
        )
        missing_ids = {r[0] for r in cursor.fetchall()}

    if not missing_ids:
        print("All player names already populated.")
        return

    print(f"{len(missing_ids)} player_ids missing names. Fetching from API...")

    name_map: dict[int, str] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for season in sorted(COMPETITION_IDS.keys()):
            if not missing_ids - name_map.keys():
                break

            for phase in ("regular", "finals"):
                comp_id = COMPETITION_IDS[season][phase]
                if comp_id is None:
                    continue

                print(f"  Season {season} ({phase})...", end=" ", flush=True)
                count = 0
                for round_num in range(1, 17):
                    if not missing_ids - name_map.keys():
                        break
                    round_empty = True
                    for game_num in range(1, 5):
                        url = f"{BASE_URL}/{comp_id}/{comp_id}{round_num:02d}0{game_num}.json"
                        resp = await client.get(url)
                        if resp.status_code == 404:
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        round_empty = False

                        # Extract names from playerInfo
                        try:
                            player_info = data["matchStats"]["playerInfo"]["player"]
                        except (KeyError, TypeError):
                            continue
                        for p in player_info:
                            pid = int(p["playerId"])
                            if pid in missing_ids and pid not in name_map:
                                first = p.get("firstname", "") or ""
                                last = p.get("surname", "") or ""
                                name = f"{first} {last}".strip()
                                if name:
                                    name_map[pid] = name
                                    count += 1
                    if round_empty:
                        break
                print(f"{count} new names (total: {len(name_map)}/{len(missing_ids)})")

    if not name_map:
        print("No names found from API.")
        return

    # Batch update
    print(f"\nUpdating {len(name_map)} player names in database...")
    with db.connection() as conn:
        for pid, name in name_map.items():
            conn.execute(
                "UPDATE player_stats SET player_name = ? WHERE player_id = ?",
                (name, pid),
            )
        conn.commit()

    still_missing = missing_ids - name_map.keys()
    print(f"Done. Updated: {len(name_map)}, Still missing: {len(still_missing)}")
    if still_missing:
        print(f"Missing IDs: {sorted(still_missing)}")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/netball.db"
    asyncio.run(backfill(db_path))
