# Champion Data MC API Reference - Super Netball

Research compiled from the `superNetballR` R package source code and direct API probing.

## 1. Base URL and Endpoint Pattern

### Match Data Endpoint (Primary - No Auth Required)

```
https://mc.championdata.com/data/{comp_id}/{match_id}.json
```

**No authentication is required.** The API is publicly accessible via simple HTTP GET requests.
The R package uses `httr::GET()` with no auth headers, tokens, or cookies.

### Match ID Construction

The `match_id` is constructed from three parts concatenated together:

```
match_id = {comp_id}{round_id_padded}{game_id_padded}
```

Where:
- `comp_id`: Competition/season identifier (5 digits, e.g., `10083`)
- `round_id_padded`: Round number zero-padded to 2 digits (e.g., `01`, `02`, ..., `14`)
- `game_id_padded`: Game number within the round, zero-padded with leading `0` to 2 digits (e.g., `01`, `02`, `03`, `04`)

### URL Construction (from `downloadMatch.R`)

```python
# Python equivalent of the R function
def build_match_url(comp_id: str, round_id: int, game_id: int) -> str:
    round_str = f"{round_id:02d}"
    game_str = f"0{game_id}"  # Note: R code uses paste0("0", game_id)
    match_id = f"{comp_id}{round_str}{game_str}"
    return f"https://mc.championdata.com/data/{comp_id}/{match_id}.json"
```

### Examples

| Season | comp_id | Round | Game | URL |
|--------|---------|-------|------|-----|
| 2017   | 10083   | 1     | 1    | `https://mc.championdata.com/data/10083/100830101.json` |
| 2017   | 10083   | 5     | 2    | `https://mc.championdata.com/data/10083/100830502.json` |
| 2017 Finals | 10084 | 15  | 1    | `https://mc.championdata.com/data/10084/100841501.json` |
| 2018   | 10393   | 1     | 4    | `https://mc.championdata.com/data/10393/103930104.json` |
| 2018 Finals | 10394 | 1   | 1    | `https://mc.championdata.com/data/10394/103940101.json` |
| 2019   | 10724   | 1     | 1    | `https://mc.championdata.com/data/10724/107240101.json` |
| 2020   | 11108   | 1     | 1    | `https://mc.championdata.com/data/11108/111080101.json` |
| 2021   | 11391   | 1     | 1    | `https://mc.championdata.com/data/11391/113910101.json` |
| 2022   | 11665   | 1     | 1    | `https://mc.championdata.com/data/11665/116650101.json` |
| 2023   | 12045   | 1     | 1    | `https://mc.championdata.com/data/12045/120450101.json` |
| 2024   | 12438   | 1     | 1    | `https://mc.championdata.com/data/12438/124380101.json` |
| 2024 Finals | 12439 | 15  | 1    | `https://mc.championdata.com/data/12439/124390101.json` |
| 2025   | 12715   | 1     | 1    | `https://mc.championdata.com/data/12715/127150101.json` |
| 2026   | 12949   | 1     | 1    | `https://mc.championdata.com/data/12949/129490101.json` |

### Match Centre Web Interface

The web interface uses a different URL pattern:
```
https://mc.championdata.com/super_netball/index.html?competitionid={comp_id}&matchid={match_id}
```

## 2. Known Competition IDs

### Complete Season Mapping (All Verified via API Probing)

| Year | Regular Season (H) | Finals (F) | Preseason (P) | Notes |
|------|-------------------|------------|---------------|-------|
| 2017 | 10083 | 10084 | - | Inaugural SSN season |
| 2018 | 10393 | 10394 | - | |
| 2019 | 10724 | 10725 | - | |
| 2020 | 11108 | 11109 | - | Super Shot introduced; COVID-affected |
| 2021 | 11391 | 11392 | - | |
| 2022 | 11665 | 11666 | - | |
| 2023 | 12045 | 12046 | - | Collingwood's final season |
| 2024 | 12438 | 12439 | 12435, 12436, 12437 | Melbourne Mavericks join |
| 2025 | 12715 | 12716 | 12815, 12816, 12817 | |
| 2026 | 12949 | TBD | TBD | Season starts March 2026 |

**Key:** matchType "H" = Home & Away, "F" = Finals, "P" = Preseason

**Pattern observed:**
- 2017-2023: Finals comp_id = regular_season_comp_id + 1 (consistent)
- 2024-2025: Multiple comp_ids per season (preseason events get separate IDs)
- The comp_id numbering is NOT sequential across years -- gaps exist for other sports/competitions Champion Data manages (NRL, AFL, AFLW, ANZ Premiership, internationals, etc.)

### Finals Round Numbering

Finals use `roundNumber: 15` (continuing from the regular season) rather than resetting to 1. The game_id within finals still starts at 01.

### Season Structure

- **Regular season**: Typically 14 rounds with 4 games per round (8 teams)
- **Finals**: Round number = 15 in the JSON; matchType = "F"
- **Game numbering**: 01-04 within each round for regular season
- **Preseason**: Separate comp_ids; matchType = "P"

## 3. JSON Response Structure

The entire response is wrapped in:
```json
{
  "matchStats": {
    "jobId": 1673219655152,
    "playerStats": { ... },
    "created": "10:34:20am",
    "matchInfo": { ... },
    "playerInfo": { ... },
    "teamInfo": { ... },
    "teamStats": { ... },
    "teamPeriodStats": { ... },
    "playerPeriodStats": { ... },
    "playerSubs": { ... },
    "scoreFlow": { ... }
  }
}
```

### 3.1 matchInfo

```json
{
  "period": 4,
  "matchType": "H",
  "finalCode": "",
  "homeSquadId": 8118,
  "periodCompleted": 4,
  "finalShortCode": "",
  "venueName": "Sydney Olympic Park Sports Centre",
  "matchStatus": "complete",
  "roundNumber": 1,
  "utcStartTime": "2017-02-18T06:10:00Z",
  "venueId": 816,
  "periodSeconds": 908,
  "awaySquadId": 806,
  "venueCode": "OLY",
  "localStartTime": "2017-02-18T17:10:00+11:00",
  "matchId": 100830101,
  "matchNumber": 1
}
```

**Note:** From 2020 onwards, an additional field `"isNetball2pt": true` appears, indicating the Super Shot era.

Key fields:
- `matchStatus`: "complete", "scheduled", "inProgress"
- `matchType`: "H" = Home & Away (regular season), "F" = Finals, "P" = Preseason
- `periodCompleted`: Number of completed quarters (4 for a finished match)
- `periodSeconds`: Seconds in the last/current period
- `homeSquadId` / `awaySquadId`: Team identifiers (see team info)
- `roundNumber`: Round number (1-14 regular season, 15 for finals)
- `matchNumber`: Game number within the round (1-4 typically)

### 3.2 teamInfo

```json
{
  "team": [
    {
      "squadName": "Melbourne Vixens",
      "squadNickname": "Vixens",
      "squadCode": "VIX",
      "squadId": 804
    },
    {
      "squadName": "Queensland Firebirds",
      "squadNickname": "Firebirds",
      "squadCode": "FIR",
      "squadId": 807
    }
  ]
}
```

### Known Squad IDs

| squadId | squadName | squadCode |
|---------|-----------|-----------|
| 801     | Adelaide Thunderbirds | THU |
| 804     | Melbourne Vixens | VIX |
| 806     | Sunshine Coast Lightning | LIG |
| 807     | Queensland Firebirds | FIR |
| 808     | West Coast Fever | FEV |
| 810     | NSW Swifts | SWI |
| 8117    | Melbourne Mavericks | MAV |
| 8118    | Giants Netball | GIA |
| 8119    | Collingwood Magpies | MAG |

### 3.3 playerInfo

```json
{
  "player": [
    {
      "shortDisplayName": "Wallace, S",
      "firstname": "Sam",
      "surname": "Wallace",
      "displayName": "S.Wallace",
      "playerId": 999823
    }
  ]
}
```

### 3.4 playerStats (Per-Match Totals)

```json
{
  "player": [
    {
      "playerId": 1001944,
      "squadId": 806,
      "startingPositionCode": "GS",
      "currentPositionCode": "GS",
      "quartersPlayed": 4,
      "minutesPlayed": 60,

      // Scoring
      "goals": 34,
      "goalAttempts": 37,
      "goalMisses": 3,
      "goalAssists": 3,
      "points": 34,
      "netPoints": 0,

      // Shooting zones (from 2020+)
      "goal_from_zone1": null,
      "goal_from_zone2": null,
      "attempt_from_zone1": null,
      "attempt_from_zone2": null,
      "goal1": null,
      "goal2": null,
      "attempt1": null,
      "attempt2": null,

      // Passing / Feeding
      "feeds": 6,
      "feedWithAttempt": 0,
      "centrePassReceives": 0,
      "secondPhaseReceive": 0,

      // Turnovers
      "generalPlayTurnovers": 6,
      "badPasses": 3,
      "badHands": 0,
      "interceptPassThrown": 0,
      "missedGoalTurnover": 3,
      "unforcedTurnovers": null,

      // Defence
      "gain": 0,
      "intercepts": 0,
      "deflectionWithGain": 0,
      "deflectionWithNoGain": 0,
      "deflectionPossessionGain": 0,
      "rebounds": 2,
      "pickups": 0,
      "blocks": 0,
      "blocked": 0,

      // Penalties
      "penalties": 3,
      "contactPenalties": 3,
      "obstructionPenalties": 0,

      // Other
      "possessionChanges": 9,
      "tossUpWin": 0,
      "breaks": 0,
      "offsides": 0,
      "turnoverHeld": 0,

      // Percentages
      "centrePassToGoalPerc": 0,
      "gainToGoalPerc": 0
    }
  ]
}
```

**Evolution over seasons:**
- **2017-2019**: Basic stat fields (goals, goalAttempts, goalMisses, etc.)
- **2020+** (Super Shot era): Added `goal_from_zone1`, `goal_from_zone2`, `attempt_from_zone1`, `attempt_from_zone2`, `goal1`, `goal2`, `attempt1`, `attempt2`, `attempts1`, `attempts2`, `goals1`, `goals2`
- **2020+**: `scorepoints` in scoreFlow can be `2` for Super Shots; `scoreName` includes `"2pt Goal"` and `"2pt Miss"`
- **2022+**: Some field name variations (`attempts2`/`attempt2`, `goals2`/`goal2`)
- **All seasons**: `netPoints` field (Champion Data's proprietary metric)

### 3.5 teamPeriodStats (Per-Quarter Team Stats)

```json
{
  "team": [
    {
      "period": 1,
      "squadId": 804,
      "rebounds": 0,
      "goalsFromCentrePass": 12,
      "turnoverHeld": 2,
      "netPoints": 66.5,
      "goalsFromTurnovers": 1,
      "centrePassToGoalPerc": 71,
      "turnoverToGoalPerc": 20,
      "penalties": 12,
      "generalPlayTurnovers": 7,
      "deflectionWithNoGain": 1,
      "interceptPassThrown": 1,
      "gain": 3,
      "points": 15,
      "timeInPossession": 51,
      "goalMisses": 0,
      "blocked": 0,
      "deflectionWithGain": 0,
      "goalAssists": 14,
      "tossUpWin": 0,
      "feeds": 17,
      "centrePassReceives": 12,
      "obstructionPenalties": 5,
      "goals": 16,
      "offsides": 0,
      "missedGoalTurnover": 0,
      "deflectionPossessionGain": 1,
      "contactPenalties": 8,
      "gainToGoalPerc": 50,
      "possessionChanges": 6,
      "goalAttempts": 18,
      "feedWithAttempt": 15,
      "pickups": 0,
      "intercepts": 0
    }
  ]
}
```

Additional team-level fields (from 2020+):
- `goalsFromGain`, `possessions`, `disposals`, `missedShotConversion`
- `goal_from_zone2`, `goal_from_zone1`, `attempt_from_zone2`, `attempt_from_zone1`
- `goal2`, `goal1`, `attempt2`, `attempt1`

### 3.6 scoreFlow

```json
{
  "score": [
    {
      "period": 1,
      "distanceCode": 0,
      "scorepoints": 1,
      "periodSeconds": 30,
      "positionCode": 0,
      "squadId": 8119,
      "playerId": 80052,
      "scoreName": "goal"
    },
    {
      "period": 1,
      "distanceCode": 0,
      "scorepoints": 0,
      "periodSeconds": 56,
      "positionCode": 1,
      "squadId": 807,
      "playerId": 80078,
      "scoreName": "miss"
    }
  ]
}
```

**Score names (pre-2020):** `"goal"`, `"miss"`
**Score names (2020+):** `"goal"`, `"miss"`, `"2pt Goal"`, `"2pt Miss"`

**scorepoints values:**
- `1` = standard goal
- `0` = miss
- `2` = Super Shot goal (2020+)

**distanceCode values:**
- `0` = close range
- `1` = medium range
- `3` = long range (observed values; mapping not fully confirmed)

**positionCode values:** Appears to indicate the shooting position on court (0, 1, 2, 3 observed).

### 3.7 playerSubs

Empty string (`""`) if no subs, otherwise:

```json
{
  "player": [
    {
      "period": 2,
      "periodSeconds": 372,
      "fromPos": "GA",
      "squadId": 804,
      "playerId": 995218,
      "toPos": "S"
    },
    {
      "period": 2,
      "periodSeconds": 372,
      "fromPos": "S",
      "squadId": 804,
      "playerId": 1019167,
      "toPos": "GA"
    }
  ]
}
```

Position codes: `"GS"`, `"GA"`, `"WA"`, `"C"`, `"WD"`, `"GD"`, `"GK"`, `"S"` (substitute/bench)

### 3.8 playerPeriodStats

Same structure as playerStats but includes a `period` field for each player-period combination. Contains stats broken down per quarter.

## 4. Authentication

**None required.** The API is completely open. The R package uses a simple `httr::GET()` call with no authentication parameters. The equivalent Python call would be:

```python
import requests
response = requests.get(url)
data = response.json()
match_data = data["matchStats"]
```

## 5. Rate Limiting / Access Notes

- No documented rate limiting, but be respectful of the service
- JSON files are static once a match is complete
- Live matches may have partial data that updates
- Some very old or invalid comp_id/match_id combinations return 404

## 6. Known Limitations

1. **Competition ID discovery**: No known API endpoint to list available competitions. IDs must be discovered through the Match Centre web interface or by probing ranges. The comp_id table in Section 2 provides all verified IDs for 2017-2026.
2. **No fixture endpoint**: The API does not provide a fixture/schedule listing. You need to iterate through round/game combinations (rounds 1-14, games 1-4) and handle 404s for invalid combinations.
3. **Finals structure**: Finals use roundNumber=15 in the JSON but are in a separate comp_id. The number of games varies (typically: 2 semis as games 01-02, 1 preliminary final, 1 grand final).
4. **Season variation**: The number of rounds varies by season (typically 14 for regular season; COVID-affected 2020 was different).
5. **Static JSON**: Match data JSON files appear to be pre-generated static files. They are created/updated during live matches and become permanent after completion.

## 7. International & Other Netball Competition IDs

### Discovered via API Probing (comp_id range 12100-12500)

These are non-SSN competitions found on the same Champion Data API. Useful for international match data.

| comp_id | Date | Description | Teams (sample) |
|---------|------|-------------|----------------|
| 12115 | 2023-07-28 | 2023 Netball World Cup | NZ vs Trinidad & Tobago |
| 12116 | 2023-08-04 | 2023 Netball World Cup | Singapore vs Sri Lanka |
| 12117 | 2023-01-25 | International Test | England vs South Africa |
| 12168 | 2023-01-19 | International Test | South Africa vs Silver Ferns |
| 12195 | 2023-02-11 | All-Stars | Maori All-Stars vs Indigenous All-Stars |
| 12206 | 2023-05-31 | State of Origin Netball | Queensland vs NSW |
| 12255 | 2023-10-12 | Constellation Cup | Australia vs Silver Ferns |
| 12305 | 2023-09-24 | International Test | Silver Ferns vs England |
| 12316 | 2023-07-24 | 2023 Netball World Cup | Australia vs Tonga |
| 12355 | 2023-10-12 | Constellation Cup | Australia vs New Zealand |
| 12356 | 2023-10-25 | International Test | Australia vs South Africa |
| 12375 | 2023-11-11 | Quad Series | Australia vs Jamaica |
| 12376 | 2023-11-12 | Quad Series | Jamaica vs Malawi |
| 12377 | 2023-10-14 | International Test | Australia vs Samoa |
| 12416 | 2023-11-11 | Quad Series | Australia vs South Africa |
| 12417 | 2023-11-12 | Quad Series | New Zealand vs Australia |
| 12465 | 2024-01-20 | International Test | Australia vs Silver Ferns |
| 12466 | 2024-01-28 | International Test | Silver Ferns vs Uganda |

### NZ ANZ Premiership (also on Champion Data)

| comp_id | Date | Description |
|---------|------|-------------|
| 12105 | 2023-03-11 | NZ Premiership (Waikato BoP vs Northern Comets) |
| 12106 | 2023-05-14 | NZ Premiership (Central Manawa vs Northern Comets) |
| 12427 | 2024-04-13 | NZ Premiership (WBOP Magic vs Northern Mystics) |
| 12428 | 2024-07-27 | NZ Premiership (Mainland Tactix vs Northern Mystics) |

### SSN Preseason & Invitational

| comp_id | Date | Description |
|---------|------|-------------|
| 12125 | 2023-02-24 | SSN Preseason (NSW Swifts vs West Coast Fever) |
| 12126 | 2023-02-26 | SSN Preseason (Collingwood Magpies vs Queensland Firebirds) |
| 12205 | 2023-02-24 | SSN Preseason (Melbourne Vixens vs GIANTS Netball) |
| 12326 | 2023-08-21 | SSN Futures (Thunderbirds Futures vs Lightning Bolts) |
| 12327 | 2023-08-26 | SSN Futures (Capital Darters vs Territory Storm) |

## 8. Other Champion Data Netball URL Paths

The Match Centre at `mc.championdata.com` also serves data for:
- **ANZ Premiership (NZ)**: `https://mc.championdata.com/netball_nz/`
- **Netball Australia internationals**: `https://mc.championdata.com/netball_aus/`

These likely follow similar JSON patterns.

## 9. Existing Python Implementations

**No Python implementations were found** that directly call these Champion Data netball API endpoints.
- The `superNetballR` R package (by Steve Lane) is the only known open-source client
- Aaron Fox (Deakin University) uses Champion Data for analysis but his code primarily uses R
- The `nrlR` R package on CRAN calls similar Champion Data endpoints for NRL rugby league
- Champion Data has an official AFL API with authentication (`docs.api.afl.championdata.com`) -- this is a separate, more formal API for AFL, not the public MC endpoint used for netball

## 10. Strategy for Finding Future Competition IDs

To discover new season competition IDs programmatically:

1. **Probe ranges**: Scan comp_id ranges above the last known ID (e.g., for 2027, start scanning from 13000+). Use HTTP status 200 to detect valid IDs, then check `matchInfo.localStartTime` and team names to identify SSN matches.
2. **Scrape the Match Centre page**: Load `https://mc.championdata.com/super_netball/index.html` and inspect the JavaScript source or network requests for competition ID references in dropdown menus.
3. **Use the match_id in the URL**: When viewing a match on the Match Centre, the URL contains both `competitionid` and `matchid` parameters.
4. **Note**: Champion Data uses sequential comp_ids across ALL sports they manage (AFL, NRL, netball, etc.), so there are large gaps between SSN season IDs.

### Discovery Script Pattern

```python
import requests

def find_ssn_comp_ids(start: int, end: int) -> list:
    """Scan a range of comp_ids to find Super Netball matches."""
    ssn_teams = {
        "Adelaide Thunderbirds", "Melbourne Vixens", "Queensland Firebirds",
        "West Coast Fever", "Sunshine Coast Lightning", "NSW Swifts",
        "GIANTS Netball", "Collingwood Magpies", "Melbourne Mavericks"
    }
    found = []
    for comp_id in range(start, end + 1):
        url = f"https://mc.championdata.com/data/{comp_id}/{comp_id}0101.json"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            teams = data["matchStats"]["teamInfo"]["team"]
            team_names = {t["squadName"] for t in teams}
            if team_names & ssn_teams:
                mi = data["matchStats"]["matchInfo"]
                found.append({
                    "comp_id": comp_id,
                    "date": mi["localStartTime"][:10],
                    "match_type": mi["matchType"],
                    "teams": [t["squadName"] for t in teams]
                })
    return found
```
