# Player Matchup Feature Design

**Date:** 2026-03-09
**Status:** Approved

## Goal

Add player-level matchup features to the existing Glicko-2 + Ridge regression model to improve win probability calibration and find a betting edge against bookmaker odds.

## Current State

- **Model:** Ridge regression on team-level features (Elo, form, rest days, travel, h2h). 62% OOS accuracy, but -85% ROI on Quarter Kelly over 278 bets.
- **Player data:** 11,041 player stat records across 539 matches, 254 unique players. All 7 positions recorded. 13 stats per player per match (goals, attempts, assists, rebounds, feeds, turnovers, gains, intercepts, deflections, penalties, centre_pass_receives). Currently unused by the model.
- **Gap:** No player-level features feed the model. The bookmaker already prices in team strength — player matchup dynamics may be an under-priced signal.

## Approach: Layered Build

Two phases, each independently testable. Phase 1 is a prerequisite for Phase 2.

### Phase 1: Position Difference Scores

For each match, build rolling 5-match player profiles, then compute stat differences between matched positions.

#### Player Profile

Per player, per match context (using their last 5 matches before the current date):

**Raw rolling averages:**
- goals, attempts, assists, rebounds, feeds, turnovers, gains, intercepts, deflections, penalties, centre_pass_receives

**Derived ratios:**
- `shooting_pct` = goals / attempts (GS, GA only)
- `clean_steal_rate` = intercepts / gains (GD, GK, WD)
- `delivery_efficiency` = feeds / turnovers (WA, C, GA)

**Fallback:** If a player has < 2 matches of history, use the position-average from their team's last 5 matches.

**Lineup assumption:** For pre-match prediction, use each team's most recent starting 7. In backtesting, use actual starting lineups from data.

#### Matchup Pairs and Features

| Home | vs | Away | Key Difference Stats |
|------|------|------|----------------------|
| GS | vs | GK | shooting_pct, goals, attempts, rebounds |
| GA | vs | GD | shooting_pct, goals, assists, feeds, intercepts |
| WA | vs | WD | feeds, assists, turnovers, centre_pass_receives, deflections |
| C | vs | C | assists, feeds, turnovers, gains, intercepts |
| WD | vs | WA | gains, intercepts, deflections, penalties |

Feature format: `{matchup}_{stat}_diff` = home rolling avg - away rolling avg.

Yields ~20-25 new features alongside existing Elo/form/travel features.

### Phase 2: Player Clustering + Matchup Types

After Phase 1 is validated, discover player archetypes and encode matchup interactions.

#### Clustering Pipeline

1. Collect all player rolling profiles (computed in Phase 1).
2. Per position: PCA to 3-5 components, k-means with k=3 (validated by silhouette score).
3. Label each player with cluster ID (e.g. `GD_type_0`).
4. Name clusters after manual inspection for interpretability.

**Constraint:** Clustering must be retrained within the walk-forward loop (train only on data available at prediction time) to avoid leakage.

#### Matchup Features

- **One-hot matchup pairs:** `gs_type1_vs_gk_type2` — 9 combos per position pair, 5 pairs = 45 binary features.
- **Cluster centroid difference scores:** Continuous alternative — `centroid_stat[home_cluster] - centroid_stat[away_cluster]` per key stat.

## Data Layer Changes

### New DB Query

`get_player_history(player_id, before_date, limit=5)` — returns up to 5 most recent stat rows for a player before a given date, ordered by date descending.

### FeatureBuilder Changes

- Accept DB instance (or pre-loaded player stats dict) in addition to matches list.
- New method: `build_player_matchup_features(match_index)` — returns dict of Phase 1 features.
- Merged into `build_row()` output alongside existing features.

## Integration

- Model stays Ridge regression. Research confirms Ridge handles correlated player features well.
- Features added as additional columns in the feature matrix — no architecture change.

## Evaluation

Re-run the full walk-forward backtest + betting simulation. Compare three feature sets:

1. **Baseline** — current features only (Elo, form, travel, h2h)
2. **+ Phase 1** — add position difference scores
3. **+ Phase 2** — add cluster matchup types

### Betting Strategies

For each feature set, simulate:
- **Flat betting** — fixed stake per bet (e.g. $50 per match where model finds edge)
- **Full Kelly** — optimal growth, high variance
- **Half Kelly** — 75% of growth, half the variance
- **Quarter Kelly** — conservative, smooth curve

### Success Criteria

- Phase 1 improves OOS accuracy by >= 2 percentage points over baseline.
- Betting ROI improves (less negative or positive) on at least the 2023-2025 seasons.
- Cluster types in Phase 2 are interpretable and stable across walk-forward splits.

## Research References

- K-means player clustering is standard in NBA analytics (3-5 clusters per position).
- Difference scores are the dominant matchup feature in tennis/basketball prediction (Ridge-friendly).
- Netball KPIs: goals from centre pass, goals from gain, and intercept rate are the strongest match outcome discriminators.
- Intransitive matchup dynamics are under-priced by bookmakers (tennis research, arXiv 2510).
- Per-position clustering outperforms all-position clustering (soccer hierarchical clustering study).
