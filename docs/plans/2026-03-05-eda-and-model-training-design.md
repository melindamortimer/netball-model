# EDA & Model Training — Design Document

**Date:** 2026-03-05
**Status:** Approved

## Goal

Ingest real match data, perform exploratory data analysis, adjust the model based on findings, and run the first real training + backtest cycle.

## Data Ingestion

Ingest all available SSN seasons (2017-2025) from Champion Data via the existing `netball ingest` CLI command. Expected yield: ~550-630 matches and ~8,000+ player stat records.

Post-ingestion validation:
- Match count per season (~60-70 each)
- No NULL scores on completed matches
- All 8 teams appear each season
- Dates are chronologically ordered

## EDA Notebook

**File:** `notebooks/eda.ipynb`

**Dependency:** Add `matplotlib` to dev dependencies.

### Sections

1. **Data Quality** — Match counts per season, missing values, score ranges, team name consistency across seasons (catch renames like Collingwood Magpies → Melbourne Mavericks).

2. **Score Distributions** — Histogram of margins (home - away). Check normality assumption used by the calibration model. Histogram of total goals. Summary stats (mean, median, std).

3. **Home Advantage** — Home win rate overall and per season. Average home margin. Quantify significance.

4. **Scoring Trends** — Average total goals per season. Check for temporal drift that might need season-adjusted features.

5. **Feature Correlations** — Build the feature matrix using `FeatureBuilder`, compute correlations with the margin target. Identify most predictive features and check for multicollinearity.

6. **Glicko-2 Rating Trajectories** — Plot rating evolution for each team over time. Verify RD decreases and ratings stabilize after burn-in.

7. **Venue Coverage** — Check which venues appear and how many map to unknown cities (travel distance = 0). Identify any venues needing addition to `VENUE_TO_CITY`.

## Potential Model Adjustments

Based on EDA findings, apply only what's warranted:

- **Venue mapping gaps:** Update `VENUE_TO_CITY` dict for unmapped venues
- **Team name changes:** Add normalization for renamed teams across seasons
- **Non-normal margins:** If skewed/fat-tailed, switch calibration from normal to t-distribution
- **Weak features:** Drop features with near-zero correlation to reduce noise
- **Scoring trends:** If significant temporal drift, add season-level normalization

## Training & Evaluation

- **Train:** `netball train --db data/netball.db` on all ingested data
- **Backtest:** `netball backtest --train-seasons 2017-2023 --test-season 2024`

### Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Win/Loss Accuracy | > 55% | Better than coin flip |
| Mean Absolute Error | < 12 goals | Typical netball margin variability |

## Approach

Sequential, using existing infrastructure:
1. Ingest via CLI (no code changes)
2. EDA in Jupyter notebook (new file)
3. Targeted code adjustments if EDA warrants
4. Train and backtest via CLI (no code changes)
