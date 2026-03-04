# EDA & Model Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ingest real SSN match data (2017-2025), perform exploratory data analysis, apply any warranted model adjustments, and run the first real training + backtest cycle.

**Architecture:** Use the existing CLI for ingestion/training/backtest. EDA lives in a Jupyter notebook (`notebooks/eda.ipynb`) that loads data from SQLite, analyzes distributions and features, and flags issues. Adjustments to model code are targeted and conditional on EDA findings.

**Tech Stack:** Existing stack + matplotlib (dev dependency for EDA charts).

**Reference docs:**
- Design: `docs/plans/2026-03-05-eda-and-model-training-design.md`
- Original design: `docs/plans/2026-02-26-netball-betting-model-design.md`

---

## Task 1: Add matplotlib Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add matplotlib to dev dependencies**

Add `matplotlib = "^3.8"` under `[tool.poetry.group.dev.dependencies]` in `pyproject.toml`.

**Step 2: Install**

```bash
poetry add --group dev matplotlib
```

**Step 3: Verify installation**

```bash
poetry run python -c "import matplotlib; print(matplotlib.__version__)"
```
Expected: Prints a version number (e.g. `3.8.x` or `3.9.x`).

**Step 4: Commit**

```
chore: add matplotlib dev dependency for EDA
```

---

## Task 2: Ingest All Seasons

**Files:**
- No code changes. Uses existing `netball ingest` CLI command.

**Context:** The Champion Data API is public (no auth). Each season takes ~30-60 seconds (many HTTP requests). The `fetch_season` method iterates rounds 1-17 with up to 4 games each, stopping when a round returns no data.

**Step 1: Create data directory if needed**

```bash
mkdir -p data
```

**Step 2: Ingest each season**

Run each command sequentially. Each takes ~30-60 seconds:

```bash
poetry run netball ingest --season 2017 --db data/netball.db
poetry run netball ingest --season 2018 --db data/netball.db
poetry run netball ingest --season 2019 --db data/netball.db
poetry run netball ingest --season 2020 --db data/netball.db
poetry run netball ingest --season 2021 --db data/netball.db
poetry run netball ingest --season 2022 --db data/netball.db
poetry run netball ingest --season 2023 --db data/netball.db
poetry run netball ingest --season 2024 --db data/netball.db
poetry run netball ingest --season 2025 --db data/netball.db
```

Expected output per season: `Ingested NN matches for SSN YYYY.` where NN is ~56-68.

**Step 3: Quick validation**

```bash
poetry run python -c "
from netball_model.data.database import Database
db = Database('data/netball.db')
for season in range(2017, 2026):
    rows = db.get_matches(season=season)
    print(f'{season}: {len(rows)} matches')
total = db.get_matches()
print(f'Total: {len(total)} matches')
"
```

Expected: Each season has 50-70 matches. Total is 500-630.

**Step 4: No commit** (data files are gitignored)

---

## Task 3: EDA Notebook — Data Quality

**Files:**
- Create: `notebooks/eda.ipynb`

**Step 1: Create the notebook with setup and data quality cells**

Create a Jupyter notebook at `notebooks/eda.ipynb` with these cells:

**Cell 1 (markdown):**
```markdown
# Netball Betting Model — Exploratory Data Analysis

Analyzing SSN match data (2017-2025) to validate data quality, understand distributions, and check model assumptions before training.
```

**Cell 2 (code) — Setup:**
```python
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data from SQLite
DB_PATH = Path("../data/netball.db")
conn = sqlite3.connect(DB_PATH)

matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
player_stats = pd.read_sql("SELECT * FROM player_stats", conn)
conn.close()

print(f"Matches: {len(matches)}")
print(f"Player stat rows: {len(player_stats)}")
print(f"Seasons: {sorted(matches['season'].unique())}")
```

**Cell 3 (markdown):**
```markdown
## 1. Data Quality
```

**Cell 4 (code) — Match counts per season:**
```python
season_counts = matches.groupby("season").size().reset_index(name="match_count")
print(season_counts.to_string(index=False))
print(f"\nTotal matches: {len(matches)}")
```

**Cell 5 (code) — Missing values:**
```python
print("=== Missing values in matches ===")
missing = matches.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values")

print("\n=== Score ranges ===")
print(f"Home scores: {matches['home_score'].min()} - {matches['home_score'].max()}")
print(f"Away scores: {matches['away_score'].min()} - {matches['away_score'].max()}")

# Check for zero-score matches (might indicate incomplete data)
zero_scores = matches[(matches["home_score"] == 0) | (matches["away_score"] == 0)]
if len(zero_scores) > 0:
    print(f"\nWARNING: {len(zero_scores)} matches with zero scores:")
    print(zero_scores[["match_id", "date", "home_team", "away_team", "home_score", "away_score"]])
else:
    print("\nNo zero-score matches — good.")
```

**Cell 6 (code) — Team name consistency:**
```python
print("=== Teams per season ===")
for season in sorted(matches["season"].unique()):
    season_matches = matches[matches["season"] == season]
    teams = sorted(set(season_matches["home_team"].unique()) | set(season_matches["away_team"].unique()))
    print(f"\n{season} ({len(teams)} teams): {', '.join(teams)}")
```

**Cell 7 (code) — Date ordering check:**
```python
print("=== Date ordering check ===")
for season in sorted(matches["season"].unique()):
    season_matches = matches[matches["season"] == season].sort_values("round_num")
    dates = pd.to_datetime(season_matches["date"].str[:10], errors="coerce")
    out_of_order = (dates.diff().dt.days < 0).sum()
    if out_of_order > 0:
        print(f"WARNING: {season} has {out_of_order} dates out of order")
    else:
        print(f"{season}: dates in order ✓")
```

**Step 2: Run the notebook**

```bash
poetry run jupyter execute notebooks/eda.ipynb 2>/dev/null || echo "Run notebook manually if jupyter not installed"
```

Alternatively, open the notebook in VS Code / Jupyter and run all cells. Verify:
- Each season has 50-70 matches
- No unexpected missing values
- Team names are consistent (note: Collingwood Magpies and Melbourne Mavericks are expected to be different teams — the Magpies rebranded to Mavericks in 2024)

**Step 3: No commit yet** (notebook will be committed after all EDA sections are complete)

---

## Task 4: EDA Notebook — Score Distributions & Home Advantage

**Files:**
- Modify: `notebooks/eda.ipynb` (append cells)

**Step 1: Add score distribution cells**

**Cell 8 (markdown):**
```markdown
## 2. Score Distributions
```

**Cell 9 (code) — Margin distribution:**
```python
matches["margin"] = matches["home_score"] - matches["away_score"]
matches["total_goals"] = matches["home_score"] + matches["away_score"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Margin histogram
axes[0].hist(matches["margin"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
axes[0].axvline(0, color="red", linestyle="--", label="Draw line")
axes[0].set_xlabel("Margin (Home - Away)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Match Margin Distribution")
axes[0].legend()

# Total goals histogram
axes[1].hist(matches["total_goals"], bins=30, edgecolor="black", alpha=0.7, color="coral")
axes[1].set_xlabel("Total Goals")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Total Goals Distribution")

plt.tight_layout()
plt.savefig("../notebooks/margin_distribution.png", dpi=100, bbox_inches="tight")
plt.show()

print(f"Margin — mean: {matches['margin'].mean():.1f}, median: {matches['margin'].median():.1f}, std: {matches['margin'].std():.1f}")
print(f"Total  — mean: {matches['total_goals'].mean():.1f}, median: {matches['total_goals'].median():.1f}, std: {matches['total_goals'].std():.1f}")
```

**Cell 10 (code) — Normality check for margin:**
```python
from scipy import stats

# Shapiro-Wilk test (on a sample if >5000 rows)
sample = matches["margin"].sample(min(len(matches), 5000), random_state=42)
stat, p_value = stats.shapiro(sample)
print(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4f}")
if p_value > 0.05:
    print("-> Margin distribution is approximately normal (p > 0.05)")
else:
    print("-> Margin distribution deviates from normal (p <= 0.05)")
    print("   Consider: skewness and kurtosis may affect calibration model")

print(f"\nSkewness: {matches['margin'].skew():.3f}")
print(f"Kurtosis: {matches['margin'].kurtosis():.3f}")
```

**Cell 11 (markdown):**
```markdown
## 3. Home Advantage
```

**Cell 12 (code) — Home advantage analysis:**
```python
home_wins = (matches["margin"] > 0).sum()
away_wins = (matches["margin"] < 0).sum()
draws = (matches["margin"] == 0).sum()

print(f"Overall: {home_wins} home wins ({home_wins/len(matches):.1%}), "
      f"{away_wins} away wins ({away_wins/len(matches):.1%}), "
      f"{draws} draws ({draws/len(matches):.1%})")
print(f"Average home margin: {matches['margin'].mean():+.1f} goals")

# Per season
print("\n=== Home advantage by season ===")
season_ha = matches.groupby("season").agg(
    home_win_rate=("margin", lambda x: (x > 0).mean()),
    avg_margin=("margin", "mean"),
    n_matches=("margin", "size"),
).reset_index()
print(season_ha.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(season_ha["season"], season_ha["home_win_rate"], color="steelblue", alpha=0.7)
ax.axhline(0.5, color="red", linestyle="--", label="50% (no advantage)")
ax.set_xlabel("Season")
ax.set_ylabel("Home Win Rate")
ax.set_title("Home Advantage by Season")
ax.legend()
ax.set_ylim(0.3, 0.7)
plt.tight_layout()
plt.savefig("../notebooks/home_advantage.png", dpi=100, bbox_inches="tight")
plt.show()
```

**Step 2: Run the new cells and check:**
- Is the margin distribution roughly normal or heavily skewed?
- Is home advantage consistent across seasons?

---

## Task 5: EDA Notebook — Scoring Trends & Feature Correlations

**Files:**
- Modify: `notebooks/eda.ipynb` (append cells)

**Step 1: Add scoring trends and feature analysis cells**

**Cell 13 (markdown):**
```markdown
## 4. Scoring Trends
```

**Cell 14 (code) — Scoring trends over time:**
```python
season_scoring = matches.groupby("season").agg(
    avg_total=("total_goals", "mean"),
    avg_home=("home_score", "mean"),
    avg_away=("away_score", "mean"),
    std_total=("total_goals", "std"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(season_scoring["season"], season_scoring["avg_total"], "o-", color="steelblue", label="Avg Total Goals")
ax.fill_between(
    season_scoring["season"],
    season_scoring["avg_total"] - season_scoring["std_total"],
    season_scoring["avg_total"] + season_scoring["std_total"],
    alpha=0.2,
)
ax.set_xlabel("Season")
ax.set_ylabel("Goals")
ax.set_title("Average Total Goals per Season")
ax.legend()
plt.tight_layout()
plt.savefig("../notebooks/scoring_trends.png", dpi=100, bbox_inches="tight")
plt.show()

print(season_scoring.to_string(index=False))

# Check if there's a significant trend
from scipy.stats import linregress
slope, intercept, r, p, se = linregress(season_scoring["season"], season_scoring["avg_total"])
print(f"\nLinear trend: slope={slope:.2f} goals/year, R²={r**2:.3f}, p={p:.4f}")
if p < 0.05:
    print("-> Significant scoring trend detected. Consider season normalization.")
else:
    print("-> No significant scoring trend.")
```

**Cell 15 (markdown):**
```markdown
## 5. Feature Correlations
```

**Cell 16 (code) — Build feature matrix and check correlations:**
```python
import sys
sys.path.insert(0, str(Path("../src").resolve()))
from netball_model.features.builder import FeatureBuilder

# Build feature matrix from all matches
match_dicts = matches.to_dict("records")
builder = FeatureBuilder(match_dicts)
features_df = builder.build_matrix(start_index=1)

print(f"Feature matrix: {features_df.shape[0]} rows × {features_df.shape[1]} columns")
print(f"\nColumns: {list(features_df.columns)}")

# Correlations with margin
non_feature = {"match_id", "home_team", "away_team", "margin", "total_goals"}
feature_cols = [c for c in features_df.columns if c not in non_feature]

correlations = features_df[feature_cols].corrwith(features_df["margin"]).sort_values(ascending=False)
print("\n=== Feature correlations with margin ===")
print(correlations.to_string())
```

**Cell 17 (code) — Correlation heatmap:**
```python
fig, ax = plt.subplots(figsize=(12, 8))
corr_matrix = features_df[feature_cols].corr()
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(feature_cols, fontsize=8)
plt.colorbar(im)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("../notebooks/feature_correlations.png", dpi=100, bbox_inches="tight")
plt.show()

# Flag highly correlated feature pairs (|r| > 0.8)
print("\n=== Highly correlated feature pairs (|r| > 0.8) ===")
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.8:
            print(f"  {feature_cols[i]} <-> {feature_cols[j]}: r={r:.3f}")
```

**Step 2: Run the new cells and note:**
- Which features correlate most strongly with margin?
- Are there highly multicollinear feature pairs that could cause issues?
- Is there a significant scoring trend across seasons?

---

## Task 6: EDA Notebook — Glicko-2 Trajectories & Venue Coverage

**Files:**
- Modify: `notebooks/eda.ipynb` (append cells)

**Step 1: Add Glicko-2 and venue coverage cells**

**Cell 18 (markdown):**
```markdown
## 6. Glicko-2 Rating Trajectories
```

**Cell 19 (code) — Track Glicko-2 ratings over time:**
```python
from netball_model.features.elo import GlickoSystem

glicko = GlickoSystem()
rating_history = []

for i, row in enumerate(match_dicts):
    hs = row.get("home_score", 0)
    as_ = row.get("away_score", 0)
    if hs is None or as_ is None:
        continue

    # Record pre-match ratings
    home_r = glicko.get_rating(row["home_team"])
    away_r = glicko.get_rating(row["away_team"])
    rating_history.append({
        "match_index": i,
        "date": row["date"],
        "season": row["season"],
        "team": row["home_team"],
        "rating": home_r["rating"],
        "rd": home_r["rd"],
    })
    rating_history.append({
        "match_index": i,
        "date": row["date"],
        "season": row["season"],
        "team": row["away_team"],
        "rating": away_r["rating"],
        "rd": away_r["rd"],
    })

    # Update after match
    if hs > as_:
        winner = "home"
    elif as_ > hs:
        winner = "away"
    else:
        winner = "draw"
    glicko.update(row["home_team"], row["away_team"], winner=winner, margin=hs - as_)

ratings_df = pd.DataFrame(rating_history)

# Plot rating trajectories
teams = ratings_df["team"].unique()
fig, ax = plt.subplots(figsize=(14, 7))
for team in sorted(teams):
    team_data = ratings_df[ratings_df["team"] == team]
    ax.plot(team_data["match_index"], team_data["rating"], label=team, alpha=0.7)

ax.axhline(1500, color="gray", linestyle="--", alpha=0.5, label="Starting rating")
ax.set_xlabel("Match Index (chronological)")
ax.set_ylabel("Glicko-2 Rating")
ax.set_title("Team Rating Trajectories (2017-2025)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("../notebooks/glicko_trajectories.png", dpi=100, bbox_inches="tight")
plt.show()

# Final ratings
print("\n=== Final Glicko-2 Ratings ===")
final_ratings = glicko.get_all_ratings()
for team, r in sorted(final_ratings.items(), key=lambda x: -x[1]["rating"]):
    print(f"  {team:30s} rating={r['rating']:7.1f}  rd={r['rd']:.1f}  vol={r['vol']:.4f}")
```

**Cell 20 (code) — RD convergence check:**
```python
# Check that rating deviation decreases over time
fig, ax = plt.subplots(figsize=(14, 5))
for team in sorted(teams):
    team_data = ratings_df[ratings_df["team"] == team]
    ax.plot(team_data["match_index"], team_data["rd"], alpha=0.5)

ax.set_xlabel("Match Index (chronological)")
ax.set_ylabel("Rating Deviation (RD)")
ax.set_title("Rating Deviation Convergence")
plt.tight_layout()
plt.savefig("../notebooks/rd_convergence.png", dpi=100, bbox_inches="tight")
plt.show()

print(f"Starting RD: 350.0")
avg_final_rd = np.mean([r["rd"] for r in final_ratings.values()])
print(f"Average final RD: {avg_final_rd:.1f}")
```

**Cell 21 (markdown):**
```markdown
## 7. Venue Coverage
```

**Cell 22 (code) — Venue mapping check:**
```python
from netball_model.features.contextual import VENUE_TO_CITY

venues = matches["venue"].value_counts()
print(f"=== {len(venues)} unique venues ===\n")

mapped = 0
unmapped_venues = []
for venue, count in venues.items():
    city = VENUE_TO_CITY.get(venue, None)
    status = city if city else "*** UNMAPPED ***"
    if city:
        mapped += 1
    else:
        unmapped_venues.append((venue, count))
    print(f"  {venue:45s} ({count:3d} matches) -> {status}")

print(f"\nMapped: {mapped}/{len(venues)} venues")
if unmapped_venues:
    print(f"\n=== UNMAPPED VENUES (need adding to VENUE_TO_CITY) ===")
    for venue, count in unmapped_venues:
        print(f"  {venue} ({count} matches)")
```

**Cell 23 (markdown):**
```markdown
## Summary

Key findings to inform model adjustments:

1. **Data quality:** [fill after running]
2. **Margin distribution:** [normal/skewed?]
3. **Home advantage:** [significant/not?]
4. **Scoring trends:** [stable/shifting?]
5. **Top features:** [which correlate most?]
6. **Multicollinearity:** [any issues?]
7. **Glicko-2 burn-in:** [how many matches before stable?]
8. **Venue gaps:** [how many unmapped?]
```

**Step 2: Run all cells and review outputs**

Record findings in the Summary cell. Note which adjustments are warranted.

**Step 3: Commit**

```
feat: add EDA notebook with data quality, distributions, and feature analysis
```

---

## Task 7: Apply Model Adjustments

**Files (conditional — only modify what EDA shows is needed):**
- Possibly modify: `src/netball_model/features/contextual.py` (venue mappings)
- Possibly modify: `src/netball_model/model/calibration.py` (distribution choice)

**Context:** This task is conditional on EDA findings. Review the EDA notebook outputs and apply only what's warranted.

**Step 1: Fix unmapped venues (if any)**

If EDA Cell 22 showed unmapped venues, add them to `VENUE_TO_CITY` in `src/netball_model/features/contextual.py`. Research the city for each unmapped venue and add the mapping.

**Step 2: Check if margin calibration needs adjustment**

If EDA Cell 10 showed significant non-normality (p < 0.05 with high kurtosis > 1):

Replace `src/netball_model/model/calibration.py` with a version that uses a t-distribution:

```python
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as t_dist


class CalibrationModel:
    def __init__(self):
        self.residual_std: float = 10.0
        self.df: float | None = None  # degrees of freedom for t-dist

    def fit(self, residuals: np.ndarray):
        self.residual_std = float(np.std(residuals))
        # Fit t-distribution to check if it's a better fit
        df, loc, scale = t_dist.fit(residuals)
        # Only use t-dist if df < 30 (heavier tails than normal)
        if df < 30:
            self.df = df
            self.residual_std = scale

    def win_probability(self, predicted_margin: float) -> float:
        """P(actual_margin > 0) given predicted_margin."""
        if self.df is not None:
            return float(t_dist.cdf(predicted_margin / self.residual_std, self.df))
        return float(norm.cdf(predicted_margin / self.residual_std))
```

Only apply this if kurtosis is notably > 0 (heavier tails than normal). If margins look approximately normal, leave calibration.py unchanged.

**Step 3: Run existing tests to verify nothing broke**

```bash
poetry run pytest tests/ -v --tb=short
```
Expected: ALL PASS (34 tests).

If the calibration model changed, update the calibration test in `tests/model/test_train.py::test_calibration` to account for the new behavior (the existing test should still pass since the t-distribution degenerates to normal for large df).

**Step 4: Commit (if changes were made)**

```
fix: update venue mappings and calibration based on EDA findings
```

---

## Task 8: Train the Model

**Files:**
- No code changes. Uses existing `netball train` CLI command.

**Step 1: Train on all data**

```bash
poetry run netball train --db data/netball.db --output data/model.pkl
```

Expected output:
```
Building features from NNN matches...
Training on NNN rows, NN features...
Model saved to data/model.pkl
Training MAE: X.X goals
```

Note the Training MAE — this is on the training set so it's optimistic. The backtest MAE will be the real evaluation.

**Step 2: Verify model file was created**

```bash
ls -la data/model.pkl
```

Expected: File exists, size ~50-200KB.

**Step 3: No commit** (model file is gitignored)

---

## Task 9: Backtest on 2024 Season

**Files:**
- No code changes. Uses existing `netball backtest` CLI command.

**Step 1: Run walk-forward backtest**

```bash
poetry run netball backtest --db data/netball.db --train-seasons 2017-2023 --test-season 2024
```

Expected output:
```
Training on NNN matches (2017-2023)
Testing on NN matches (2024)

Backtest Results (2024):
  Matches: NN
  Win/Loss Accuracy: XX.X%
  Mean Absolute Error: X.X goals
```

**Step 2: Evaluate results against targets**

| Metric | Target | Actual |
|--------|--------|--------|
| Win/Loss Accuracy | > 55% | [fill in] |
| Mean Absolute Error | < 12 goals | [fill in] |

**Step 3: Record results in EDA notebook**

Add a final cell to `notebooks/eda.ipynb` with the backtest results:

**Cell 24 (markdown):**
```markdown
## 8. Backtest Results

| Metric | Target | Result |
|--------|--------|--------|
| Win/Loss Accuracy | > 55% | [fill in] |
| MAE | < 12 goals | [fill in] |
| Training MAE | — | [fill in] |

### Interpretation

[Fill in after seeing results]
```

**Step 4: Commit**

```
feat: record backtest results in EDA notebook
```

---

## Summary

| Task | Component | Key Deliverable |
|------|-----------|----------------|
| 1 | Setup | matplotlib dependency |
| 2 | Ingestion | 2017-2025 SSN data in SQLite |
| 3 | EDA | Data quality validation cells |
| 4 | EDA | Score distributions + home advantage |
| 5 | EDA | Feature correlations + scoring trends |
| 6 | EDA | Glicko-2 trajectories + venue coverage |
| 7 | Adjustments | Fix venue mappings, calibration if needed |
| 8 | Training | Trained model.pkl |
| 9 | Evaluation | 2024 backtest results |
