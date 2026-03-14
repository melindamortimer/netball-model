# Batch Screenshot Prediction Flow — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-match "Manual entry" mode with a batch screenshot workflow — upload multiple bet365 screenshots, review/edit parsed odds, predict all matches at once with value analysis.

**Architecture:** 3-step linear wizard in `app.py` (Upload → Review → Predict) tracked via `st.session_state["wizard_step"]`. OCR uses existing `parse_screenshot()` made safe for batch use. Odds stored to both SQLite (`odds_history`) and JSON (`data/odds/`). Each match predicted independently against the same base historical match list.

**Tech Stack:** Streamlit (st.data_editor, st.file_uploader, session_state), easyocr (existing), SQLite (existing), JSON

**Spec:** `docs/superpowers/specs/2026-03-14-batch-screenshot-predictions-design.md`

**Git policy:** Do NOT run any git commands. Leave all git operations to the user.

---

## Chunk 1: Backend Changes

### Task 1: Make `parse_screenshot()` safe for batch use

**Files:**
- Modify: `src/netball_model/data/bet365_screenshot.py:85-126`
- Test: `tests/data/test_bet365_screenshot.py`

Currently `parse_screenshot()` raises `FileNotFoundError` and `ValueError`. For batch processing, one bad image must not abort the entire batch. Wrap the function body so it returns `None` on failure.

- [ ] **Step 1: Write failing test — parse_screenshot returns None on empty OCR**

Add to `tests/data/test_bet365_screenshot.py`:

```python
def test_empty_ocr_returns_none(self, tmp_path):
    """parse_screenshot returns None when OCR produces no results."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    img_path = tmp_path / "empty.png"
    img.save(img_path)

    with patch("netball_model.data.bet365_screenshot.easyocr") as mock_easyocr:
        reader = MagicMock()
        reader.readtext.return_value = []  # empty OCR
        mock_easyocr.Reader.return_value = reader

        result = parse_screenshot(str(img_path))

    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/data/test_bet365_screenshot.py::TestParseScreenshot::test_empty_ocr_returns_none -v`
Expected: FAIL — currently raises `ValueError("OCR produced no results from the image")`

- [ ] **Step 3: Modify parse_screenshot to return None on failure**

In `src/netball_model/data/bet365_screenshot.py`, wrap the body of `parse_screenshot()`:

```python
def parse_screenshot(image_path: str | Path) -> dict | None:
    """Parse a bet365 screenshot and return structured odds data.

    Returns None if the image cannot be parsed (missing file, OCR failure, etc.).
    """
    path = Path(image_path)
    if not path.exists():
        return None

    try:
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(str(path))
    except Exception:
        return None

    if not results:
        return None

    # ... rest of the function unchanged from line 97 onwards ...
```

Keep everything from `texts = [(r[0], r[1], r[2]) for r in results]` through the end of the function unchanged.

- [ ] **Step 4: Update the existing `test_invalid_image_raises` test**

The existing test expects an exception. Now it should expect `None`:

```python
def test_invalid_image_returns_none(self, tmp_path):
    bad_path = tmp_path / "nonexistent.png"
    result = parse_screenshot(str(bad_path))
    assert result is None
```

- [ ] **Step 5: Run all screenshot tests**

Run: `poetry run pytest tests/data/test_bet365_screenshot.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full test suite**

Run: `poetry run pytest -v`
Expected: 153 tests PASS (no regressions)

---

### Task 2: JSON odds storage helper

**Files:**
- Modify: `app.py` (add `save_odds_json()` function near top, after existing helpers)
- Test: `tests/test_odds_storage.py` (new file)

A pure function to save/load odds JSON with append + deduplication.

- [ ] **Step 1: Write failing test — save_odds_json creates new file**

Create `tests/test_odds_storage.py`:

```python
import json
import datetime
from pathlib import Path


def test_save_odds_json_creates_new_file(tmp_path):
    """First save creates the file with correct structure."""
    from app import save_odds_json

    matches = [
        {
            "date": "2026-03-15",
            "home_team": "West Coast Fever",
            "away_team": "Sunshine Coast Lightning",
            "home_odds": 2.05,
            "away_odds": 1.75,
            "handicap_line": 1.5,
            "handicap_home_odds": 1.80,
            "handicap_away_odds": 2.00,
            "total_line": 125.5,
            "over_odds": 1.87,
            "under_odds": 1.87,
        }
    ]

    out_path = save_odds_json(matches, output_dir=tmp_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["source"] == "bet365_screenshot"
    assert data["session_date"] == datetime.date.today().isoformat()
    assert len(data["matches"]) == 1
    assert data["matches"][0]["home_team"] == "West Coast Fever"


def test_save_odds_json_appends_and_deduplicates(tmp_path):
    """Second save appends new matches and deduplicates existing ones."""
    from app import save_odds_json

    match_a = {
        "date": "2026-03-15",
        "home_team": "West Coast Fever",
        "away_team": "Sunshine Coast Lightning",
        "home_odds": 2.05,
        "away_odds": 1.75,
    }
    match_b = {
        "date": "2026-03-15",
        "home_team": "Melbourne Vixens",
        "away_team": "NSW Swifts",
        "home_odds": 1.50,
        "away_odds": 2.50,
    }

    # First save
    save_odds_json([match_a], output_dir=tmp_path)

    # Second save with match_a updated + new match_b
    match_a_updated = dict(match_a)
    match_a_updated["home_odds"] = 2.10  # odds changed
    save_odds_json([match_a_updated, match_b], output_dir=tmp_path)

    data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
    assert len(data["matches"]) == 2  # deduped, not 3
    # The updated odds should be present (replaced)
    fever_match = [m for m in data["matches"] if m["home_team"] == "West Coast Fever"][0]
    assert fever_match["home_odds"] == 2.10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_odds_storage.py -v`
Expected: FAIL — `save_odds_json` does not exist yet

- [ ] **Step 3: Implement save_odds_json in app.py**

Add this function after the existing helper functions (after `_make_synthetic_starters`):

```python
def save_odds_json(
    matches: list[dict],
    output_dir: Path | None = None,
) -> Path:
    """Save parsed odds to a JSON file with append + deduplication.

    Filename is based on today's date (session date).
    If the file exists, appends new matches and deduplicates by
    home_team + away_team + date.
    """
    import json

    if output_dir is None:
        output_dir = _ROOT / "data" / "odds"
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today().isoformat()
    filepath = output_dir / f"{today}.json"

    # Load existing data if file exists
    if filepath.exists():
        existing = json.loads(filepath.read_text())
        existing_matches = existing.get("matches", [])
    else:
        existing_matches = []

    # Dedup key: home_team + away_team + date
    dedup_keys = {
        (m["home_team"], m["away_team"], m.get("date", ""))
        for m in matches
    }
    # Keep existing matches that don't conflict with new ones
    kept = [
        m for m in existing_matches
        if (m["home_team"], m["away_team"], m.get("date", "")) not in dedup_keys
    ]
    all_matches = kept + matches

    data = {
        "session_date": today,
        "source": "bet365_screenshot",
        "matches": all_matches,
    }
    filepath.write_text(json.dumps(data, indent=2))
    return filepath
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_odds_storage.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `poetry run pytest -v`
Expected: All 155 tests PASS (153 existing + 2 new)

---

## Chunk 2: App Rewrite — Wizard UI

### Task 3: App scaffolding — wizard state management and upload step

**Files:**
- Modify: `app.py` (rewrite the `main()` function)

This task replaces "Manual entry" with "Screenshot round" and implements the upload step (Step 1 of the wizard). The review and predict steps will render placeholder text until Tasks 4 and 5.

- [ ] **Step 1: Remove old manual entry code and add wizard mode**

In `app.py`, replace the mode radio and the entire `else` block (lines 196-236) with:

```python
mode = st.radio("Source", ["Upcoming (from DB)", "Screenshot round"], horizontal=True)
```

- [ ] **Step 2: Initialize wizard session state**

After the mode radio, add initialization:

```python
if mode == "Screenshot round":
    # Initialize wizard state
    if "wizard_step" not in st.session_state:
        st.session_state["wizard_step"] = "upload"
    if "pasted_images" not in st.session_state:
        st.session_state["pasted_images"] = []
    if "parsed_matches" not in st.session_state:
        st.session_state["parsed_matches"] = []
    if "last_paste_hash" not in st.session_state:
        st.session_state["last_paste_hash"] = None
```

- [ ] **Step 3: Implement upload step UI**

The upload step shows a multi-file uploader and the paste component:

```python
if st.session_state["wizard_step"] == "upload":
    st.subheader("Step 1: Upload Screenshots")
    st.caption("Upload bet365 match screenshots — one per match")

    # File uploader (multiple files)
    uploaded_files = st.file_uploader(
        "Drop all screenshots here",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="screenshot_uploader",
    )

    # Paste component
    st.markdown("**Or paste from clipboard:**")
    pasted_data = _paste_component(key="paste_screenshot", default=None)

    # Deduplicated paste append
    if pasted_data:
        paste_hash = hash(pasted_data[:100])  # hash first 100 chars for speed
        if paste_hash != st.session_state["last_paste_hash"]:
            try:
                _, b64 = pasted_data.split(",", 1)
                img_bytes = base64.b64decode(b64)
                st.session_state["pasted_images"].append(img_bytes)
                st.session_state["last_paste_hash"] = paste_hash
            except Exception:
                st.error("Could not decode pasted image.")

    # Show collection summary
    n_uploaded = len(uploaded_files) if uploaded_files else 0
    n_pasted = len(st.session_state["pasted_images"])
    total = n_uploaded + n_pasted
    if total > 0:
        st.info(f"{total} screenshot(s) collected ({n_uploaded} uploaded, {n_pasted} pasted)")

    # Parse all button
    if total > 0 and st.button("Parse all screenshots", type="primary"):
        all_images = []
        if uploaded_files:
            for f in uploaded_files:
                all_images.append(f.getvalue())
        all_images.extend(st.session_state["pasted_images"])

        parsed_list = []
        progress = st.progress(0, text="Parsing screenshots...")
        for idx, img_bytes in enumerate(all_images):
            progress.progress(
                (idx + 1) / len(all_images),
                text=f"Parsing {idx + 1}/{len(all_images)}...",
            )
            parsed = _parse_screenshot_image(img_bytes)
            if parsed:
                # Normalize team names
                from netball_model.data.team_names import normalise_team
                home = normalise_team(parsed.get("home_team") or "")
                away = normalise_team(parsed.get("away_team") or "")
                teams = get_teams(matches)
                parsed_list.append({
                    "Date": datetime.date.today(),
                    "Home Team": home if home else teams[0],
                    "Away Team": away if away else (teams[1] if len(teams) > 1 else teams[0]),
                    "H2H Home": parsed.get("home_odds"),
                    "H2H Away": parsed.get("away_odds"),
                    "Line": parsed.get("handicap_line"),
                    "HC Home": parsed.get("handicap_home_odds"),
                    "HC Away": parsed.get("handicap_away_odds"),
                    "Total": parsed.get("total_line"),
                    "Over": parsed.get("over_odds"),
                    "Under": parsed.get("under_odds"),
                })
            else:
                st.warning(f"Screenshot {idx + 1} could not be parsed — skipped")

        if parsed_list:
            st.session_state["parsed_matches"] = parsed_list
            st.session_state["wizard_step"] = "review"
            st.rerun()
        else:
            st.error("No screenshots could be parsed. Check the images and try again.")
```

- [ ] **Step 4: Add placeholder for review and predict steps**

```python
elif st.session_state["wizard_step"] == "review":
    st.subheader("Step 2: Review & Edit")
    st.write("(Review step — implemented in Task 4)")

elif st.session_state["wizard_step"] == "predict":
    st.subheader("Step 3: Results")
    st.write("(Results step — implemented in Task 5)")
```

- [ ] **Step 5: Keep the "Upcoming (from DB)" mode and single-match predict unchanged**

The `if mode == "Upcoming (from DB)"` block stays exactly as-is (lines 201-213 in current app.py). The Predict button and single-match display below it also stay for this mode.

- [ ] **Step 6: Manual smoke test**

Run: `poetry run streamlit run app.py`
Verify:
- Radio shows "Upcoming (from DB)" and "Screenshot round"
- Screenshot round shows file uploader and paste component
- Can upload a file (no parse yet — just count display)
- "Parse all" button appears when files are collected

- [ ] **Step 7: Run full test suite for regressions**

Run: `poetry run pytest -v`
Expected: All tests PASS

---

### Task 4: Review step — editable data_editor table

**Files:**
- Modify: `app.py` (replace review step placeholder)

- [ ] **Step 1: Replace review placeholder with data_editor**

Replace the `elif st.session_state["wizard_step"] == "review":` block:

```python
elif st.session_state["wizard_step"] == "review":
    st.subheader("Step 2: Review & Edit Parsed Odds")
    st.caption("Fix any OCR errors, add missing matches, delete bad rows")

    teams = get_teams(matches)

    # Build DataFrame from parsed matches
    df = pd.DataFrame(st.session_state["parsed_matches"])

    # Configure column types for data_editor
    column_config = {
        "Date": st.column_config.DateColumn("Date", default=datetime.date.today()),
        "Home Team": st.column_config.SelectboxColumn(
            "Home Team", options=teams, required=True,
        ),
        "Away Team": st.column_config.SelectboxColumn(
            "Away Team", options=teams, required=True,
        ),
        "H2H Home": st.column_config.NumberColumn("H2H Home", min_value=1.01, max_value=100.0, format="%.2f"),
        "H2H Away": st.column_config.NumberColumn("H2H Away", min_value=1.01, max_value=100.0, format="%.2f"),
        "Line": st.column_config.NumberColumn("Line", min_value=-50.0, max_value=50.0, format="%.1f"),
        "HC Home": st.column_config.NumberColumn("HC Home", min_value=1.01, max_value=100.0, format="%.2f"),
        "HC Away": st.column_config.NumberColumn("HC Away", min_value=1.01, max_value=100.0, format="%.2f"),
        "Total": st.column_config.NumberColumn("Total", min_value=50.0, max_value=200.0, format="%.1f"),
        "Over": st.column_config.NumberColumn("Over", min_value=1.01, max_value=100.0, format="%.2f"),
        "Under": st.column_config.NumberColumn("Under", min_value=1.01, max_value=100.0, format="%.2f"),
    }

    edited_df = st.data_editor(
        df,
        column_config=column_config,
        num_rows="dynamic",  # allows add/delete rows
        use_container_width=True,
        hide_index=True,
        key="odds_editor",
    )

    if st.button("Confirm & Predict", type="primary"):
        if edited_df.empty:
            st.error("No matches to predict. Add at least one row.")
        else:
            st.session_state["confirmed_matches"] = edited_df.to_dict("records")
            st.session_state["wizard_step"] = "predict"
            st.rerun()
```

- [ ] **Step 2: Manual smoke test**

Run: `poetry run streamlit run app.py`
Verify:
- After parsing screenshots (or manually advancing to review), the data_editor table appears
- Columns have correct types (dropdowns for teams, numbers for odds)
- Can edit cells, delete rows, add new rows
- "Confirm & Predict" button advances to predict step

- [ ] **Step 3: Run full test suite**

Run: `poetry run pytest -v`
Expected: All tests PASS

---

### Task 5: Predict & results step — batch predictions with value analysis

**Files:**
- Modify: `app.py` (replace predict step placeholder)

This is the final step. For each confirmed match: save to DB + JSON, run prediction independently, compute value, display results.

- [ ] **Step 1: Replace predict placeholder with batch prediction logic**

Replace the `elif st.session_state["wizard_step"] == "predict":` block:

```python
elif st.session_state["wizard_step"] == "predict":
    st.subheader("Step 3: Predictions & Value")

    confirmed = st.session_state.get("confirmed_matches", [])
    if not confirmed:
        st.warning("No matches confirmed.")
        return

    model = load_model()
    detector = ValueDetector(min_edge=min_edge)

    # --- Save odds to DB + JSON ---
    db = load_db()
    odds_for_db = []
    odds_for_json = []
    for row in confirmed:
        match_id = f"manual_{row['Home Team']}_{row['Away Team']}"
        date_val = row.get("Date")
        if isinstance(date_val, datetime.date):
            date_str = date_val.isoformat()
        else:
            date_str = str(date_val) if date_val else datetime.date.today().isoformat()

        odds_for_db.append({
            "match_id": match_id,
            "source": "bet365_screenshot",
            "home_back_odds": row.get("H2H Home"),
            "away_back_odds": row.get("H2H Away"),
            "home_lay_odds": None,
            "away_lay_odds": None,
            "home_volume": None,
            "away_volume": None,
            "timestamp": datetime.datetime.now().isoformat(),
            "handicap_line": row.get("Line"),
            "handicap_home_odds": row.get("HC Home"),
            "handicap_away_odds": row.get("HC Away"),
            "total_line": row.get("Total"),
            "over_odds": row.get("Over"),
            "under_odds": row.get("Under"),
        })
        odds_for_json.append({
            "date": date_str,
            "home_team": row["Home Team"],
            "away_team": row["Away Team"],
            "home_odds": row.get("H2H Home"),
            "away_odds": row.get("H2H Away"),
            "handicap_line": row.get("Line"),
            "handicap_home_odds": row.get("HC Home"),
            "handicap_away_odds": row.get("HC Away"),
            "total_line": row.get("Total"),
            "over_odds": row.get("Over"),
            "under_odds": row.get("Under"),
        })

    try:
        db.upsert_odds_extended_batch(odds_for_db)
    except Exception as e:
        st.warning(f"Could not save to DB: {e}")

    json_path = save_odds_json(odds_for_json)
    st.success(f"Saved odds to DB + {json_path.relative_to(_ROOT)}")

    # --- Run predictions ---
    results_rows = []
    progress = st.progress(0, text="Running predictions...")
    for idx, row in enumerate(confirmed):
        progress.progress(
            (idx + 1) / len(confirmed),
            text=f"Predicting {idx + 1}/{len(confirmed)}...",
        )
        home = row["Home Team"]
        away = row["Away Team"]
        date_val = row.get("Date")
        if isinstance(date_val, datetime.date):
            date_str = date_val.isoformat()
            season = date_val.year
        else:
            date_str = str(date_val) if date_val else datetime.date.today().isoformat()
            season = datetime.date.today().year

        # Build independent match list (base matches + this one match)
        temp_match = {
            "match_id": f"manual_{home}_{away}",
            "date": date_str,
            "season": season,
            "round_num": 0,
            "home_team": home,
            "away_team": away,
            "home_score": None,
            "away_score": None,
            "venue": "",
        }
        match_list = matches + [temp_match]
        match_index = len(match_list) - 1

        prediction = predict_match(
            match_list, match_index,
            player_stats if player_stats else None,
        )

        # Evaluate value
        win_prob = prediction["win_probability"]
        pred_dict = {
            "margin": prediction["predicted_margin"],
            "total_goals": prediction["predicted_total"],
            "win_prob": win_prob,
            "residual_std": model.residual_std,
            "total_residual_std": model.total_residual_std,
        }
        odds_dict = {}
        if row.get("H2H Home"):
            odds_dict["home_odds"] = row["H2H Home"]
        if row.get("H2H Away"):
            odds_dict["away_odds"] = row["H2H Away"]
        if row.get("Line") is not None:
            odds_dict["handicap_line"] = row["Line"]
            odds_dict["handicap_home_odds"] = row.get("HC Home")
            odds_dict["handicap_away_odds"] = row.get("HC Away")
        if row.get("Total") is not None:
            odds_dict["total_line"] = row["Total"]
            odds_dict["over_odds"] = row.get("Over")
            odds_dict["under_odds"] = row.get("Under")

        value_results = detector.evaluate(pred_dict, odds_dict) if odds_dict else []

        # Extract best edge per market
        def best_edge(market_name):
            market_results = [v for v in value_results if v["market"] == market_name]
            if not market_results:
                return None
            # Prefer positive edge; if none, show least negative
            positive = [v for v in market_results if v["edge"] > 0]
            if positive:
                return max(positive, key=lambda v: v["edge"])
            return max(market_results, key=lambda v: v["edge"])

        h2h_best = best_edge("h2h")
        hc_best = best_edge("handicap")
        tot_best = best_edge("total")

        value_markets = []
        if h2h_best and h2h_best["edge"] >= min_edge:
            value_markets.append("H2H")
        if hc_best and hc_best["edge"] >= min_edge:
            value_markets.append("HC")
        if tot_best and tot_best["edge"] >= min_edge:
            value_markets.append("Total")

        results_rows.append({
            "Match": f"{home} v {away}",
            "Date": date_str[:10],
            "Margin": f"{prediction['predicted_margin']:+.1f}",
            "Home Win%": f"{win_prob:.0%}",
            "Total": f"{prediction['predicted_total']:.1f}",
            "H2H Edge": f"{h2h_best['edge']:+.1%}" if h2h_best else "—",
            "HC Edge": f"{hc_best['edge']:+.1%}" if hc_best else "—",
            "Total Edge": f"{tot_best['edge']:+.1%}" if tot_best else "—",
            "Value": ", ".join(value_markets) if value_markets else "",
            # Store raw for callouts below
            "_value_details": value_results,
            "_home": home,
            "_away": away,
        })

    progress.empty()

    # --- Display results table ---
    display_df = pd.DataFrame(results_rows)
    display_cols = ["Match", "Date", "Margin", "Home Win%", "Total",
                    "H2H Edge", "HC Edge", "Total Edge", "Value"]

    def highlight_value(row):
        if row["Value"]:
            return ["background-color: #1a7a3a; color: white"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df[display_cols].style.apply(highlight_value, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # --- Value callouts ---
    for row in results_rows:
        for v in row["_value_details"]:
            if v.get("edge", 0) >= min_edge:
                side = v["side"]
                market = v["market"]
                if market == "h2h":
                    label = row["_home"] if side == "home" else row["_away"]
                elif market == "handicap":
                    label = f"{'Home' if side == 'home' else 'Away'} ({v.get('line', '')})"
                else:
                    label = f"{side.title()} {v.get('line', '')}"
                st.success(
                    f"**{row['Match']}** — {market.upper()} — {label} — "
                    f"{v['edge']:.1%} edge (odds {v.get('odds', 0):.2f})"
                )

    # Reset button
    if st.button("Start new round"):
        for key in ["wizard_step", "pasted_images", "parsed_matches",
                     "last_paste_hash", "confirmed_matches"]:
            st.session_state.pop(key, None)
        st.rerun()
```

- [ ] **Step 2: Remove old single-match odds section for Screenshot round mode**

Make sure the old single-match prediction display + odds input sections (lines 259-478 in current app.py) only render for the "Upcoming (from DB)" mode. The "Screenshot round" mode uses the wizard exclusively.

Wrap the old prediction display and odds sections with:

```python
if mode == "Upcoming (from DB)":
    # ... existing Predict button, prediction display, odds input, value table ...
```

- [ ] **Step 3: Clean up unused code**

Remove from `app.py`:
- `_fill_odds_from_parsed()` function — no longer used (was for single-match screenshot fill)
- The `tab_upload` / `tab_paste` screenshot import section — replaced by the wizard upload step
- The old H2H / Handicap / Total number inputs — replaced by the data_editor in the wizard
- The old value table rendering — replaced by the batch results table

Keep:
- `_parse_screenshot_image()` — reused by the wizard's batch parse
- `predict_match()` — reused by the wizard's batch predict
- `save_odds_json()` — used by the wizard's predict step
- `_paste_component` declaration — reused by the wizard's upload step
- All `@st.cache_*` functions — unchanged
- `build_squad_starters()` / `_make_synthetic_starters()` — used by predict_match

- [ ] **Step 4: Manual end-to-end smoke test**

Run: `poetry run streamlit run app.py`

Test the full flow:
1. Select "Screenshot round"
2. Upload 1-2 test screenshots (or use paste)
3. Click "Parse all screenshots"
4. Verify the review table shows parsed odds with team dropdowns
5. Edit a cell, add a row, delete a row
6. Click "Confirm & Predict"
7. Verify predictions table shows all matches with edge columns
8. Verify value bets are highlighted green
9. Verify `data/odds/` JSON file was created
10. Click "Start new round" — verify wizard resets
11. Switch to "Upcoming (from DB)" — verify it still works as before

- [ ] **Step 5: Run full test suite**

Run: `poetry run pytest -v`
Expected: All tests PASS (153 existing + 2 new odds storage = 155)
