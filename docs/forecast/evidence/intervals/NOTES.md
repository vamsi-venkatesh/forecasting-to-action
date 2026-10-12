# Interval study 1: record of how it ran

- **2026-09-28 14:43:02 UTC.** Rules frozen: `study_frozen.json`, sha256 `c024221a…`, committed in `fc70987` before any result.
- **Development run** → `pooled_w10` selected. `development.json` was committed before the test was run.
- **First test run crashed** before any metric was computed. The cause: forecasts for periods after the data's last period have no actual value, and the "imputed" flag of those rows is empty (NaN).
  - The fix is in code only (`interval_study.py`: drop rows without an actual first).
  - The rules and the study hash did not change; the script refuses to run otherwise.
  - No test number existed before the fix.
- **Test run → verdict NOT validated** (`test.json`).
  - `pooled_w10`: 82.0% on a fresh NordWerk realisation (seed 20261015, 671 points), but 58.0% on UCI Bike Sharing daily (1 series, 112 points).
  - The rule required 75–85% on both.
  - On the bike data, LightGBM's point forecasts are poor as well (scaled error 2.01). The test window (the last 28 days of 2012) includes the winter decline, which recent calibration windows do not represent.
- **Consequence, as frozen:**
  - intervals are shown as NOT validated everywhere;
  - nothing relies on them (planning uses point forecasts only);
  - `pooled_w10` replaces `conformal_w3`, because it is better or equal on all four datasets. That does not make it validated.
- **The NordWerk holdout's 64.2%** was seen before this study and was not used by it.

