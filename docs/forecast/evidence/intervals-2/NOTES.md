# Interval study 2: record of how it ran (2026-09-28)

- **Rules frozen before any result:** `study_frozen.json`, sha256 `61f5bdb3…`, committed in `74b8e07`.
- **Why a second study:** study 1 (`../intervals/`) was NOT validated: pooled_w10 covered 82.0% on a fresh NordWerk
  realisation but 58.0% on Bike Sharing daily. Its four datasets are now seen, so they are development data here.
- **Hypothesis tested:** bike daily is one series; pooled_w10 calibrates on 10 consecutive daily origins (10 scores per
  step, all from one fortnight). `spread_y1` spreads the calibration origins over the last year (daily 52 origins a
  week apart; weekly 26 origins two weeks apart).
- **Development run** (`development.json`, committed before the test): the hypothesis is **falsified**.
  - bike daily: pooled_w10 58.0%, pooled_w20 55.4%, spread_y1 59.8% (112 points; nominal 80%).
  - spread_y1 over-covers retail (88.2%) and NordWerk A (87.4%).
  - No method covered 75-85% on all four sets, so by the frozen rule no method can be validated; pooled_w10 (3 of 4
    inside) was tested for the record.
- **Test run on data no interval study had used** (`test.json`): verdict **NOT validated**.
  - NordWerk seed 20261101 (synthetic, weekly): 83.6%, mean scaled width 2.83, 671 points.
  - SMARD daily grid load of the four German control areas (real, 2019-01-01 to 2026-09-27, retrieved 2026-09-28
    21:53 UTC): 88.0%, mean scaled width 1.86, 448 points. Too wide by the frozen rule (above 85%).
- **Consequence, unchanged:** ranges stay labelled "not validated" everywhere; planning uses point forecasts only;
  nothing interval-dependent is automated. `spread_y1` stays in code as a measured, rejected alternative and is not
  used by the product.
- **What the two studies together show:** pooled_w10 lands inside 80 ± 5% on 4 of 6 datasets (NordWerk A, B, C and UCI
  retail, all weekly); it under-covers one daily series with a strong seasonal decline in the test window (bike) and
  over-covers four smooth daily series (SMARD). One pooled quantile per step does not fit every kind of series; a
  next study would need per-dataset or level-aware calibration, chosen on these six and tested on new data.

