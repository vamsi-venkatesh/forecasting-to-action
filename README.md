# Forecasting-to-Action

A Python worker for daily and weekly forecasts, model comparison, and replenishment planning.

[Try the live application](https://lab.vvdexops.com/forecast) · [Results](docs/results.md) · [Data format](docs/data-format.md) · [Known limits](docs/limitations.md)

The worker reads CSV files, checks their structure, compares forecasting models on past periods, and writes forecast and planning files. The live application adds the interface and human approval steps. This repository contains the numerical worker, tests, sample data, and evaluation records. It does not contain the hosted application's account, database, or deployment services.

## What it does

- Profiles data for duplicates, date problems, missing periods, unit changes, and ambiguous quantity columns.
- Accepts confirmed column mappings rather than assuming what a file means.
- Compares naive, seasonal-naive, ETS, and lag-based LightGBM forecasts on the same evaluation points.
- Records the selected model before scoring the final period.
- Issues a dated forecast and computes a replenishment proposal from stock, confirmed receipts, and ordering rules.
- Recalculates a plan when a shipment is delayed.
- Handles recurring CSV batches and compares previously issued forecasts with arriving actuals.

## Run an example

Tested on Linux with Python 3.12.14. Create an environment in a clean checkout:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_demo.py smoke --out output/synthetic
```

This creates a labelled synthetic dataset, compares models, issues a replay forecast, and writes a plan. Outputs include evaluation_report.json, evaluation_forecasts.csv, issued/issued_forecast.csv, and plan/plan.json. It makes no AI API calls or business-system writes.

To evaluate the included public Bike Sharing data:

```bash
.venv/bin/python scripts/run_demo.py bike --data examples/bike-sharing --out output/bike-sharing
.venv/bin/python -m pytest forecast_worker/tests -q
```

The tests run without server credentials. One dataset-cache test skips if its original ZIP cache is absent; the supplied public dataset can be evaluated with the command above. A separate clean-copy check on 12 October 2026 passed 48 tests with one skip. [Reproduction details](examples/README.md) cover data acquisition and checksums.

## Recorded results

MASE uses a lag-1 naive-error scale; lower values are better. These are historical evaluations, not promises about new data.

| Dataset | Selected model | Final-period MASE | Naive MASE | Evidence |
|---|---|---:|---:|---|
| UCI retail, 50 products | ETS | 0.7456 | 0.7902 | [Stored result](evaluation/recorded/public-data/run-uci-retail-operator-benchmark.json) |
| UCI retail, 8 products | ETS | 0.5745 | 0.6973 | [Stored result](evaluation/recorded/public-data/run-uci-retail-visitor-subset.json) |
| UCI Bike Sharing, daily | ETS | 1.8529 | 1.7484 | [Reproduced result](evaluation/bike-sharing/run-summary.json) |

ETS was selected on Bike Sharing's validation periods, but naive performed better on the final period. On the stored SMARD electricity run, seasonal naive stayed in use because no candidate cleared the selection threshold. Full metric definitions and evaluation settings are in [results](docs/results.md).

## Limits to read before using it

Daily and weekly CSV data are supported. Monthly forecasts and hierarchical reconciliation are not implemented here. Prediction intervals have failed the recorded validation rules; planning uses point forecasts.

Two known preparation issues are retained in this snapshot: interpolation can use later values before an evaluation split, and forecast issuance can include an unavailable historical row when later rows have arrived. The included checks reproduce both cases. Read [known limits](docs/limitations.md) before using missing-data or irregular-arrival results. A successful software test is not proof of forecast quality.

Retail planning examples use real historical sales with synthetic stock, delivery and purchase-cost inputs. The live sample's approval path writes only to an isolated demo ledger, not an ERP or a real order. The hosted application's current commit is not established by this source snapshot.

## Code map

| Area | Files |
|---|---|
| Profiling and mapping | [profile.py](forecast_worker/profile.py), [mapping.py](forecast_worker/mapping.py) |
| Models and evaluation | [baseline.py](forecast_worker/baseline.py), [evaluate.py](forecast_worker/evaluate.py), [intervals.py](forecast_worker/intervals.py) |
| Forecasts and plans | [operate.py](forecast_worker/operate.py), [planning.py](forecast_worker/planning.py) |
| Incoming data | [ingest.py](forecast_worker/ingest.py) |
| Text and scenarios | [context](forecast_worker/context/), [signals.py](forecast_worker/signals.py) |
| Worker entry point | [run.py](forecast_worker/run.py) |

Copyright 2026 Vamsi Venkatesh. The code is publicly readable and remains all rights reserved. Public data and dependencies keep their own licences; see [third-party notices](THIRD_PARTY_NOTICES.md).
