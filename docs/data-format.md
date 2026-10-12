# Data format

Use one row per series and period. Forecasting needs a numeric quantity, a period date, a series identifier, and the time the observation became available.

| Column | Meaning |
|---|---|
| series_id | Product, location, or other series to forecast separately |
| week_start or period_start | Period start as YYYY-MM-DD; weekly periods start on Monday |
| available_at | Availability timestamp with a time zone |
| quantity or a confirmed quantity column | Numeric target to forecast |
| unit | Unit of the target, such as pcs or rentals |

```csv
series_id,week_start,available_at,quantity,unit
P-100,2026-01-05,2026-01-12T06:00:00+00:00,42,pcs
P-100,2026-01-12,2026-01-19T06:00:00+00:00,0,pcs
```

Standard worker input files use the sales_*.csv naming pattern. Quantity-column mapping and the meaning of absent periods must be supplied in params.answers. The worker can propose a mapping for unfamiliar files; a person must confirm it. An absent period and a recorded zero are different.

Planning also requires inventory.csv, receipts.csv, and rules.csv. Stock and confirmed deliveries must refer to the same units as the forecast. Rules include buffer, pack size, minimum order, currency and purchase cost. An unknown price stays unknown. See the generated synthetic example for the full schemas.

The worker accepts job files through `python -m forecast_worker.run JOB.json OUT_DIR`. A job contains kind, inputs mapping file names to local paths, and params. Supported kinds are profile, baseline, evaluate, issue, plan, context, scenario, signals and ingest. [run_demo.py](../scripts/run_demo.py) shows direct calls without the hosted application's database or API.

Availability metadata must reflect what was known at the time. Merely adding an available_at column does not resolve the known irregular-arrival and interpolation issues in this snapshot; see [limitations](limitations.md).
