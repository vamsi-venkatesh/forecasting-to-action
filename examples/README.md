# Examples

## Synthetic pipeline

`python scripts/run_demo.py smoke --out output/synthetic` generates seeded synthetic sales, stock, deliveries, ordering rules and messages, then runs profile, evaluate, issue and plan. It deliberately includes faults and stockout periods. Its results are software checks, not business outcomes.

## Bike Sharing

The supplied CSV contains 731 daily total-rental observations from UCI Bike Sharing. Attribution: Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894. Licence: CC BY 4.0.

Changes: retain date and total rentals; add a series ID, unit and assumed next-day availability; omit observed weather and same-day target components. The manifest records the transformation and checksums.

```bash
python scripts/run_demo.py bike --data examples/bike-sharing --out output/bike-sharing
```

To download and rebuild from the original public ZIP:

```bash
curl -L --fail 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip' -o bike-sharing.zip
python -m forecast_worker.benchmarks.uci_bike_sharing bike-sharing.zip examples/bike-sharing
```

Expected ZIP SHA256: b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401. If the source bytes change, check the source rather than assuming the same experiment was reproduced.

Commands write to their output directories. Use a new directory to preserve an earlier run. No engine credentials, paid AI calls or external business-system writes are involved.
