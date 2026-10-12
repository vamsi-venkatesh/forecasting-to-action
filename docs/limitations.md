# Known limits

## Missing values can use later observations

The preserved baseline loader interpolates the full series before the time split. A later value can therefore change a missing training input. Removing imputed targets from scoring does not remove that influence on the model inputs.

`python scripts/check_preprocessing.py output/preprocessing-check.json` reproduces this failure with synthetic data. Changing a later observed value from 20 to 200 changes an earlier missing input from 15 to 105.

## An unavailable historical row can enter an issued forecast

Forecast issuance identifies the latest available period, then retains every row up to it. An earlier row can still have an availability time after the issue time. The same check changes such a synthetic row and observes a change in the issued forecast.

Evaluation uses a gap inferred from the earliest origin rather than checking every later row's availability at every origin. Irregular delays and historical revisions require more complete as-of filtering. Current duplicate handling keeps the last period record rather than reconstructing revision history.

These counterexamples do not quantify the effect on every dataset. The gap-free Bike Sharing run reproduced its stored scores under its regular, assumed availability schedule. Results with missing data or changing delays need separate checks.

`scripts/causal_preparation_demo.py` illustrates filtering raw observations before filling gaps from earlier values. It is a small standalone example, not a replacement for the worker's complete evaluation path.

## Other limits

- Only daily and weekly calendars are supported.
- Prediction intervals remain unvalidated. The plan uses point forecasts.
- Current replenishment arithmetic is a test policy. It does not compare sequential inventory costs or optimize capacity.
- Public observations do not include real stock, lead-time or purchase-cost records. Retail planning inputs are synthetic.
- Synthetic text experiments do not establish the value of real news or business messages.
- A live demo ledger is not a production ERP. The account, approval and ledger backend is outside this source extract.
- The hosted application's deployed commit was not identified. Source, recorded evidence and live behavior have separate provenance.
- Software-test counts describe the tested code and environment. They do not establish accuracy, security or business value on all inputs.
