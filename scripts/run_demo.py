"""Run a synthetic example or evaluate the public Bike Sharing dataset. No external writes."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast_worker.evaluate import evaluate
from forecast_worker.fixtures.nordwerk import build
from forecast_worker.operate import issue, run_plan
from forecast_worker.profile import profile

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['smoke', 'bike'])
    p.add_argument('--out', required=True)
    p.add_argument('--data')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'smoke':
        data = out / 'synthetic-inputs'; build(data)
        answers = {'quantity_columns': {'qty': 'net_units', 'quantity': 'net_units'}, 'missing_weeks': 'missing'}
        params = {'answers': answers}
    else:
        data = Path(a.data or 'examples/bike-sharing')
        answers = {'quantity_columns': {'rentals': 'rentals'}, 'missing_weeks': 'zero'}
        params = {'frequency': 'daily', 'answers': answers}
    (out / 'profile.json').write_text(json.dumps(profile(data), indent=2))
    result = evaluate(data, out, params)
    if a.mode == 'smoke':
        issued = out / 'issued'; issued.mkdir(exist_ok=True)
        result['issue'] = issue(data, issued, {**params, 'model': result['summary']['selected'], 'replay': True, 'issued_at': '2026-06-01T09:00:00+00:00'})
        import shutil
        for name in ['inventory.csv', 'receipts.csv', 'rules.csv']:
            shutil.copyfile(data / name, issued / name)
        plan_out = out / 'plan'; plan_out.mkdir(exist_ok=True)
        result['plan'] = run_plan(issued, plan_out, {})
    result['data_status'] = 'worker run on synthetic data' if a.mode == 'smoke' else 'worker run on public observations with assumed availability'
    (out / 'run-summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
