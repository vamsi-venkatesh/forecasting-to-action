"""Focused adversarial checks of the preserved source; findings are not production fixes."""
import csv
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from forecast_worker.baseline import _load
from forecast_worker.operate import issue

def main():
    root = Path(tempfile.mkdtemp(prefix='forecast-check-'))
    answers = {'quantity_columns': {'quantity': 'units'}, 'missing_weeks': 'missing'}
    def data(name, future):
        d = root / name; d.mkdir()
        rows = []
        for i, date in enumerate(pd.date_range('2025-01-06', periods=14, freq='W-MON')):
            if i == 9: continue
            val = 10 if i < 10 else future
            rows.append(['A', date.date().isoformat(), (date + pd.Timedelta(days=7)).isoformat()+'+00:00', val])
        with (d/'sales_a.csv').open('w', newline='') as f:
            w=csv.writer(f);w.writerow(['series_id','week_start','available_at','quantity']);w.writerows(rows)
        return d
    a, b = data('a', 20), data('b', 200)
    pa, _ = _load(a, answers); pb, _ = _load(b, answers)
    missing = pd.Timestamp('2025-03-10')
    va=float(pa.loc[pa.ds.eq(missing),'y'].iloc[0]); vb=float(pb.loc[pb.ds.eq(missing),'y'].iloc[0])
    def late(name, quantity):
        d=root/name;d.mkdir()
        with (d/'sales_late.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['series_id','week_start','available_at','quantity'])
            for i,dt in enumerate(pd.date_range('2025-01-06',periods=14,freq='W-MON')):
                at='2026-01-01T00:00:00+00:00' if i==7 else (dt+pd.Timedelta(days=7)).isoformat()+'+00:00'
                w.writerow(['A',dt.date().isoformat(),at,quantity if i==7 else 10+i])
        return d
    predictions=[]
    for name,q in [('late-a',10),('late-b',10000)]:
        d=late(name,q);o=d/'out';o.mkdir()
        issue(d,o,{'answers':answers,'model':'ets','issued_at':'2025-04-14T00:00:00+00:00','replay':True})
        predictions.append(float(pd.read_csv(o/'issued_forecast.csv').yhat.iloc[0]))
    report={'fixture':'explicitly synthetic adversarial inputs', 'pre_split_interpolation':{'input_before_future_change':va,'input_after_future_change':vb,'future_influences_past':va!=vb},'late_row_issue_filter':{'forecast_before_late_value_change':predictions[0],'forecast_after_late_value_change':predictions[1],'unavailable_historical_row_influences_issue':predictions[0]!=predictions[1]},'source_modified':False,'scope':'These checks demonstrate failure cases; they do not quantify bias in every historical benchmark.'}
    out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__': main()
