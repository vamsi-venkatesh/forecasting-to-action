"""Standalone example of filter-before-fill; not a repaired production evaluator."""
import json
import sys
from pathlib import Path
import pandas as pd

def prepare(rows, issue_time, origin):
    raw=pd.DataFrame(rows)
    raw['ds']=pd.to_datetime(raw['ds'])
    raw['available_at']=pd.to_datetime(raw['available_at'],utc=True)
    known=raw[(raw.available_at<=pd.Timestamp(issue_time))&(raw.ds<=pd.Timestamp(origin))]
    known=known.sort_values('ds').drop_duplicates('ds',keep='last').set_index('ds')
    grid=pd.date_range(known.index.min(),origin,freq='W-MON')
    return known.reindex(grid)['y'].ffill().tolist()

if __name__=='__main__':
    initial=[{'ds':'2025-01-06','available_at':'2025-01-13T00:00:00Z','y':10},
             {'ds':'2025-01-20','available_at':'2025-01-27T00:00:00Z','y':20}]
    changed=[dict(r) for r in initial];changed[1]['y']=200
    a=prepare(initial,'2025-01-20T00:00:00Z','2025-01-13')
    b=prepare(changed,'2025-01-20T00:00:00Z','2025-01-13')
    assert a==b==[10.0,10.0]
    result={'scope':'synthetic causal preparation demonstration only','before':a,'after_future_value_change':b,'invariant':a==b,'original_worker_modified':False}
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps(result))
