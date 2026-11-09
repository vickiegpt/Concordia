#!/usr/bin/env python3
"""Dependency-free layer trace comparison on the RISC-V host."""
import argparse
import array
import json
import math
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('pacc',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();rows=[]
actual={r['key']:r for r in map(json.loads,(a.pacc/'tensors.jsonl').read_text().splitlines())}
for line in (a.cpu/'tensors.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['name']=='result_output':continue
    q=actual.get(r['key'])
    if q is None or r['ne']!=q['ne']:
        rows.append(dict(r,missing_or_shape_mismatch=True));continue
    x=array.array('f');y=array.array('f')
    x.frombytes((a.cpu/(r['key']+'.f32')).read_bytes());y.frombytes((a.pacc/(r['key']+'.f32')).read_bytes())
    if len(x)!=len(y):raise ValueError(r['key'])
    delta=[float(b)-float(c) for b,c in zip(y,x)]
    rows.append(dict(r,max_abs=max(map(abs,delta)),relative_rms=math.sqrt(sum(v*v for v in delta)/max(sum(v*v for v in x),1e-30)),
        finite=all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y),actual_buffer=q.get('buffer')))
a.output.write_text(json.dumps(rows,indent=2)+'\n')
for name in dict.fromkeys(r['name'] for r in rows):
    valid=[r for r in rows if r['name']==name and 'relative_rms' in r]
    if valid:print(name,max(r['max_abs'] for r in valid),max(r['relative_rms'] for r in valid),valid[0].get('actual_buffer'))
