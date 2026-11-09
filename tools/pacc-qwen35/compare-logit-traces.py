#!/usr/bin/env python3
"""Compare forced-prefix F32 traces; never convert diagnostic timings into TPS."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('actual',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
def compare(ref,got):
    x=np.fromfile(ref,dtype='<f4').astype(np.float64);y=np.fromfile(got,dtype='<f4').astype(np.float64)
    if x.shape!=y.shape:raise ValueError(f'shape mismatch: {ref}')
    d=y-x
    return dict(count=x.size,max_abs=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))),
        relative_rms=float(np.linalg.norm(d)/max(np.linalg.norm(x),1e-30)),
        finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),
        cpu_argmax=int(np.argmax(x)),pacc_argmax=int(np.argmax(y)),
        cpu_margin=float(np.max(x)-np.partition(x,-2)[-2]),
        cpu_best_on_pacc=float(y[np.argmax(x)]),pacc_best=float(np.max(y)))
records=[]
for line in (a.reference/'tensors.jsonl').read_text().splitlines():
    t=json.loads(line);file=t['key']+'.f32'
    if not (a.actual/file).exists():
        records.append(dict(t,missing=True));continue
    records.append(dict(t,**compare(a.reference/file,a.actual/file)))
logits=[]
for ref in sorted(a.reference.glob('logits-*.f32')):
    if (a.actual/ref.name).exists():logits.append(dict(key=ref.stem,**compare(ref,a.actual/ref.name)))
out=dict(reference=str(a.reference),actual=str(a.actual),tensors=records,logits=logits)
a.output.write_text(json.dumps(out,indent=2)+'\n')
for name in dict.fromkeys(r['name'] for r in records):
    rows=[r for r in records if r['name']==name and not r.get('missing')]
    if rows:print(name,'max_abs',max(r['max_abs'] for r in rows),'max_rel_rms',max(r['relative_rms'] for r in rows))
for r in logits:print(r['key'],r['cpu_argmax'],r['pacc_argmax'],'margin',r['cpu_margin'],'max_abs',r['max_abs'],'rel',r['relative_rms'])
