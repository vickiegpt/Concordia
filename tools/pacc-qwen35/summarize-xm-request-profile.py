#!/usr/bin/env python3
"""Pair native device timings with operator wall time; do not label residual IRQ time."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('benchmark',type=Path);p.add_argument('log',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();bench=json.loads(a.benchmark.read_text());groups=defaultdict(list)
for line in a.log.read_text().splitlines():
    if not line.startswith('PACC_QK_PROFILE '):continue
    row={k:float(v) for k,v in (part.split('=',1) for part in line.split()[1:])}
    groups[(int(row['dev']),int(row['qtype']),int(row['m']),int(row['n']),int(row['nb'])*256,int(row['gate']))].append(row)
result=[]
for case in bench['cases']:
    rows=groups[(case.get('device',bench.get('device')),case.get('qtype',bench.get('qtype')),
                 case['m'],case['n'],case['k'],case.get('gate',0))]
    modes=case.get('results',{'baseline':case})
    assert len(rows)==9*len(modes)
    for index,(mode,measure) in enumerate(modes.items()):
        assert len(measure['latency_us'])==9
        times={key:mean(r[key] for r in rows[index*9+1:index*9+9]) for key in
               ('device_us','activation_us','weights_us','xm_us','reduce_us','local_copy_us')}
        accounted=sum(times[k] for k in ('activation_us','weights_us','xm_us','reduce_us','local_copy_us'))
        assert 0<accounted<=times['device_us']<=measure['mean_us']
        result.append(dict(m=case['m'],n=case['n'],k=case['k'],gate=case.get('gate',0),mode=mode,
                           wall_us=measure['mean_us'],**times,
                           device_other_us=times['device_us']-accounted,
                           non_device_residual_us=measure['mean_us']-times['device_us']))
report=dict(source=str(a.benchmark),profiles=result,
            limits=['Instrumentation adds overhead.',
                    'Non-device residual includes host staging, transport, dispatch, completion observation and copyback; not pure interrupt/DMA time.',
                    'These are operator measurements, not full-model TPS.'])
with a.output.open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2))
