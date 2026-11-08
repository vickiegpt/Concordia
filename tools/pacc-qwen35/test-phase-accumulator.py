#!/usr/bin/env python3
from jobd_phase_accumulator import PhaseAccumulator

def rows():
    return [dict(index=101+i,seq=9,status=s,job=5,aux=0x20036 if s==0x5330 else 10+i)
            for i,s in enumerate([0x5140,0x5330,*range(0x5310,0x5317)])]
a=PhaseAccumulator(100,0x20036)
for r in rows():a.observe(r);a.observe(r)
assert a.report(0,False)['capture_valid']
assert a.report(0,False)['phases_us']['compute']['sum_us']==14
assert not a.report(1,False)['capture_valid']
assert not a.report(0,True)['capture_valid']
for missing in range(1,9):
    a=PhaseAccumulator(100,0x20036)
    for i,r in enumerate(rows()):
        if i!=missing:a.observe(r)
    assert not a.report(0,False)['capture_valid']
    a.observe(rows()[missing])
    assert a.report(0,False)['capture_valid']
a=PhaseAccumulator(100,0x20037)
for r in rows():a.observe(r)
assert not a.report(0,False)['capture_valid']
print('PASS streaming phases, duplicates, recoverable gaps, missing phases/build and failed execution')
for split in range(9):
    a=PhaseAccumulator(100,0x20036)
    events=rows()
    for r in events[split:]+events[:split]:a.observe(r)
    assert a.report(0,False)['capture_valid'],split
a=PhaseAccumulator(100,0x20036)
for r in rows():a.observe(r)
try:
    a.observe(dict(index=110,seq=9,status=0x5140,job=5,aux=0))
except RuntimeError:
    pass
else:
    raise AssertionError('reused completed sequence accepted')
print('PASS wrapped ring order and reused-sequence rejection')
