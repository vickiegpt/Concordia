#!/usr/bin/env python3
import runpy
from pathlib import Path

audit = runpy.run_path(str(Path(__file__).with_name('audit-four-pacc-staging.py')))['audit']
for device in range(4):
    offset = 0x200000 + device * 0x80000
    base = 0x2075d000000 + offset
    lines = [f'helper slot layout dev={device} slot_bytes=524288 slot_off={offset:#x} required=64',
             f"staged kernel='test' arg=0 host=0x1234 shared={base+64:#x} size=16",
             f'helper-submit kernel=test shared={base:#x} submit=64 bytes on sifive{device} slot0']
    assert audit(lines, device)['passed']
    assert not audit(lines, (device+1)%4)['passed']
    assert not audit(lines + [f"staged kernel='bad' arg=0 host=0x1 shared={base+0x7ffff:#x} size=2"], device)['passed']
    assert not audit(lines + [f'helper-submit kernel=bad shared={base:#x} submit=524289 bytes on sifive{device} slot0'], device)['passed']
assert not audit([], 0)['passed']
print('PASS four physical ids, crossing bounds, wrong affinity, oversize and missing evidence')
