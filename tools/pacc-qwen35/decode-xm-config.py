#!/usr/bin/env python3
import json
from pathlib import Path
import sys

data = json.loads(Path(sys.argv[1]).read_text())
records = {r['status']-0x5400: r['aux'] for r in data['raw_records'] if 0x5400 <= r['status'] < 0x5448}
if len(records) != 72:
    raise SystemExit(f'incomplete configuration capture: {len(records)}/72')
result = {'formats': {}, 'guest': {}}
for mode, name in enumerate(('bf16_w2', 'f16_w2', 'f32_w1', 'i8_w4')):
    row = [records[mode*16+i] for i in range(16)]
    vtypes = [row[lo] | row[lo+1] << 32 for lo in (2, 6, 9)]
    result['formats'][name] = dict(stage=row[0], signal=row[1],
        vtypes=[hex(x) for x in vtypes], vill=[bool(x >> 63) for x in vtypes],
        vl=row[4], tn=row[5], tm=row[8], tk=row[11], tile_zero_completed=bool(row[12]),
        fault_instruction=hex(row[13]), fault_offset=row[14], supervisor_error=row[15])
result['guest'] = dict(isa_names=records[64], klog_errno=records[65],
                      mammoth_disabled=records[66], tile_allocation_failed=records[67])
print(json.dumps(result, indent=2))
