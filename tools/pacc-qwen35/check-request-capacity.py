#!/usr/bin/env python3
"""Query the built llama ABI before allocating a large independent-sequence batch."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--requests',type=int,required=True)
p.add_argument('--devices',type=int,default=4)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.requests<1 or a.devices<1 or a.output.exists():p.error('invalid dimensions or output exists')
lib=ctypes.CDLL(str(a.library.resolve()))
limit=lib.llama_max_parallel_sequences
limit.argtypes=[];limit.restype=ctypes.c_size_t
maximum=limit();per_device=(a.requests+a.devices-1)//a.devices
fits=per_device<=maximum
report=dict(stage='read-only ABI preflight; no inference or device allocation',
            requested_independent_sequences=a.requests,devices=a.devices,
            sequences_per_device=per_device,max_sequences_per_instance=maximum,
            within_compiled_sequence_limit=fits,aggregate_decode_tps=None,
            library=str(a.library.resolve()),
            library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limitations=['A passing sequence-limit check does not prove memory capacity, operator support or correct generation.',
                         'Logical token batch, active sequences and XM matrix columns are different dimensions.'])
if not fits:report['blocker']='requested per-device sequence count exceeds compiled llama ABI limit'
with a.output.open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2))
raise SystemExit(0 if fits else 1)
