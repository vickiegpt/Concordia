#!/usr/bin/env python3
"""Run hardware ABI regressions serially inside one isolated device lease."""
import os
from pathlib import Path
import subprocess
import sys

root = Path('/srv/lanxin/qwen35-a3b')
env = os.environ.copy()
for key in ('REQUEST_BATCH', 'LOCAL_BATCH', 'VECTOR_PACK', 'EXPERT_BATCH'):
    env['HETGPU_SIFIVE_QK_' + key] = '1'
tests = [(name, {}) for name in (
    'test-cross-layout.py', 'test-cross-reduce.py', 'test-cross-argsort.py',
    'test-topk-cuda-abi.py', 'test-cross-mmf.py')]
tests += [('test-xm-request-batch.py', {'QK_TEST_TYPE': str(q)}) for q in (4, 5, 8)]
for name, extra in tests:
    print('BEGIN', name, extra, flush=True)
    subprocess.run([sys.executable, str(root / 'bin' / name)],
                   env=env | extra, check=True, timeout=300)
print('PASS all cross-request hardware regressions', flush=True)
