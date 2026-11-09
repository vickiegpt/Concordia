#!/usr/bin/env python3
"""Bounded CPU reference for the 512-state bring-up case, not a PACC TPS run."""
import os
from pathlib import Path
import resource
import subprocess

root = Path('/srv/lanxin/qwen35-a3b')
env = os.environ.copy()
env.pop('LD_PRELOAD', None)
env['LD_LIBRARY_PATH'] = ':'.join(str(root / p) for p in
    ('build-seq512/bin', 'llama.cpp/build-pacc-cuda-gcc14/bin'))
resource.setrlimit(resource.RLIMIT_AS, (80 * 1024**3, 80 * 1024**3))
with (root / 'logs/recovery-512-cpu-reference.log').open('x') as log:
    result = subprocess.run([
        str(root / 'build-seq512/bin/qwen-multiseq-v4'),
        str(root / 'models/Qwen3.5-35B-A3B-Q4_K_M.gguf'), '0',
        str(root / 'bin/cross-512-recovery.json'),
        str(root / 'logs/recovery-512-cpu-reference.json'), '-', '8'],
        env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, cwd=root)
raise SystemExit(result.returncode)
