#!/usr/bin/python3
"""Isolated JIT compiler-option experiment; never silently reuse the O0 cache."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

args = sys.argv[1:]
inputs = [Path(arg) for arg in args if arg.endswith(('.bc', '.ll')) and Path(arg).is_file()]
codegen = bool(inputs and ('-c' in args or '-S' in args))
extra = ['-O3'] if codegen else []
started = time.monotonic()
rc = subprocess.run(['/usr/bin/clang-20', *args, *extra]).returncode
if codegen:
    record = dict(args=args, added=extra, returncode=rc,
                  elapsed_s=time.monotonic()-started,
                  inputs=[dict(path=str(p), bytes=p.stat().st_size,
                               sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    if '-o' in args:
        output = Path(args[args.index('-o')+1])
        if output.is_file():
            record['output_bytes'] = output.stat().st_size
    fd = os.open('/srv/lanxin/qwen35-a3b/logs/clang-o3-audit.jsonl',
                 os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(record)+'\n').encode())
    finally:
        os.close(fd)
sys.exit(rc)
