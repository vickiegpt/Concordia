#!/usr/bin/env python3
"""A short bring-up case: 512 independent states, distinct numeric prompts."""
import json
from pathlib import Path

path = Path('/srv/lanxin/qwen35-a3b/bin/cross-512-recovery.json')
with path.open('x') as output:
    json.dump(dict(prompts=[str(i + 1) for i in range(512)], max_tokens=2,
                   min_tokens=2, batched_prefill=True), output, indent=2)
    output.write('\n')
print(path)
