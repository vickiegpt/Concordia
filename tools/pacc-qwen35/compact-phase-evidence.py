#!/usr/bin/env python3
"""Retain phase/profile gates and raw evidence identity without committing huge rings."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('destination', type=Path)
args = parser.parse_args()
with args.source.open() as stream:
    raw = json.load(stream)
excluded = {'raw_records', 'submissions', 'submission_builds', 'observed_index_gaps'}
compact = {key: value for key, value in raw.items() if key not in excluded}
compact['observed_index_gap_count'] = len(raw.get('observed_index_gaps', []))
compact['submission_build_counts'] = {}
for build in raw.get('submission_builds', {}).values():
    key = hex(build)
    compact['submission_build_counts'][key] = compact['submission_build_counts'].get(key, 0) + 1
with args.source.open('rb') as stream:
    compact['raw_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
compact['raw_path'] = str(args.source.resolve())
args.destination.write_text(json.dumps(compact, indent=2)+'\n')
