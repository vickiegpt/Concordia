#!/usr/bin/env python3
"""Collect stable, per-submission device durations, never subtract clock epochs."""
import argparse
from collections import deque
import gc
import json
import math
import mmap
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

EVENT = struct.Struct("<QIIIIQ")
MAGIC = 0x4847505544494147
RING_OFFSET, RING_SIZE = 0x10000, 192 * EVENT.size
PHASES = ("dst_prepare", "input_copy", "compute", "output_copy", "cleanup",
          "completion_publish", "device_before_publish")


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, min(len(values)-1, math.ceil(len(values)*fraction)-1))]


def valid_capture(returncode, timed_out, complete_count, incomplete, gaps, expected, build_matches):
    expected_matches = expected is not None and complete_count == expected
    return bool(returncode == 0 and not timed_out and complete_count and not incomplete and
                (expected is None or expected_matches) and
                (not gaps or expected_matches) and build_matches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--sample-ms", type=float, default=5)
    parser.add_argument("--expected-submissions", type=int)
    parser.add_argument("--require-build", type=lambda value: int(value, 0))
    parser.add_argument("--device", type=int, choices=range(4), default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0.5 <= args.sample_ms <= 100:
        parser.error('--sample-ms must be between 0.5 and 100')
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a benchmark command is required")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(f"/dev/hetgpu_sifive_mbox_ddr_coh{args.device}", os.O_RDONLY)
    base, = struct.unpack("<Q", os.pread(fd, 8, 0x02004000))
    if base != int(os.environ["HETGPU_SIFIVE_SHARED_DDR_BASE"], 0):
        raise SystemExit("shared DDR allocation mismatch")
    ring_offset = RING_OFFSET + args.device * 0x2000
    region = mmap.mmap(fd, ring_offset+RING_SIZE, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ)

    def sample():
        a = region[ring_offset:ring_offset+RING_SIZE]
        b = region[ring_offset:ring_offset+RING_SIZE]
        for offset in range(0, RING_SIZE, EVENT.size):
            raw = a[offset:offset+EVENT.size]
            if raw != b[offset:offset+EVENT.size]:
                continue
            magic, index, status, job, aux, seq = EVENT.unpack(raw)
            if magic == MAGIC:
                yield (index, seq, status, job, aux)

    # Ignore old firmware/job events already present at collection start.
    seen = set(sample())
    recent = deque(seen)
    seen_limit = 4 * (RING_SIZE // EVENT.size)
    records = []
    started = time.monotonic()
    timed_out = False
    last_sample = None
    max_sample_gap = 0.0
    with output.with_suffix(".log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        finished = None
        # Capture records are acyclic. Avoid full-heap scans while the tiny ring runs.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            while True:
                now = time.monotonic()
                if last_sample is not None:
                    max_sample_gap = max(max_sample_gap, now-last_sample)
                last_sample = now
                for event in sample():
                    if event not in seen:
                        seen.add(event)
                        recent.append(event)
                        # Retain several ring rotations, not an ever-growing hash table.
                        if len(recent) > seen_limit:
                            seen.remove(recent.popleft())
                        records.append(dict(zip(("index", "seq", "status", "job", "aux"), event),
                                            observed_seconds=now-started))
                if child.poll() is not None:
                    if finished is None:
                        finished = now
                    if now-finished > 1:
                        break
                elif now-started > args.timeout:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                time.sleep(args.sample_ms / 1000)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            region.close()
            os.close(fd)
            if gc_was_enabled:
                gc.enable()
    submissions, capabilities, builds = {}, {}, {}
    for record in records:
        status = record["status"]
        if 0x5320 <= status <= 0x532f:
            capabilities[hex(status)] = record["aux"]
        if record["job"] != 5:
            continue
        if status == 0x5330:
            builds[str(record["seq"])] = record["aux"]
        if status == 0x5140 or 0x5310 <= status <= 0x5316:
            row = submissions.setdefault(str(record["seq"]), {})
            if 0x5310 <= status <= 0x5316:
                row[PHASES[status-0x5310]] = record["aux"]
    complete_ids = {seq for seq, row in submissions.items()
                    if len(row) == len(PHASES) and max(row.values()) < 0xffffffff}
    complete = [row for seq, row in submissions.items() if seq in complete_ids]
    incomplete = {seq: row for seq, row in submissions.items() if seq not in complete_ids}
    stats = {}
    for phase in PHASES:
        values = [row[phase] for row in complete]
        if values:
            stats[phase] = {"sum_us": sum(values), "p50_us": percentile(values, .5),
                            "p99_us": percentile(values, .99), "max_us": max(values)}
    indices = sorted(set(record["index"] for record in records))
    gaps = [(a, b) for a, b in zip(indices, indices[1:]) if b != a+1]
    result = dict(command=command, device=args.device, returncode=child.returncode, timed_out=timed_out,
                  sample_ms=args.sample_ms,
                  max_sample_gap_ms=max_sample_gap*1000,
                  recent_event_limit=seen_limit,
                  complete_submissions=len(complete), incomplete_submissions=incomplete,
                  observed_index_gaps=gaps, phases_us=stats, capabilities=capabilities,
                  submissions=submissions, raw_records=records,
                  limitations=["Double-read stability is not a source-side atomic snapshot guarantee.",
                               "Completion publish excludes subsequent diagnostic writes.",
                               "Host/device elapsed differences include dispatch and polling, not IRQ alone."])
    expected_matches = args.expected_submissions is not None and len(complete) == args.expected_submissions
    result["expected_submissions"] = args.expected_submissions
    result["expected_count_matches"] = expected_matches
    result["submission_builds"] = builds
    result["build_matches"] = args.require_build is None or (
        set(builds) == set(submissions) and all(value == args.require_build for value in builds.values()))
    result["capture_valid"] = valid_capture(child.returncode, timed_out, len(complete),
        incomplete, gaps, args.expected_submissions, result["build_matches"])
    output.write_text(json.dumps(result, indent=2)+"\n")
    summary = {key: value for key, value in result.items()
               if key not in ("raw_records", "submissions", "submission_builds", "observed_index_gaps")}
    summary["observed_index_gap_count"] = len(gaps)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if result["capture_valid"] else 1)


if __name__ == "__main__":
    main()
