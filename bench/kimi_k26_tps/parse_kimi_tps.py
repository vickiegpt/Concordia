#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from pathlib import Path

CSV_FIELDS = [
    "case",
    "status",
    "tps",
    "tokens",
    "total_ms",
    "aof_bytes",
    "runner",
    "model",
    "gpu",
    "commit",
    "checkpoint_markers",
    "message",
]

GENERATION_LINE_RE = re.compile(
    r"""
    (?:\beval\s+time\b|\bgeneration\b)[^\n]*?
    =
    \s*(?P<eval_ms>[0-9]+(?:\.[0-9]+)?)\s*ms
    \s*/\s*
    (?P<tokens>[0-9]+)
    \s*(?:runs?|tokens?)?
    [^\n]*?
    (?P<tps>[0-9]+(?:\.[0-9]+)?)
    \s*(?:tokens\s+per\s+second|tok/s|tokens/s)
    """,
    re.IGNORECASE | re.VERBOSE,
)

TOTAL_RE = re.compile(
    r"""
    (?:
      ^|\n
    )
    [^\n]*total\s+time[^\n]*?
    =
    \s*(?P<total_ms>[0-9]+(?:\.[0-9]+)?)\s*ms
    """,
    re.IGNORECASE | re.VERBOSE,
)

CHECKPOINT_MARKER_RE = re.compile(
    r"kimi concordia|concordia.*checkpoint|checkpointed|base snapshot|dirty scan|aof",
    re.IGNORECASE,
)


def main() -> int:
    args = parse_args()
    stdout_text = read_text(args.stdout)
    stderr_text = read_text(args.stderr)
    combined = stdout_text + "\n" + stderr_text

    parsed = parse_timings(combined)
    total_ms = parsed["total_ms"] if parsed["total_ms"] is not None else args.total_ms
    checkpoint_markers = count_checkpoint_markers(combined)
    aof_bytes = path_bytes(args.aof)

    status, message = resolve_status(args.status, args.exit_code, parsed)
    row = {
        "case": args.case,
        "status": status,
        "tps": round_or_zero(parsed["tps"]),
        "tokens": parsed["tokens"] or 0,
        "total_ms": round_or_zero(total_ms),
        "aof_bytes": aof_bytes,
        "runner": args.runner,
        "model": args.model,
        "gpu": args.gpu,
        "commit": args.commit,
        "checkpoint_markers": checkpoint_markers,
        "message": message,
    }

    append_csv(args.csv, row)
    append_jsonl(args.jsonl, row)
    print(json.dumps(row, sort_keys=True))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Parse Kimi K2.6 BitNet TPS evidence logs")
    parser.add_argument("--case", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--total-ms", type=float, default=0.0)
    parser.add_argument("--aof", default="")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--status", default="")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--jsonl", required=True)
    return parser.parse_args()


def read_text(path):
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8", errors="replace")


def parse_timings(text):
    eval_match = None
    for line in text.splitlines():
        if "prompt eval time" in line.lower():
            continue
        match = GENERATION_LINE_RE.search(line)
        if not match:
            continue
        eval_match = match

    total_match = None
    for match in TOTAL_RE.finditer(text):
        total_match = match

    return {
        "tps": float(eval_match.group("tps")) if eval_match else None,
        "tokens": int(eval_match.group("tokens")) if eval_match else None,
        "total_ms": float(total_match.group("total_ms")) if total_match else None,
    }


def count_checkpoint_markers(text):
    return sum(1 for line in text.splitlines() if CHECKPOINT_MARKER_RE.search(line))


def path_bytes(path):
    if not path:
        return 0
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def resolve_status(status_override, exit_code, parsed):
    if status_override:
        if status_override == "pass":
            return "pass", "ok"
        return status_override, status_override
    if exit_code != 0:
        return "run_failed", f"runner_exit_{exit_code}"
    if parsed["tps"] is None or parsed["tokens"] is None:
        return "missing_timing", "no_generation_timing"
    return "pass", "ok"


def round_or_zero(value):
    if value is None:
        return 0
    rounded = round(float(value), 6)
    return int(rounded) if rounded.is_integer() else rounded


def append_csv(path, row):
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in CSV_FIELDS})


def append_jsonl(path, row):
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
