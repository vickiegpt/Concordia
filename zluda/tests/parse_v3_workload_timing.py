#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


KIMI_TIMING_RE = re.compile(
    r"(?:\beval\s+time\b|\bgeneration\b)[^\n]*?=\s*"
    r"(?P<eval_ms>[0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*"
    r"(?P<tokens>[0-9]+)\s*(?:runs?|tokens?)?[^\n]*?"
    r"(?P<tps>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?:tokens\s+per\s+second|tok/s|tokens/s)",
    re.IGNORECASE,
)
MMFREELM_ITERATION_RE = re.compile(
    r"Iteration\s+(?P<iteration>[0-9]+):\s*"
    r"(?P<tps>[0-9]+(?:\.[0-9]+)?)\s+tok/s,\s*"
    r"(?P<gflops>[0-9]+(?:\.[0-9]+)?)\s+GFLOPS/s,\s*"
    r"(?P<seconds>[0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)
MMFREELM_RUNS_RE = re.compile(r"Total\s+Runs:\s*(?P<runs>[0-9]+)", re.IGNORECASE)
MMFREELM_TOKENS_RE = re.compile(
    r"Tokens\s+Generated:\s*.*?Total:\s*(?P<tokens>[0-9]+)",
    re.IGNORECASE | re.DOTALL,
)
MMFREELM_MEAN_RE = re.compile(
    r"Tokens\s+Per\s+Second\s*\(TPS\):\s*.*?"
    r"Mean:\s*(?P<mean>[0-9]+(?:\.[0-9]+)?)\s+tok/s",
    re.IGNORECASE | re.DOTALL,
)


def parse_kimi_timing(text: str) -> dict:
    match = None
    for line in text.splitlines():
        if "prompt eval time" in line.lower():
            continue
        candidate = KIMI_TIMING_RE.search(line)
        if candidate:
            match = candidate
    if match is None:
        raise ValueError("no recognized Kimi/llama generation timing")

    eval_ms = float(match.group("eval_ms"))
    tokens = int(match.group("tokens"))
    tps = float(match.group("tps"))
    if eval_ms <= 0 or tokens <= 0 or tps <= 0:
        raise ValueError("Kimi/llama timing must contain positive metrics")
    expected_tps = tokens / (eval_ms / 1000.0)
    if not math.isclose(tps, expected_tps, rel_tol=5e-4, abs_tol=0.02):
        raise ValueError(
            f"Kimi/llama timing TPS {tps} contradicts tokens/eval_ms {expected_tps}"
        )
    return {
        "workload_kind": "kimi",
        "validated": True,
        "tokens": tokens,
        "eval_ms": eval_ms,
        "tokens_per_second": tps,
    }


def parse_matmulfreellm_timing(text: str) -> dict:
    iterations = [
        {
            "iteration": int(match.group("iteration")),
            "tokens_per_second": float(match.group("tps")),
            "gflops_per_second": float(match.group("gflops")),
            "generation_seconds": float(match.group("seconds")),
        }
        for match in MMFREELM_ITERATION_RE.finditer(text)
    ]
    runs_match = MMFREELM_RUNS_RE.search(text)
    tokens_match = MMFREELM_TOKENS_RE.search(text)
    mean_match = MMFREELM_MEAN_RE.search(text)
    if not iterations or runs_match is None or tokens_match is None or mean_match is None:
        raise ValueError("no recognized MatMulFreeLM benchmark timing summary")
    if any(
        item["tokens_per_second"] <= 0
        or item["gflops_per_second"] <= 0
        or item["generation_seconds"] <= 0
        for item in iterations
    ):
        raise ValueError("MatMulFreeLM iteration timing must contain positive metrics")

    runs = int(runs_match.group("runs"))
    total_tokens = int(tokens_match.group("tokens"))
    reported_mean = float(mean_match.group("mean"))
    if runs != len(iterations) or runs <= 0:
        raise ValueError(
            f"MatMulFreeLM Total Runs {runs} contradicts {len(iterations)} iteration records"
        )
    if total_tokens <= 0:
        raise ValueError("MatMulFreeLM total generated tokens must be positive")
    calculated_mean = sum(item["tokens_per_second"] for item in iterations) / runs
    if not math.isclose(reported_mean, calculated_mean, rel_tol=5e-4, abs_tol=0.02):
        raise ValueError(
            f"MatMulFreeLM mean TPS {reported_mean} contradicts iterations {calculated_mean}"
        )
    return {
        "workload_kind": "matmulfreellm",
        "validated": True,
        "runs": runs,
        "total_tokens": total_tokens,
        "mean_tokens_per_second": reported_mean,
        "iterations": iterations,
    }


def parse_workload_timing(text: str, kind: str) -> dict:
    if kind == "kimi":
        return parse_kimi_timing(text)
    if kind == "matmulfreellm":
        return parse_matmulfreellm_timing(text)
    if kind != "auto":
        raise ValueError(f"unsupported workload timing kind: {kind}")
    errors = []
    for parser in (parse_kimi_timing, parse_matmulfreellm_timing):
        try:
            return parser(text)
        except ValueError as error:
            errors.append(str(error))
    raise ValueError("no recognized timing: " + "; ".join(errors))


def parse_args(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("baseline", "enabled"))
    parser.add_argument("--kind", default="auto", choices=("auto", "kimi", "matmulfreellm"))
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def run_cli(arguments=None):
    args = parse_args(arguments)
    stdout = args.stdout.read_text(encoding="utf-8", errors="replace")
    stderr = args.stderr.read_text(encoding="utf-8", errors="replace")
    timing = parse_workload_timing(stdout + "\n" + stderr, args.kind)
    timing["label"] = args.label
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(timing, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(timing, sort_keys=True))


if __name__ == "__main__":
    run_cli()
