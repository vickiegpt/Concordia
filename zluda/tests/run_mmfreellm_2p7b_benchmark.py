#!/usr/bin/env python3
"""Time real MatMulFreeLLM generation for CPU/CUDA or the TernIP adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import NamedTuple


DEFAULT_MODEL = (
    "/root/.cache/huggingface/hub/models--ridger--MMfreeLM-2.7B/"
    "snapshots/77deff0c1c9ac79aa51eb3ab7dd34fc375bf9324"
)


class MeasuredRun(NamedTuple):
    iteration: int
    generated_tokens: int
    generation_seconds: float
    tokens_per_second: float
    gflops_per_second: float


def summarize_runs(runs: list[MeasuredRun], *, parameter_count: int) -> dict:
    if not runs:
        raise ValueError("benchmark requires at least one measured run")
    if parameter_count <= 0:
        raise ValueError("parameter count must be positive")
    if any(
        run.generated_tokens <= 0
        or run.generation_seconds <= 0
        or run.tokens_per_second <= 0
        or run.gflops_per_second <= 0
        for run in runs
    ):
        raise ValueError("all benchmark measurements must be positive")
    return {
        "schema": "matmulfreellm-generation-benchmark-v1",
        "validated": True,
        "parameter_count": parameter_count,
        "total_runs": len(runs),
        "total_generated_tokens": sum(run.generated_tokens for run in runs),
        "mean_tokens_per_second": statistics.mean(
            run.tokens_per_second for run in runs
        ),
        "median_tokens_per_second": statistics.median(
            run.tokens_per_second for run in runs
        ),
        "runs": [run._asdict() for run in runs],
    }


def format_text_report(runs: list[MeasuredRun], summary: dict) -> str:
    lines = [
        f"  Iteration {run.iteration}: {run.tokens_per_second:.2f} tok/s, "
        f"{run.gflops_per_second:.2f} GFLOPS/s, {run.generation_seconds:.4f}s"
        for run in runs
    ]
    lines.extend(
        [
            "Tokens Per Second (TPS):",
            f"  Mean:   {summary['mean_tokens_per_second']:.2f} tok/s",
            f"  Median: {summary['median_tokens_per_second']:.2f} tok/s",
            f"  Total Runs: {summary['total_runs']}",
            "Tokens Generated:",
            f"  Total: {summary['total_generated_tokens']}",
        ]
    )
    return "\n".join(lines)


def format_json_record(summary: dict) -> str:
    return "MMFREELM_BENCHMARK_JSON=" + json.dumps(
        summary, sort_keys=True, separators=(",", ":")
    )


def extract_generated_token_ids(
    output_ids, *, prompt_tokens: int, max_new_tokens: int
) -> list[int]:
    generated = output_ids[0][prompt_tokens:]
    if hasattr(generated, "detach"):
        generated = generated.detach().cpu().tolist()
    token_ids = [int(token_id) for token_id in generated]
    if len(token_ids) != max_new_tokens:
        raise RuntimeError(
            "generated-token count does not match requested max_new_tokens"
        )
    return token_ids


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Benchmark real MatMulFreeLLM 2.7B generation without model-load time"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repo-root", type=Path, default=Path("/root/matmulfreellm"))
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float", "half", "bf16"), default="float")
    parser.add_argument("--online", action="store_true")
    return parser.parse_args(arguments)


def _synchronize(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_benchmark(args) -> tuple[list[MeasuredRun], dict]:
    if args.max_new_tokens <= 0 or args.runs <= 0 or args.warmup_runs < 0:
        raise ValueError("token count and runs must be positive; warmups may be zero")
    if not args.repo_root.is_dir():
        raise ValueError(f"MatMulFreeLLM repository does not exist: {args.repo_root}")
    model_path = Path(args.model)
    if not args.online and not model_path.exists():
        raise ValueError(f"offline model path does not exist: {model_path}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.repo_root))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import mmfreelm  # noqa: F401

    dtype = {
        "float": torch.float32,
        "half": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    if args.device == "cpu" and dtype == torch.float16:
        raise ValueError("CPU float16 generation is unsupported; use float or bf16")

    load_options = {"local_files_only": not args.online}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_options)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_options).eval()
    model = model.to(device=args.device, dtype=dtype)
    encoded = tokenizer(args.prompt, return_tensors="pt", padding=False)
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    generation_options = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    output_ids = None
    with torch.no_grad():
        for _ in range(args.warmup_runs):
            output_ids = model.generate(**encoded, **generation_options)
        measurements = []
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        for iteration in range(1, args.runs + 1):
            _synchronize(torch, args.device)
            started = time.perf_counter()
            output_ids = model.generate(**encoded, **generation_options)
            _synchronize(torch, args.device)
            elapsed = time.perf_counter() - started
            generated_tokens = int(output_ids.shape[-1]) - prompt_tokens
            if generated_tokens <= 0 or elapsed <= 0:
                raise RuntimeError("generation produced no measurable tokens")
            tps = generated_tokens / elapsed
            estimated_gflops = 2 * parameter_count * generated_tokens / elapsed / 1e9
            measurements.append(
                MeasuredRun(iteration, generated_tokens, elapsed, tps, estimated_gflops)
            )

    assert output_ids is not None
    output = tokenizer.batch_decode(output_ids.detach().cpu(), skip_special_tokens=True)[0]
    generated_token_ids = extract_generated_token_ids(
        output_ids,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    summary = summarize_runs(measurements, parameter_count=parameter_count)
    summary.update(
        {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "prompt": args.prompt,
            "output": output,
            "generated_token_ids": generated_token_ids,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "bitlinear_backend": os.environ.get("MMFREELM_BITLINEAR_BACKEND", "default"),
            "ternip_adapter": os.environ.get("MMFREELM_TERNIP_ADAPTER"),
        }
    )
    return measurements, summary


def main(arguments=None) -> int:
    args = parse_args(arguments)
    runs, summary = run_benchmark(args)
    print(f"Input: {args.prompt}")
    print(f"Output: {summary['output']}")
    print(format_text_report(runs, summary))
    print(format_json_record(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
