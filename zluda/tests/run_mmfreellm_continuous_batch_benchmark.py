#!/usr/bin/env python3
"""Qualify MatMulFreeLM CUDA throughput with bounded windowed batching."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Sequence

from mmfreellm_continuous_batch import (
    BackendBatchResult,
    BenchmarkConfig,
    GeneratedOutput,
    RequestSpec,
    run_windowed_batch,
    summarize_run,
)


DEFAULT_MODEL = (
    "/root/.cache/huggingface/hub/models--ridger--MMfreeLM-2.7B/"
    "snapshots/77deff0c1c9ac79aa51eb3ab7dd34fc375bf9324"
)
DEFAULT_REPO = Path("/root/matmulfreellm")
JSON_PREFIX = "MMFREELM_CONTINUOUS_BATCH_JSON="


class CudaGenerationBackend:
    def __init__(
        self,
        *,
        torch_module,
        tokenizer,
        model,
        model_path: str,
        dtype_name: str,
        parameter_count: int,
        bitlinear_backend: str,
        clock=time.perf_counter,
    ) -> None:
        self.torch = torch_module
        self.tokenizer = tokenizer
        self.model = model
        self.model_path = model_path
        self.dtype_name = dtype_name
        self.parameter_count = parameter_count
        self.bitlinear_backend = bitlinear_backend
        self.clock = clock

    @classmethod
    def load(cls, args) -> "CudaGenerationBackend":
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if not args.online:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, str(args.repo_root))

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import mmfreelm  # noqa: F401

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        load_options = {"local_files_only": not args.online}
        tokenizer = AutoTokenizer.from_pretrained(args.model, **load_options)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("tokenizer has neither pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        dtype = {"half": torch.float16, "bf16": torch.bfloat16}[args.dtype]
        model = AutoModelForCausalLM.from_pretrained(
            args.model, **load_options
        ).eval()
        model = model.to(device="cuda", dtype=dtype)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        return cls(
            torch_module=torch,
            tokenizer=tokenizer,
            model=model,
            model_path=args.model,
            dtype_name=args.dtype,
            parameter_count=parameter_count,
            bitlinear_backend=os.environ.get("MMFREELM_BITLINEAR_BACKEND", "default"),
        )

    def generate(self, requests: Sequence[RequestSpec]) -> BackendBatchResult:
        if not requests:
            raise ValueError("generation requires at least one request")
        requested_counts = {request.max_new_tokens for request in requests}
        if len(requested_counts) != 1 or next(iter(requested_counts)) <= 0:
            raise ValueError("microbatch requests must have one positive token count")
        requested_tokens = next(iter(requested_counts))
        prompts = [request.prompt for request in requests]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {name: tensor.to("cuda") for name, tensor in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])

        self.torch.cuda.reset_peak_memory_stats()
        self.torch.cuda.synchronize()
        started = self.clock()
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=requested_tokens,
                min_new_tokens=requested_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self.torch.cuda.synchronize()
        elapsed = self.clock() - started
        if elapsed <= 0 or not math.isfinite(elapsed):
            raise RuntimeError("generation service time must be finite and positive")
        if int(output_ids.shape[0]) != len(requests):
            raise RuntimeError("generation output row count does not match requests")

        decoded = self.tokenizer.batch_decode(
            output_ids.detach().cpu(), skip_special_tokens=True
        )
        if len(decoded) != len(requests):
            raise RuntimeError("decoded output count does not match requests")
        outputs = []
        for row, (request, text) in enumerate(zip(requests, decoded)):
            generated = output_ids[row, input_width:].detach().cpu().tolist()
            token_ids = tuple(int(token_id) for token_id in generated)
            if len(token_ids) != requested_tokens:
                raise RuntimeError(
                    f"request {request.request_id} generated-token count mismatch"
                )
            if not text or not text.startswith(request.prompt):
                raise RuntimeError(
                    f"request {request.request_id} decoded output does not retain prompt"
                )
            outputs.append(GeneratedOutput(text, token_ids))
        return BackendBatchResult(
            outputs=tuple(outputs),
            service_seconds=elapsed,
            peak_cuda_memory_bytes=int(self.torch.cuda.max_memory_allocated()),
        )

    def warmup(
        self, batch_size: int, prompt: str, max_new_tokens: int, runs: int
    ) -> None:
        requests = tuple(
            RequestSpec(index, prompt, max_new_tokens) for index in range(batch_size)
        )
        for _ in range(runs):
            self.generate(requests)


def qualify_summaries(run_summaries: Sequence[dict], threshold: float) -> dict:
    if len(run_summaries) != 2:
        raise ValueError("qualification requires exactly two measured runs")
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("qualification threshold must be finite and positive")
    failure_reasons = []
    for summary in run_summaries:
        tps = float(summary["aggregate_tokens_per_second"])
        if not math.isfinite(tps) or tps < threshold:
            failure_reasons.append(
                f"run {summary['run_index']} aggregate TPS {tps:.6f} "
                f"is below {threshold:.6f}"
            )

    token_maps = [
        {
            int(request["request_id"]): tuple(request["generated_token_ids"])
            for request in summary["requests"]
        }
        for summary in run_summaries
    ]
    deterministic = token_maps[0] == token_maps[1]
    if not deterministic:
        failure_reasons.append("generated token determinism mismatch")
    return {
        "qualification_passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "deterministic_generated_token_ids": deterministic,
    }


def format_json_record(record: dict) -> str:
    return JSON_PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"))


def write_result_exclusive(path: Path, record: dict) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite result file: {path}") from error


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Qualify MatMulFreeLM 2.7B aggregate CUDA throughput"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--request-count", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--queue-timeout-ms", type=float, default=2.0)
    parser.add_argument("--interarrival-ms", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--min-aggregate-tps", type=float, default=200.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("half", "bf16"), default="half")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args(arguments)


def config_from_args(args) -> BenchmarkConfig:
    return BenchmarkConfig(
        request_count=args.request_count,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        queue_timeout_ms=args.queue_timeout_ms,
        interarrival_ms=args.interarrival_ms,
        min_aggregate_tps=args.min_aggregate_tps,
    )


def validate_cli(args) -> None:
    if args.device != "cuda":
        raise ValueError("CUDA is mandatory for continuous-batch qualification")
    config_from_args(args).validate()
    if args.runs != 2:
        raise ValueError("qualification requires exactly two measured runs")
    if args.warmup_runs < 1:
        raise ValueError("qualification requires at least one warmup run")
    if not args.prompt:
        raise ValueError("prompt must be non-empty")
    if not args.repo_root.is_dir():
        raise ValueError(f"MatMulFreeLM repository does not exist: {args.repo_root}")
    if not args.online and not Path(args.model).exists():
        raise ValueError(f"offline model path does not exist: {args.model}")
    backend = os.environ.get("MMFREELM_BITLINEAR_BACKEND", "default")
    if backend != "default":
        raise ValueError("MMFREELM_BITLINEAR_BACKEND must be default")
    adapter = os.environ.get("MMFREELM_TERNIP_ADAPTER", "disabled").lower()
    if adapter not in ("", "0", "false", "disabled"):
        raise ValueError("MMFREELM_TERNIP_ADAPTER must be disabled")
    if args.result_json is not None and not args.result_json.parent.is_dir():
        raise ValueError("result JSON parent directory does not exist")


def run_benchmark(args) -> dict:
    validate_cli(args)
    config = config_from_args(args)
    backend = CudaGenerationBackend.load(args)
    backend.warmup(
        config.max_batch_size,
        args.prompt,
        config.max_new_tokens,
        args.warmup_runs,
    )
    run_summaries = []
    prompts = [args.prompt] * config.request_count
    for run_index in range(1, args.runs + 1):
        result = run_windowed_batch(
            run_index, prompts, backend, config
        )
        run_summaries.append(summarize_run(result, config))
    qualification = qualify_summaries(
        run_summaries, config.min_aggregate_tps
    )
    return {
        "schema": "matmulfreellm-continuous-batch-benchmark-v1",
        "validated": True,
        **qualification,
        "model": args.model,
        "device": "cuda",
        "dtype": args.dtype,
        "parameter_count": backend.parameter_count,
        "prompt": args.prompt,
        "request_count": config.request_count,
        "max_batch_size": config.max_batch_size,
        "max_new_tokens": config.max_new_tokens,
        "queue_timeout_ms": config.queue_timeout_ms,
        "interarrival_ms": config.interarrival_ms,
        "qualification_runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "min_aggregate_tps": config.min_aggregate_tps,
        "bitlinear_backend": backend.bitlinear_backend,
        "ternip_adapter": "disabled",
        "fpga_tps_reported": False,
        "runs": run_summaries,
    }


def main(arguments=None) -> int:
    try:
        args = parse_args(arguments)
        record = run_benchmark(args)
        if args.result_json is not None:
            write_result_exclusive(args.result_json, record)
    except Exception as error:
        print(f"run_mmfreellm_continuous_batch_benchmark: {error}", file=sys.stderr)
        return 2
    print(format_json_record(record))
    return 0 if record["qualification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
