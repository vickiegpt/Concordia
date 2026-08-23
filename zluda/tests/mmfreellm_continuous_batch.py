#!/usr/bin/env python3
"""Core scheduling records and metrics for MatMulFreeLM batching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class BenchmarkConfig:
    request_count: int = 64
    max_batch_size: int = 16
    max_new_tokens: int = 8
    queue_timeout_ms: float = 2.0
    interarrival_ms: float = 0.0
    min_aggregate_tps: float = 200.0

    def validate(self) -> None:
        if self.request_count <= 0:
            raise ValueError("request count must be positive")
        if not 1 <= self.max_batch_size <= 16:
            raise ValueError("max batch size must be in 1..16")
        if self.max_new_tokens <= 0:
            raise ValueError("generated-token count must be positive")
        if self.queue_timeout_ms < 0 or not math.isfinite(self.queue_timeout_ms):
            raise ValueError("queue timeout must be finite and non-negative")
        if self.interarrival_ms < 0 or not math.isfinite(self.interarrival_ms):
            raise ValueError("interarrival delay must be finite and non-negative")
        if self.min_aggregate_tps <= 0 or not math.isfinite(
            self.min_aggregate_tps
        ):
            raise ValueError("acceptance threshold must be finite and positive")


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt: str
    max_new_tokens: int


@dataclass(frozen=True)
class QueuedRequest:
    spec: RequestSpec
    enqueued_at: float


@dataclass(frozen=True)
class GeneratedOutput:
    output: str
    generated_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class BackendBatchResult:
    outputs: tuple[GeneratedOutput, ...]
    service_seconds: float
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class RequestResult:
    request_id: int
    prompt: str
    output: str
    generated_token_ids: tuple[int, ...]
    enqueued_at: float
    dispatched_at: float
    completed_at: float


@dataclass(frozen=True)
class MicrobatchResult:
    request_ids: tuple[int, ...]
    dispatched_at: float
    completed_at: float
    service_seconds: float
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class RunResult:
    run_index: int
    first_enqueued_at: float
    last_completed_at: float
    requests: tuple[RequestResult, ...]
    microbatches: tuple[MicrobatchResult, ...]


def batch_size_if_ready(
    pending_count: int,
    oldest_age_seconds: float,
    max_batch_size: int,
    timeout_seconds: float,
    producer_closed: bool,
) -> int:
    if pending_count <= 0:
        return 0
    if pending_count >= max_batch_size:
        return max_batch_size
    if producer_closed or oldest_age_seconds >= timeout_seconds:
        return pending_count
    return 0


def percentile(values: Sequence[float], percent: float) -> float:
    if not values or not 0 <= percent <= 100:
        raise ValueError("percentile requires values and a percentage in 0..100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _finite_ordered(*values: float) -> bool:
    return all(math.isfinite(value) for value in values) and all(
        left <= right for left, right in zip(values, values[1:])
    )


def validate_run(run: RunResult, config: BenchmarkConfig) -> None:
    config.validate()
    if not _finite_ordered(run.first_enqueued_at, run.last_completed_at) or (
        run.last_completed_at <= run.first_enqueued_at
    ):
        raise ValueError("run timing must be finite and positive")

    request_ids = [request.request_id for request in run.requests]
    expected_ids = list(range(config.request_count))
    if len(request_ids) != config.request_count or sorted(request_ids) != expected_ids:
        raise ValueError("request IDs must be unique and complete")

    for request in run.requests:
        if not request.prompt:
            raise ValueError(f"request {request.request_id} has an empty prompt")
        if not request.output:
            raise ValueError(f"request {request.request_id} has empty decoded output")
        if not request.output.startswith(request.prompt):
            raise ValueError(f"request {request.request_id} output does not retain prompt")
        if len(request.generated_token_ids) != config.max_new_tokens:
            raise ValueError(
                f"request {request.request_id} generated-token count mismatch"
            )
        if not all(isinstance(token_id, int) for token_id in request.generated_token_ids):
            raise ValueError(f"request {request.request_id} has non-integer token IDs")
        if not _finite_ordered(
            run.first_enqueued_at,
            request.enqueued_at,
            request.dispatched_at,
            request.completed_at,
            run.last_completed_at,
        ):
            raise ValueError(f"request {request.request_id} timing is invalid")

    if not run.microbatches:
        raise ValueError("run must contain at least one microbatch")
    dispatched_ids: list[int] = []
    for index, microbatch in enumerate(run.microbatches):
        if not 1 <= len(microbatch.request_ids) <= config.max_batch_size:
            raise ValueError(f"microbatch {index} size is outside configured bounds")
        dispatched_ids.extend(microbatch.request_ids)
        if not _finite_ordered(
            run.first_enqueued_at,
            microbatch.dispatched_at,
            microbatch.completed_at,
            run.last_completed_at,
        ) or microbatch.completed_at <= microbatch.dispatched_at:
            raise ValueError(f"microbatch {index} timing is invalid")
        if (
            not math.isfinite(microbatch.service_seconds)
            or microbatch.service_seconds <= 0
        ):
            raise ValueError(f"microbatch {index} service time must be positive")
        if microbatch.peak_cuda_memory_bytes < 0:
            raise ValueError(f"microbatch {index} peak memory must be non-negative")
    if sorted(dispatched_ids) != expected_ids or len(dispatched_ids) != len(expected_ids):
        raise ValueError("request IDs must appear in exactly one microbatch")


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def summarize_run(run: RunResult, config: BenchmarkConfig) -> dict:
    validate_run(run, config)
    elapsed = run.last_completed_at - run.first_enqueued_at
    total_generated_tokens = sum(
        len(request.generated_token_ids) for request in run.requests
    )
    queue_latencies = [
        request.dispatched_at - request.enqueued_at for request in run.requests
    ]
    service_latencies = [
        request.completed_at - request.dispatched_at for request in run.requests
    ]
    end_to_end_latencies = [
        request.completed_at - request.enqueued_at for request in run.requests
    ]
    return {
        "schema": "matmulfreellm-continuous-batch-run-v1",
        "run_index": run.run_index,
        "requested_requests": config.request_count,
        "completed_requests": len(run.requests),
        "failed_requests": config.request_count - len(run.requests),
        "total_generated_tokens": total_generated_tokens,
        "end_to_end_seconds": elapsed,
        "aggregate_tokens_per_second": total_generated_tokens / elapsed,
        "configured_max_batch_size": config.max_batch_size,
        "observed_batch_sizes": [
            len(microbatch.request_ids) for microbatch in run.microbatches
        ],
        "microbatch_count": len(run.microbatches),
        "microbatches": [
            {
                "request_ids": list(microbatch.request_ids),
                "dispatched_at": microbatch.dispatched_at,
                "completed_at": microbatch.completed_at,
                "service_seconds": microbatch.service_seconds,
                "peak_cuda_memory_bytes": microbatch.peak_cuda_memory_bytes,
            }
            for microbatch in run.microbatches
        ],
        "peak_cuda_memory_bytes": max(
            microbatch.peak_cuda_memory_bytes for microbatch in run.microbatches
        ),
        "latency_seconds": {
            "queue": _latency_summary(queue_latencies),
            "service": _latency_summary(service_latencies),
            "end_to_end": _latency_summary(end_to_end_latencies),
        },
        "requests": [
            {
                "request_id": request.request_id,
                "prompt": request.prompt,
                "output": request.output,
                "generated_token_ids": list(request.generated_token_ids),
                "generated_tokens": len(request.generated_token_ids),
                "enqueued_at": request.enqueued_at,
                "dispatched_at": request.dispatched_at,
                "completed_at": request.completed_at,
                "queue_seconds": request.dispatched_at - request.enqueued_at,
                "service_seconds": request.completed_at - request.dispatched_at,
                "end_to_end_seconds": request.completed_at - request.enqueued_at,
            }
            for request in run.requests
        ],
    }
