#!/usr/bin/env python3
"""Core scheduling records and metrics for MatMulFreeLM batching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from threading import Condition, Lock, Thread
import time
from typing import Callable, Protocol, Sequence


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


class GenerationBackend(Protocol):
    def generate(self, requests: Sequence[RequestSpec]) -> BackendBatchResult:
        raise NotImplementedError


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


class WindowedRequestQueue:
    def __init__(
        self,
        max_batch_size: int,
        timeout_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self._max_batch_size = max_batch_size
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._pending: deque[QueuedRequest] = deque()
        self._closed = False
        self._condition = Condition()

    def submit(self, request: RequestSpec) -> QueuedRequest:
        with self._condition:
            while len(self._pending) >= self._max_batch_size and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("cannot submit after producer closure")
            queued = QueuedRequest(request, self._clock())
            self._pending.append(queued)
            self._condition.notify()
            return queued

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def take_batch(self) -> tuple[QueuedRequest, ...] | None:
        with self._condition:
            while True:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending and self._closed:
                    return None
                oldest_age = max(0.0, self._clock() - self._pending[0].enqueued_at)
                count = batch_size_if_ready(
                    len(self._pending),
                    oldest_age,
                    self._max_batch_size,
                    self._timeout_seconds,
                    self._closed,
                )
                if count:
                    batch = tuple(self._pending.popleft() for _ in range(count))
                    self._condition.notify_all()
                    return batch
                remaining = max(0.0, self._timeout_seconds - oldest_age)
                self._condition.wait(remaining)


def run_windowed_batch(
    run_index: int,
    prompts: Sequence[str],
    backend: GenerationBackend,
    config: BenchmarkConfig,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> RunResult:
    config.validate()
    if len(prompts) != config.request_count:
        raise ValueError("prompt count must equal configured request count")

    request_queue = WindowedRequestQueue(
        config.max_batch_size, config.queue_timeout_ms / 1000, clock
    )
    results: list[RequestResult] = []
    microbatches: list[MicrobatchResult] = []
    errors: list[Exception] = []
    state_lock = Lock()

    def record_error(error: Exception) -> None:
        with state_lock:
            errors.append(error)
        request_queue.close()

    def produce() -> None:
        try:
            for request_id, prompt in enumerate(prompts):
                request_queue.submit(
                    RequestSpec(request_id, prompt, config.max_new_tokens)
                )
                if request_id + 1 < len(prompts) and config.interarrival_ms:
                    sleeper(config.interarrival_ms / 1000)
        except Exception as error:
            record_error(error)
        finally:
            request_queue.close()

    def work() -> None:
        try:
            while True:
                queued = request_queue.take_batch()
                if queued is None:
                    return
                dispatched_at = clock()
                batch_result = backend.generate(
                    tuple(request.spec for request in queued)
                )
                completed_at = clock()
                if len(batch_result.outputs) != len(queued):
                    raise RuntimeError("backend output count does not match microbatch")
                request_results = tuple(
                    RequestResult(
                        request_id=request.spec.request_id,
                        prompt=request.spec.prompt,
                        output=output.output,
                        generated_token_ids=output.generated_token_ids,
                        enqueued_at=request.enqueued_at,
                        dispatched_at=dispatched_at,
                        completed_at=completed_at,
                    )
                    for request, output in zip(queued, batch_result.outputs)
                )
                microbatch = MicrobatchResult(
                    request_ids=tuple(
                        request.spec.request_id for request in queued
                    ),
                    dispatched_at=dispatched_at,
                    completed_at=completed_at,
                    service_seconds=batch_result.service_seconds,
                    peak_cuda_memory_bytes=batch_result.peak_cuda_memory_bytes,
                )
                with state_lock:
                    results.extend(request_results)
                    microbatches.append(microbatch)
        except Exception as error:
            record_error(error)

    producer = Thread(target=produce, name=f"mmfreelm-producer-{run_index}")
    worker = Thread(target=work, name=f"mmfreelm-worker-{run_index}")
    producer.start()
    worker.start()
    producer.join()
    worker.join()

    if errors:
        raise errors[0]
    ordered_results = tuple(sorted(results, key=lambda result: result.request_id))
    run = RunResult(
        run_index=run_index,
        first_enqueued_at=min(result.enqueued_at for result in ordered_results),
        last_completed_at=max(result.completed_at for result in ordered_results),
        requests=ordered_results,
        microbatches=tuple(microbatches),
    )
    validate_run(run, config)
    return run


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
