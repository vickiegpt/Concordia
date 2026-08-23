#!/usr/bin/env python3
import math
import unittest

from mmfreellm_continuous_batch import (
    BackendBatchResult,
    BenchmarkConfig,
    GeneratedOutput,
    MicrobatchResult,
    RequestResult,
    RunResult,
    batch_size_if_ready,
    percentile,
    run_windowed_batch,
    summarize_run,
    validate_run,
)


def valid_run() -> RunResult:
    requests = tuple(
        RequestResult(
            request_id=index,
            prompt="The quick brown fox",
            output="The quick brown fox jumps",
            generated_token_ids=(10, 11),
            enqueued_at=1.0,
            dispatched_at=1.1,
            completed_at=1.3,
        )
        for index in range(2)
    )
    return RunResult(
        run_index=1,
        first_enqueued_at=1.0,
        last_completed_at=1.3,
        requests=requests,
        microbatches=(
            MicrobatchResult(
                request_ids=(0, 1),
                dispatched_at=1.1,
                completed_at=1.3,
                service_seconds=0.2,
                peak_cuda_memory_bytes=1024,
            ),
        ),
    )


class CoreContractTests(unittest.TestCase):
    def test_batch_size_is_capped_at_sixteen(self):
        with self.assertRaisesRegex(ValueError, "1..16"):
            BenchmarkConfig(max_batch_size=17).validate()

    def test_dispatch_is_full_timeout_or_close_only(self):
        self.assertEqual(batch_size_if_ready(16, 0.0, 16, 0.002, False), 16)
        self.assertEqual(batch_size_if_ready(3, 0.003, 16, 0.002, False), 3)
        self.assertEqual(batch_size_if_ready(3, 0.0, 16, 0.002, True), 3)
        self.assertEqual(batch_size_if_ready(3, 0.001, 16, 0.002, False), 0)

    def test_percentile_is_linear_interpolation(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 95), 2.9)

    def test_summary_uses_tokens_over_end_to_end_time(self):
        summary = summarize_run(
            valid_run(), BenchmarkConfig(request_count=2, max_new_tokens=2)
        )

        self.assertEqual(summary["total_generated_tokens"], 4)
        self.assertAlmostEqual(summary["aggregate_tokens_per_second"], 4 / 0.3)
        self.assertEqual(summary["observed_batch_sizes"], [2])
        self.assertAlmostEqual(summary["latency_seconds"]["queue"]["p50"], 0.1)

    def test_validation_rejects_duplicate_ids(self):
        run = valid_run()
        broken = RunResult(
            run_index=1,
            first_enqueued_at=1.0,
            last_completed_at=1.3,
            requests=(run.requests[0], run.requests[0]),
            microbatches=run.microbatches,
        )

        with self.assertRaisesRegex(ValueError, "request IDs"):
            validate_run(broken, BenchmarkConfig(request_count=2, max_new_tokens=2))

    def test_validation_rejects_bad_token_count_empty_output_and_time(self):
        config = BenchmarkConfig(request_count=2, max_new_tokens=2)
        run = valid_run()
        replacements = (
            (
                "generated-token count",
                RequestResult(
                    1,
                    run.requests[1].prompt,
                    run.requests[1].output,
                    (10,),
                    1.0,
                    1.1,
                    1.3,
                ),
            ),
            (
                "empty decoded output",
                RequestResult(
                    1,
                    run.requests[1].prompt,
                    "",
                    (10, 11),
                    1.0,
                    1.1,
                    1.3,
                ),
            ),
            (
                "timing",
                RequestResult(
                    1,
                    run.requests[1].prompt,
                    run.requests[1].output,
                    (10, 11),
                    1.4,
                    1.1,
                    1.3,
                ),
            ),
        )
        for message, replacement in replacements:
            with self.subTest(message=message):
                broken = RunResult(
                    1,
                    1.0,
                    1.3,
                    (run.requests[0], replacement),
                    run.microbatches,
                )
                with self.assertRaisesRegex(ValueError, message):
                    validate_run(broken, config)

    def test_non_finite_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "acceptance"):
            BenchmarkConfig(min_aggregate_tps=math.nan).validate()


class RecordingBackend:
    def __init__(self):
        self.batches = []
        self.active = False

    def generate(self, requests):
        if self.active:
            raise AssertionError("backend called concurrently")
        self.active = True
        try:
            self.batches.append([request.request_id for request in requests])
            return BackendBatchResult(
                outputs=tuple(
                    GeneratedOutput(request.prompt + " jumps", (10, 11))
                    for request in requests
                ),
                service_seconds=0.01,
                peak_cuda_memory_bytes=2048,
            )
        finally:
            self.active = False


class SchedulerTests(unittest.TestCase):
    def test_full_then_closed_partial_is_fifo(self):
        backend = RecordingBackend()
        config = BenchmarkConfig(
            request_count=6,
            max_batch_size=4,
            max_new_tokens=2,
            queue_timeout_ms=1000,
        )

        result = run_windowed_batch(
            1, ["The quick brown fox"] * 6, backend, config
        )

        self.assertEqual(backend.batches, [[0, 1, 2, 3], [4, 5]])
        self.assertEqual(
            [request.request_id for request in result.requests], list(range(6))
        )

    def test_timeout_flushes_a_slow_arrival(self):
        backend = RecordingBackend()
        config = BenchmarkConfig(
            request_count=3,
            max_batch_size=16,
            max_new_tokens=2,
            queue_timeout_ms=2,
            interarrival_ms=15,
        )

        run_windowed_batch(1, ["The quick brown fox"] * 3, backend, config)

        self.assertEqual(backend.batches, [[0], [1], [2]])

    def test_worker_exception_reaches_caller(self):
        class BrokenBackend:
            def generate(self, requests):
                raise RuntimeError("injected backend failure")

        with self.assertRaisesRegex(RuntimeError, "injected backend failure"):
            run_windowed_batch(
                1,
                ["The quick brown fox"],
                BrokenBackend(),
                BenchmarkConfig(request_count=1, max_batch_size=1, max_new_tokens=2),
            )


if __name__ == "__main__":
    unittest.main()
