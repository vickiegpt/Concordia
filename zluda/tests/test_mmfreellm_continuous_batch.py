#!/usr/bin/env python3
import math
from pathlib import Path
import sys
from threading import Event, Thread
import unittest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from mmfreellm_continuous_batch import (
    BackendBatchResult,
    BenchmarkConfig,
    GeneratedOutput,
    MicrobatchResult,
    RequestResult,
    RequestSpec,
    RunResult,
    WindowedRequestQueue,
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
    def test_queue_timeout_uses_injected_clock(self):
        now = [10.0]
        request_queue = WindowedRequestQueue(16, 0.002, lambda: now[0])
        request_queue.submit(RequestSpec(0, "The quick brown fox", 2))
        now[0] = 10.003

        batch = request_queue.take_batch()

        self.assertEqual([request.spec.request_id for request in batch], [0])

    def test_queue_blocks_producer_until_capacity_is_released(self):
        now = [1.0]
        request_queue = WindowedRequestQueue(1, 10.0, lambda: now[0])
        first = RequestSpec(0, "The quick brown fox", 2)
        second = RequestSpec(1, "The quick brown fox", 2)
        request_queue.submit(first)
        submit_started = Event()
        submit_finished = Event()
        admitted = []

        def submit_second():
            submit_started.set()
            admitted.append(request_queue.submit(second))
            submit_finished.set()

        producer = Thread(target=submit_second)
        producer.start()
        self.assertTrue(submit_started.wait(1.0))
        self.assertFalse(submit_finished.wait(0.02))

        now[0] = 5.0
        first_batch = request_queue.take_batch()

        self.assertTrue(submit_finished.wait(1.0))
        second_batch = request_queue.take_batch()
        request_queue.close()
        producer.join()
        self.assertEqual([item.spec.request_id for item in first_batch], [0])
        self.assertEqual([item.spec.request_id for item in second_batch], [1])
        self.assertEqual(admitted[0].enqueued_at, 5.0)

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
