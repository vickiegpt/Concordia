#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_mmfreellm_2p7b_benchmark.py")


def load_benchmark():
    spec = importlib.util.spec_from_file_location("run_mmfreellm_2p7b_benchmark", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MatMulFreeLMBenchmarkTests(unittest.TestCase):
    def test_summary_is_derived_from_measured_runs(self):
        benchmark = load_benchmark()
        runs = [
            benchmark.MeasuredRun(1, 12, 2.0, 6.0, 32.4),
            benchmark.MeasuredRun(2, 8, 1.0, 8.0, 43.2),
        ]

        summary = benchmark.summarize_runs(runs, parameter_count=2_700_000_000)

        self.assertEqual(summary["schema"], "matmulfreellm-generation-benchmark-v1")
        self.assertEqual(summary["total_runs"], 2)
        self.assertEqual(summary["total_generated_tokens"], 20)
        self.assertEqual(summary["mean_tokens_per_second"], 7.0)
        self.assertEqual(summary["parameter_count"], 2_700_000_000)

    def test_summary_rejects_empty_or_nonpositive_measurements(self):
        benchmark = load_benchmark()
        with self.assertRaisesRegex(ValueError, "at least one"):
            benchmark.summarize_runs([], parameter_count=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            benchmark.summarize_runs(
                [benchmark.MeasuredRun(1, 0, 1.0, 0.0, 0.0)],
                parameter_count=1,
            )

    def test_text_output_matches_the_strict_workload_parser(self):
        benchmark = load_benchmark()
        runs = [
            benchmark.MeasuredRun(1, 10, 2.0, 5.0, 27.0),
            benchmark.MeasuredRun(2, 10, 1.0, 10.0, 54.0),
        ]
        summary = benchmark.summarize_runs(runs, parameter_count=2_700_000_000)

        text = benchmark.format_text_report(runs, summary)

        parser_path = MODULE_PATH.with_name("parse_v3_workload_timing.py")
        spec = importlib.util.spec_from_file_location("parse_v3_workload_timing", parser_path)
        parser = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(parser)
        parsed = parser.parse_workload_timing(text, "matmulfreellm")
        self.assertEqual(parsed["runs"], 2)
        self.assertEqual(parsed["total_tokens"], 20)
        self.assertEqual(parsed["mean_tokens_per_second"], 7.5)

    def test_json_record_is_canonical_and_contains_semantic_output(self):
        benchmark = load_benchmark()
        runs = [benchmark.MeasuredRun(1, 4, 0.5, 8.0, 43.2)]
        summary = benchmark.summarize_runs(runs, parameter_count=2_700_000_000)
        summary.update({"model": "2.7b", "output": "The quick brown fox jumps"})

        line = benchmark.format_json_record(summary)

        self.assertTrue(line.startswith("MMFREELM_BENCHMARK_JSON="))
        payload = json.loads(line.split("=", 1)[1])
        self.assertEqual(payload["output"], "The quick brown fox jumps")
        self.assertEqual(payload["mean_tokens_per_second"], 8.0)


if __name__ == "__main__":
    unittest.main()
