#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("parse_v3_workload_timing.py")


def load_parser():
    spec = importlib.util.spec_from_file_location("parse_v3_workload_timing", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WorkloadTimingParserTests(unittest.TestCase):
    def test_parses_and_validates_kimi_llama_generation_timing(self):
        parser = load_parser()
        text = (
            "llama_perf_context_print: prompt eval time = 10.00 ms / 2 tokens "
            "(5.00 ms per token, 200.00 tokens per second)\n"
            "llama_perf_context_print: eval time = 500.00 ms / 25 tokens "
            "(20.00 ms per token, 50.00 tokens per second)\n"
        )

        timing = parser.parse_workload_timing(text, "kimi")

        self.assertEqual(timing["workload_kind"], "kimi")
        self.assertEqual(timing["tokens"], 25)
        self.assertEqual(timing["eval_ms"], 500.0)
        self.assertEqual(timing["tokens_per_second"], 50.0)
        self.assertTrue(timing["validated"])

    def test_parses_validated_matmulfreellm_benchmark_summary(self):
        parser = load_parser()
        text = """
  Iteration 1: 10.00 tok/s, 20.00 GFLOPS/s, 2.0000s
  Iteration 2: 20.00 tok/s, 40.00 GFLOPS/s, 1.0000s
Tokens Per Second (TPS):
  Mean:        15.00 tok/s
Compute Performance (GFLOPS/s):
  Total Runs: 2
Tokens Generated:
  Total: 30
"""

        timing = parser.parse_workload_timing(text, "matmulfreellm")

        self.assertEqual(timing["workload_kind"], "matmulfreellm")
        self.assertEqual(timing["runs"], 2)
        self.assertEqual(timing["total_tokens"], 30)
        self.assertEqual(timing["mean_tokens_per_second"], 15.0)
        self.assertTrue(timing["validated"])

    def test_rejects_missing_or_internally_inconsistent_timing(self):
        parser = load_parser()
        with self.assertRaisesRegex(ValueError, "recognized timing"):
            parser.parse_workload_timing("Output: semantic text only", "auto")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            parser.parse_workload_timing(
                "llama_perf_context_print: eval time = 500.00 ms / 25 tokens "
                "(20.00 ms per token, 99.00 tokens per second)",
                "kimi",
            )

    def test_cli_writes_one_structured_record(self):
        parser = load_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            output = root / "timing.json"
            stdout.write_text("", encoding="utf-8")
            stderr.write_text(
                "llama_perf_context_print: eval time = 1000.00 ms / 50 tokens "
                "(20.00 ms per token, 50.00 tokens per second)\n",
                encoding="utf-8",
            )

            parser.run_cli(
                [
                    "--label",
                    "baseline",
                    "--kind",
                    "kimi",
                    "--stdout",
                    str(stdout),
                    "--stderr",
                    str(stderr),
                    "--output",
                    str(output),
                ]
            )

            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["label"], "baseline")
            self.assertEqual(record["tokens"], 50)
            self.assertTrue(record["validated"])


if __name__ == "__main__":
    unittest.main()
