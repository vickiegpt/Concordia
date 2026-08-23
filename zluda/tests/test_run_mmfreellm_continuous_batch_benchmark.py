#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from mmfreellm_continuous_batch import RequestSpec


MODULE_PATH = Path(__file__).with_name(
    "run_mmfreellm_continuous_batch_benchmark.py"
)


def load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "run_mmfreellm_continuous_batch_benchmark", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_summary(run_index, tps, token_ids):
    return {
        "run_index": run_index,
        "aggregate_tokens_per_second": tps,
        "requests": [
            {"request_id": 0, "generated_token_ids": list(token_ids)},
            {"request_id": 1, "generated_token_ids": list(token_ids)},
        ],
    }


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.to_devices = []

    @property
    def shape(self):
        return (len(self.values), len(self.values[0]))

    def to(self, device):
        self.to_devices.append(device)
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            return FakeVector(self.values[row][column])
        return FakeVector(self.values[key])


class FakeVector:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = []
        self.input_ids = FakeTensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        self.attention_mask = FakeTensor([[1, 1, 1, 1], [1, 1, 1, 1]])

    def __call__(self, prompts, *, return_tensors, padding):
        self.calls.append((list(prompts), return_tensors, padding))
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

    def batch_decode(self, rows, *, skip_special_tokens):
        self.decode_call = (rows, skip_special_tokens)
        return [
            "The quick brown fox jumps",
            "The quick brown fox jumps",
        ]


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return FakeTensor(
            [[1, 2, 3, 4, 10, 11], [1, 2, 3, 4, 10, 11]]
        )


class FakeInferenceMode:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        self.owner.inference_entries += 1

    def __exit__(self, exception_type, exception, traceback):
        return False


class FakeCuda:
    def __init__(self):
        self.synchronize_calls = 0
        self.reset_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1

    def reset_peak_memory_stats(self):
        self.reset_calls += 1

    def max_memory_allocated(self):
        return 4096


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()
        self.inference_entries = 0

    def inference_mode(self):
        return FakeInferenceMode(self)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.benchmark = load_benchmark()

    def test_each_run_must_meet_threshold(self):
        result = self.benchmark.qualify_summaries(
            [
                valid_summary(1, 199.9, [10, 11]),
                valid_summary(2, 260.0, [10, 11]),
            ],
            200.0,
        )

        self.assertFalse(result["qualification_passed"])
        self.assertIn(
            "run 1 aggregate TPS 199.900000 is below 200.000000",
            result["failure_reasons"],
        )

    def test_generated_ids_must_match_between_runs(self):
        result = self.benchmark.qualify_summaries(
            [
                valid_summary(1, 250.0, [10, 11]),
                valid_summary(2, 260.0, [10, 12]),
            ],
            200.0,
        )

        self.assertFalse(result["deterministic_generated_token_ids"])
        self.assertFalse(result["qualification_passed"])
        self.assertIn("generated token determinism mismatch", result["failure_reasons"])

    def test_result_file_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            self.benchmark.write_result_exclusive(path, {"validated": True})
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                self.benchmark.write_result_exclusive(path, {"validated": True})

    def test_cpu_and_batch_seventeen_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA is mandatory"):
            self.benchmark.validate_cli(
                self.benchmark.parse_args(["--online", "--device", "cpu"])
            )
        with self.assertRaisesRegex(ValueError, "1..16"):
            self.benchmark.validate_cli(
                self.benchmark.parse_args(
                    ["--online", "--max-batch-size", "17"]
                )
            )

    def test_json_marks_fpga_false(self):
        line = self.benchmark.format_json_record({"fpga_tps_reported": False})

        self.assertTrue(line.startswith("MMFREELM_CONTINUOUS_BATCH_JSON="))
        self.assertFalse(json.loads(line.split("=", 1)[1])["fpga_tps_reported"])

    def test_cuda_backend_batches_and_slices_generated_ids(self):
        fake_torch = FakeTorch()
        tokenizer = FakeTokenizer()
        model = FakeModel()
        times = iter((1.0, 1.25))
        backend = self.benchmark.CudaGenerationBackend(
            torch_module=fake_torch,
            tokenizer=tokenizer,
            model=model,
            model_path="model",
            dtype_name="half",
            parameter_count=123,
            bitlinear_backend="default",
            clock=lambda: next(times),
        )
        requests = (
            RequestSpec(0, "The quick brown fox", 2),
            RequestSpec(1, "The quick brown fox", 2),
        )

        result = backend.generate(requests)

        self.assertEqual(tokenizer.calls[0][1:], ("pt", True))
        self.assertEqual(tokenizer.input_ids.to_devices, ["cuda"])
        self.assertEqual(tokenizer.attention_mask.to_devices, ["cuda"])
        self.assertEqual(fake_torch.inference_entries, 1)
        self.assertEqual(fake_torch.cuda.synchronize_calls, 2)
        self.assertEqual(fake_torch.cuda.reset_calls, 1)
        self.assertEqual(model.calls[0]["min_new_tokens"], 2)
        self.assertEqual(model.calls[0]["max_new_tokens"], 2)
        self.assertFalse(model.calls[0]["do_sample"])
        self.assertEqual(
            [output.generated_token_ids for output in result.outputs],
            [(10, 11), (10, 11)],
        )
        self.assertEqual(result.service_seconds, 0.25)
        self.assertEqual(result.peak_cuda_memory_bytes, 4096)


if __name__ == "__main__":
    unittest.main()
