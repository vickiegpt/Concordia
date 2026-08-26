#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_v3_workload_completion.py")


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_v3_workload_completion", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_record():
    return {
        "event": "ternip_v3_iq1s_execution",
        "status": "completed",
        "kernel": "iq1s_tmatmul_batch4",
        "logical_batch": 4,
        "descriptor_count": 8,
        "unique_submission_count": 2,
        "lane_mask": 3,
        "per_lane_completion_counts": [4, 4, 0, 0],
        "total_accelerator_cycles": 10,
        "total_matrix_bytes_read": 20,
        "total_input_bytes_read": 30,
        "total_output_bytes_written": 40,
    }


def valid_attention_record():
    return {
        "event": "hetgpu_gpu_attention_execution",
        "status": "completed",
        "kernel": "flash_attn_fwd",
        "launch_count": 1,
        "device": "cuda",
        "cpu_fallback": False,
        "emulator_fallback": False,
    }


class CompletionValidatorTests(unittest.TestCase):
    def test_accepts_complete_schema_and_writes_only_validated_records(self):
        validator = load_validator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "enabled.log"
            output = root / "completion.jsonl"
            record = valid_record()
            log.write_text(
                "noise\n"
                + json.dumps(record)
                + "\n"
                + json.dumps(valid_attention_record())
                + "\n",
                encoding="utf-8",
            )

            count = validator.extract_validated_records([log], output)

            self.assertEqual(count, 2)
            self.assertEqual(
                [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                ],
                [record, valid_attention_record()],
            )

    def test_rejects_each_malformed_required_field(self):
        validator = load_validator()
        invalid_values = {
            "logical_batch": 0,
            "descriptor_count": 0,
            "unique_submission_count": 0,
            "lane_mask": 0,
            "per_lane_completion_counts": [8, 0, 0],
            "total_accelerator_cycles": -1,
            "total_matrix_bytes_read": "20",
            "total_input_bytes_read": True,
            "total_output_bytes_written": -1,
        }
        for field, value in invalid_values.items():
            with self.subTest(field=field):
                record = valid_record()
                record[field] = value
                with self.assertRaises(ValueError):
                    validator.validate_record(record)

    def test_rejects_wrong_event_status_and_inconsistent_counts(self):
        validator = load_validator()
        for patch in (
            {"event": "something_else"},
            {"status": "submitted"},
            {"kernel": ""},
            {"unique_submission_count": 9},
            {"per_lane_completion_counts": [3, 4, 0, 0]},
            {"lane_mask": 7},
        ):
            with self.subTest(patch=patch):
                record = valid_record()
                record.update(patch)
                with self.assertRaises(ValueError):
                    validator.validate_record(record)

    def test_matching_malformed_record_fails_closed(self):
        validator = load_validator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "enabled.log"
            output = root / "completion.jsonl"
            record = valid_record()
            del record["descriptor_count"]
            log.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                validator.extract_validated_records([log], output)
            self.assertFalse(output.exists())

    def test_requires_native_cuda_attention_evidence(self):
        validator = load_validator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "enabled.log"
            output = root / "completion.jsonl"
            log.write_text(json.dumps(valid_record()) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "attention"):
                validator.extract_validated_records([log], output)
            self.assertFalse(output.exists())

    def test_rejects_malformed_attention_and_fallback_evidence(self):
        validator = load_validator()
        patches = (
            {"status": "submitted"},
            {"kernel": "ffn_gate"},
            {"launch_count": 0},
            {"launch_count": 2},
            {"device": "cpu"},
            {"cpu_fallback": True},
            {"emulator_fallback": True},
        )
        for patch in patches:
            with self.subTest(patch=patch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                log = root / "enabled.log"
                output = root / "completion.jsonl"
                attention = valid_attention_record()
                attention.update(patch)
                log.write_text(
                    json.dumps(valid_record()) + "\n" + json.dumps(attention) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    validator.extract_validated_records([log], output)
                self.assertFalse(output.exists())

    def test_rejects_eio_nonzero_completion_and_lane_outside_four(self):
        validator = load_validator()
        bad_logs = []

        eio = [json.dumps(valid_record()), json.dumps(valid_attention_record()), "EIO -5"]
        bad_logs.append("\n".join(eio) + "\n")

        completion = valid_record()
        completion["completion_statuses"] = [0, -5]
        bad_logs.append(
            json.dumps(completion) + "\n" + json.dumps(valid_attention_record()) + "\n"
        )

        lane = valid_record()
        lane["lane_mask"] = 0x10
        lane["per_lane_completion_counts"] = [0, 0, 0, 0]
        bad_logs.append(
            json.dumps(lane) + "\n" + json.dumps(valid_attention_record()) + "\n"
        )

        for text in bad_logs:
            with self.subTest(log=text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                log = root / "enabled.log"
                output = root / "completion.jsonl"
                log.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validator.extract_validated_records([log], output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
