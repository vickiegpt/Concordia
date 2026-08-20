#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


EVENT = "ternip_v3_iq1s_execution"
STATUS = "completed"
EXPECTED_LANES = 4
COUNTER_FIELDS = (
    "total_accelerator_cycles",
    "total_matrix_bytes_read",
    "total_input_bytes_read",
    "total_output_bytes_written",
)


def require_integer(record: dict, field: str, *, positive: bool = False) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def validate_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("completion record must be a JSON object")
    if record.get("event") != EVENT:
        raise ValueError(f"event must be {EVENT}")
    if record.get("status") != STATUS:
        raise ValueError(f"status must be {STATUS}")
    kernel = record.get("kernel")
    if not isinstance(kernel, str) or not kernel.strip():
        raise ValueError("kernel must be a nonempty string")

    require_integer(record, "logical_batch", positive=True)
    descriptor_count = require_integer(record, "descriptor_count", positive=True)
    submission_count = require_integer(
        record, "unique_submission_count", positive=True
    )
    if submission_count > descriptor_count:
        raise ValueError("unique_submission_count exceeds descriptor_count")

    lane_mask = require_integer(record, "lane_mask", positive=True)
    per_lane = record.get("per_lane_completion_counts")
    if not isinstance(per_lane, list) or len(per_lane) != EXPECTED_LANES:
        raise ValueError(
            f"per_lane_completion_counts must contain exactly {EXPECTED_LANES} lanes"
        )
    normalized_counts = []
    for index, value in enumerate(per_lane):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"per_lane_completion_counts[{index}] must be a nonnegative integer"
            )
        normalized_counts.append(value)
    if sum(normalized_counts) != descriptor_count:
        raise ValueError(
            "sum(per_lane_completion_counts) must equal descriptor_count"
        )
    expected_lane_mask = sum(
        1 << index for index, value in enumerate(normalized_counts) if value > 0
    )
    if lane_mask != expected_lane_mask:
        raise ValueError(
            "lane_mask must exactly identify lanes with positive completion counts"
        )

    for field in COUNTER_FIELDS:
        require_integer(record, field)
    return record


def extract_validated_records(log_paths, output_path: Path) -> int:
    records = []
    for log_path in log_paths:
        source = Path(log_path)
        for line_number, raw in enumerate(
            source.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("event") != EVENT:
                continue
            try:
                validate_record(record)
            except ValueError as error:
                raise ValueError(
                    f"malformed {EVENT} record in {source}:{line_number}: {error}"
                ) from error
            records.append(record)
    if not records:
        raise ValueError("live workload produced no validated TernIP v3 IQ1_S record")

    output_path = Path(output_path)
    with output_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    return parser.parse_args(arguments)


def run_cli(arguments=None):
    args = parse_args(arguments)
    try:
        count = extract_validated_records(args.logs, args.output)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"validated_completion_records": count}, sort_keys=True))


if __name__ == "__main__":
    run_cli()
