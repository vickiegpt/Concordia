#!/usr/bin/env python3
"""Fail-closed validator for the Qwen 3.5 TQ1 CUDA/AU250 proof bundle."""

import json
import math
import statistics
import sys
from pathlib import Path


MODEL_SIZE = 94_155_830_880
MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
LLAMA_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"
MEASUREMENT_COUNT = 5
CU_COUNT = 4
RAW_MIN = -16_384
RAW_MAX = 16_384
TIMING_KEYS = (
    "model_load_ms",
    "prompt_tokens_per_second",
    "ttft_ms",
    "generation_tokens_per_second",
    "end_to_end_ms",
)


class ProofInvalid(ValueError):
    """Raised whenever proof evidence is absent, inconsistent, or unsafe."""


def _fail(message):
    raise ProofInvalid(message)


def _mapping(value, label):
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _list(value, label):
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    return value


def _integer(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _fail(f"{label} must be finite and nonnegative")
    return result


def _keys(value, keys, label):
    missing = [key for key in keys if key not in value]
    if missing:
        _fail(f"{label} missing keys: {', '.join(missing)}")


def _read_json(path):
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except ProofInvalid:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read {path}: {error}")


def _expect(value, expected, label):
    if value != expected:
        _fail(f"{label} must equal {expected!r}")


def _validate_health(mode, record):
    health = _mapping(record["device_health"], f"{mode}.device_health")
    _keys(health, ("before", "after"), f"{mode}.device_health")
    for phase in ("before", "after"):
        item = _mapping(health[phase], f"{mode}.device_health.{phase}")
        _keys(item, ("firewall", "fatal_errors"), f"{mode}.device_health.{phase}")
        _expect(item["firewall"], "GOOD", f"{mode}.device_health.{phase}.firewall")
        fatal = _list(item["fatal_errors"], f"{mode}.device_health.{phase}.fatal_errors")
        if fatal:
            _fail(f"{mode}.device_health.{phase}.fatal_errors must be empty")


def _summarize_measurements(mode, measurements):
    measurements = _list(measurements, f"{mode}.measurements")
    if len(measurements) != MEASUREMENT_COUNT:
        _fail(f"{mode}.measurements must contain exactly {MEASUREMENT_COUNT} entries")

    columns = {key: [] for key in TIMING_KEYS}
    for index, raw_measurement in enumerate(measurements):
        measurement = _mapping(raw_measurement, f"{mode}.measurements[{index}]")
        _keys(measurement, TIMING_KEYS, f"{mode}.measurements[{index}]")
        for key in TIMING_KEYS:
            columns[key].append(_number(measurement[key], f"{mode}.measurements[{index}].{key}"))

    summaries = {}
    for key, values in columns.items():
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values)
        summaries[key] = {
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "population_stdev": stdev,
            "cv": 0.0 if mean == 0.0 else stdev / mean,
        }
    return summaries


def _validate_xrt(mode, record, hybrid):
    xrt = _mapping(record["xrt"], f"{mode}.xrt")
    required = (
        "submission_count",
        "completion_count",
        "per_cu_submissions",
        "per_cu_completions",
        "request_ids",
        "stall_codes",
        "raw_min",
        "raw_max",
    )
    _keys(xrt, required, f"{mode}.xrt")
    submissions = _integer(xrt["submission_count"], f"{mode}.xrt.submission_count", 0)
    completions = _integer(xrt["completion_count"], f"{mode}.xrt.completion_count", 0)
    if submissions != completions:
        _fail(f"{mode}.xrt submission/completion counts differ")

    per_submit = _list(xrt["per_cu_submissions"], f"{mode}.xrt.per_cu_submissions")
    per_complete = _list(xrt["per_cu_completions"], f"{mode}.xrt.per_cu_completions")
    if len(per_submit) != CU_COUNT or len(per_complete) != CU_COUNT:
        _fail(f"{mode}.xrt per-CU arrays must contain {CU_COUNT} entries")
    per_submit = [_integer(value, f"{mode}.xrt.per_cu_submissions[{index}]", 0) for index, value in enumerate(per_submit)]
    per_complete = [_integer(value, f"{mode}.xrt.per_cu_completions[{index}]", 0) for index, value in enumerate(per_complete)]
    if per_submit != per_complete or sum(per_submit) != submissions:
        _fail(f"{mode}.xrt per-CU accounting is inconsistent")

    request_ids = _list(xrt["request_ids"], f"{mode}.xrt.request_ids")
    request_ids = [_integer(value, f"{mode}.xrt.request_ids[{index}]", 1) for index, value in enumerate(request_ids)]
    if len(request_ids) != completions:
        _fail(f"{mode}.xrt request ID count differs from completions")
    if len(set(request_ids)) != len(request_ids):
        _fail(f"{mode}.xrt request IDs must be unique")

    stalls = _list(xrt["stall_codes"], f"{mode}.xrt.stall_codes")
    if len(stalls) != completions:
        _fail(f"{mode}.xrt stall-code count differs from completions")
    for index, value in enumerate(stalls):
        _expect(_integer(value, f"{mode}.xrt.stall_codes[{index}]"), 1, f"{mode}.xrt.stall_codes[{index}]")

    raw_min = _integer(xrt["raw_min"], f"{mode}.xrt.raw_min")
    raw_max = _integer(xrt["raw_max"], f"{mode}.xrt.raw_max")
    if raw_min > raw_max or raw_min < RAW_MIN or raw_max > RAW_MAX:
        _fail(f"{mode}.xrt raw output range is invalid")

    if hybrid:
        if submissions <= 0 or any(value <= 0 for value in per_complete):
            _fail("hybrid.xrt must complete work on all four CUs")
    elif any((submissions, completions, *per_submit, *per_complete)) or request_ids or stalls or raw_min != 0 or raw_max != 0:
        _fail("cuda.xrt must contain only zero accounting")
    normalized_xrt = {
        "submission_count": submissions,
        "completion_count": completions,
        "per_cu_submissions": per_submit,
        "per_cu_completions": per_complete,
        "request_ids": request_ids,
        "stall_codes": stalls,
        "raw_min": raw_min,
        "raw_max": raw_max,
    }
    return (all(value > 0 for value in per_complete) if hybrid else False), normalized_xrt


def _validate_mode(mode, record):
    record = _mapping(record, mode)
    required = (
        "schema_version",
        "mode",
        "model_size",
        "model_sha256",
        "llama_revision",
        "binary_sha256",
        "placement",
        "prompt_tokens",
        "prompt_token_ids",
        "generated_token_ids",
        "semantic",
        "routes",
        "xrt",
        "measurements",
        "process",
        "device_health",
    )
    _keys(record, required, mode)
    _expect(record["schema_version"], 1, f"{mode}.schema_version")
    _expect(record["mode"], mode, f"{mode}.mode")
    _expect(record["model_size"], MODEL_SIZE, f"{mode}.model_size")
    _expect(record["model_sha256"], MODEL_SHA256, f"{mode}.model_sha256")
    _expect(record["llama_revision"], LLAMA_REVISION, f"{mode}.llama_revision")
    binary_sha = record["binary_sha256"]
    if not isinstance(binary_sha, str) or len(binary_sha) != 64:
        _fail(f"{mode}.binary_sha256 must be a 64-character string")

    placement = _mapping(record["placement"], f"{mode}.placement")
    _keys(placement, ("all_layers_on_gpu", "cpu_layers"), f"{mode}.placement")
    _expect(placement["all_layers_on_gpu"], True, f"{mode}.placement.all_layers_on_gpu")
    _expect(placement["cpu_layers"], 0, f"{mode}.placement.cpu_layers")

    _expect(record["prompt_tokens"], 256, f"{mode}.prompt_tokens")
    prompt_ids = _list(record["prompt_token_ids"], f"{mode}.prompt_token_ids")
    if len(prompt_ids) != 256:
        _fail(f"{mode}.prompt_token_ids must contain exactly 256 IDs")
    for index, value in enumerate(prompt_ids):
        _integer(value, f"{mode}.prompt_token_ids[{index}]", 0)
    generated_ids = _list(record["generated_token_ids"], f"{mode}.generated_token_ids")
    for index, value in enumerate(generated_ids):
        _integer(value, f"{mode}.generated_token_ids[{index}]", 0)

    semantic = _mapping(record["semantic"], f"{mode}.semantic")
    _keys(semantic, ("text", "token_ids"), f"{mode}.semantic")
    _expect(semantic["text"], "OK", f"{mode}.semantic.text")
    semantic_ids = _list(semantic["token_ids"], f"{mode}.semantic.token_ids")
    if not semantic_ids:
        _fail(f"{mode}.semantic.token_ids must not be empty")
    for index, value in enumerate(semantic_ids):
        _integer(value, f"{mode}.semantic.token_ids[{index}]", 0)

    process = _mapping(record["process"], f"{mode}.process")
    _keys(process, ("exit_code",), f"{mode}.process")
    _expect(process["exit_code"], 0, f"{mode}.process.exit_code")
    _validate_health(mode, record)

    routes = _mapping(record["routes"], f"{mode}.routes")
    _keys(routes, ("eligible", "handled", "fallback", "error"), f"{mode}.routes")
    route_values = {key: _integer(routes[key], f"{mode}.routes.{key}", 0) for key in ("eligible", "handled", "fallback", "error")}
    hybrid = mode == "hybrid"
    if hybrid:
        if route_values["eligible"] <= 0:
            _fail("hybrid.routes.eligible must be positive")
        if route_values["handled"] != route_values["eligible"]:
            _fail("hybrid must handle every eligible route")
        if route_values["fallback"] or route_values["error"]:
            _fail("hybrid fallback and error counts must be zero")
    elif any(route_values.values()):
        _fail("cuda routes must all be zero")

    all_cus_active, xrt = _validate_xrt(mode, record, hybrid)
    return {
        "binary_sha256": binary_sha,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_ids,
        "semantic_token_ids": semantic_ids,
        "routes": route_values,
        "all_cus_active": all_cus_active,
        "xrt": xrt,
        "metrics": _summarize_measurements(mode, record["measurements"]),
    }


def _validate_numerical(record):
    record = _mapping(record, "numerical")
    _keys(record, ("schema_version", "status", "cases"), "numerical")
    _expect(record["schema_version"], 1, "numerical.schema_version")
    _expect(record["status"], "pass", "numerical.status")
    cases = _mapping(record["cases"], "numerical.cases")
    _keys(cases, ("single_tile", "tiled"), "numerical.cases")
    for name in ("single_tile", "tiled"):
        case = _mapping(cases[name], f"numerical.cases.{name}")
        _keys(case, ("status", "max_absolute_error"), f"numerical.cases.{name}")
        _expect(case["status"], "pass", f"numerical.cases.{name}.status")
        _number(case["max_absolute_error"], f"numerical.cases.{name}.max_absolute_error")


def validate_proof(proof_root):
    root = Path(proof_root)
    cuda = _validate_mode("cuda", _read_json(root / "cuda.json"))
    hybrid = _validate_mode("hybrid", _read_json(root / "hybrid.json"))
    _validate_numerical(_read_json(root / "numerical" / "summary.json"))

    if cuda["binary_sha256"] != hybrid["binary_sha256"]:
        _fail("CUDA and hybrid binaries differ")
    if cuda["prompt_token_ids"] != hybrid["prompt_token_ids"]:
        _fail("CUDA and hybrid prompt token IDs differ")
    if cuda["generated_token_ids"] != hybrid["generated_token_ids"]:
        _fail("CUDA and hybrid generated token IDs differ")
    if cuda["semantic_token_ids"] != hybrid["semantic_token_ids"]:
        _fail("CUDA and hybrid semantic token IDs differ")

    eligible = hybrid["routes"]["eligible"]
    handled = hybrid["routes"]["handled"]
    return {
        "schema_version": 1,
        "status": "pass",
        "modes": {
            "cuda": {
                "measurements": MEASUREMENT_COUNT,
                "metrics": cuda["metrics"],
                "routes": cuda["routes"],
                "xrt": cuda["xrt"],
            },
            "hybrid": {
                "measurements": MEASUREMENT_COUNT,
                "metrics": hybrid["metrics"],
                "routes": hybrid["routes"],
                "xrt": hybrid["xrt"],
            },
        },
        "token_ids_match": True,
        "eligible_route_coverage": handled / eligible,
        "all_cus_active": hybrid["all_cus_active"],
    }


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    try:
        if len(arguments) != 1:
            _fail("usage: validate_qwen35_tq1_au250_proof.py PROOF_DIR")
        result = validate_proof(arguments[0])
    except (ProofInvalid, OSError, ValueError, TypeError) as error:
        print(f"QWEN_TQ1_PROOF_INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
