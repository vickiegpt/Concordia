#!/usr/bin/env python3
"""Fail-closed validator for Qwen mixed-quant CUDA/AU250 IQ1_S proof bundles."""

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


MODEL_SIZE = 94_155_830_880
MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
LLAMA_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"
EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}
MEASUREMENT_COUNT = 5
CU_COUNT = 4
RAW_MIN = -4096
RAW_MAX = 4096
TIMING_KEYS = (
    "model_load_ms",
    "prompt_tokens_per_second",
    "ttft_ms",
    "generation_tokens_per_second",
    "end_to_end_ms",
)


class ProofInvalid(ValueError):
    """Raised when any required proof boundary is absent or inconsistent."""


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


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "nonnegative"
        _fail(f"{label} must be finite and {qualifier}")
    return result


def _keys(value, keys, label):
    missing = [key for key in keys if key not in value]
    if missing:
        _fail(f"{label} missing keys: {', '.join(missing)}")


def _expect(value, expected, label):
    if value != expected:
        _fail(f"{label} must equal {expected!r}")


def _read_bytes(path):
    try:
        return Path(path).read_bytes()
    except OSError as error:
        _fail(f"cannot read {path}: {error}")


def _read_json(path):
    raw = _read_bytes(path)
    try:
        return _mapping(json.loads(raw), str(path))
    except ProofInvalid:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot parse {path}: {error}")


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _validate_audit(path):
    raw = _read_bytes(path)
    try:
        audit = _mapping(json.loads(raw), "model-tensor-audit.json")
    except ProofInvalid:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot parse {path}: {error}")
    required = {
        "schema_version": 1,
        "status": "pass",
        "model_sha256": MODEL_SHA256,
        "architecture": "qwen35moe",
        "routed_expert_count": 180,
        "routed_expert_types": EXPERT_TYPES,
        "tq1_0_total": 0,
        "non_expert_iq1s": [],
    }
    for key, expected in required.items():
        _expect(audit.get(key), expected, f"model_audit.{key}")
    return audit, _sha256(raw)


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
            columns[key].append(
                _number(
                    measurement[key],
                    f"{mode}.measurements[{index}].{key}",
                    positive=key != "model_load_ms",
                )
            )
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


def _validate_routes(label, value, hybrid):
    routes = _mapping(value, f"{label}.routes")
    names = ("eligible", "handled", "fallback", "error")
    _keys(routes, names, f"{label}.routes")
    normalized = {
        name: _integer(routes[name], f"{label}.routes.{name}", 0) for name in names
    }
    if hybrid:
        if normalized["eligible"] <= 0:
            _fail(f"{label}.routes.eligible must be positive")
        if normalized["handled"] != normalized["eligible"]:
            _fail(f"{label} must handle every eligible IQ1_S route")
        if normalized["fallback"] or normalized["error"]:
            _fail(f"{label} fallback and error counts must be zero")
    elif any(normalized.values()):
        _fail(f"{label} routes must all be zero")
    return normalized


def _validate_xrt(label, value, hybrid):
    xrt = _mapping(value, f"{label}.xrt")
    required = (
        "submission_count",
        "completion_count",
        "per_cu_submissions",
        "per_cu_completions",
        "request_ids",
        "stall_codes",
        "raw_min",
        "raw_max",
        "reference_checked_components",
        "operation_count",
    )
    _keys(xrt, required, f"{label}.xrt")
    submissions = _integer(xrt["submission_count"], f"{label}.xrt.submission_count", 0)
    completions = _integer(xrt["completion_count"], f"{label}.xrt.completion_count", 0)
    if submissions != completions:
        _fail(f"{label}.xrt submission/completion counts differ")
    per_submit = [
        _integer(item, f"{label}.xrt.per_cu_submissions[{index}]", 0)
        for index, item in enumerate(
            _list(xrt["per_cu_submissions"], f"{label}.xrt.per_cu_submissions")
        )
    ]
    per_complete = [
        _integer(item, f"{label}.xrt.per_cu_completions[{index}]", 0)
        for index, item in enumerate(
            _list(xrt["per_cu_completions"], f"{label}.xrt.per_cu_completions")
        )
    ]
    if len(per_submit) != CU_COUNT or len(per_complete) != CU_COUNT:
        _fail(f"{label}.xrt per-CU arrays must contain four entries")
    if per_submit != per_complete or sum(per_submit) != submissions:
        _fail(f"{label}.xrt per-CU accounting is inconsistent")
    request_ids = [
        _integer(item, f"{label}.xrt.request_ids[{index}]", 1)
        for index, item in enumerate(_list(xrt["request_ids"], f"{label}.xrt.request_ids"))
    ]
    if len(request_ids) != completions or len(request_ids) != len(set(request_ids)):
        _fail(f"{label}.xrt request IDs are missing or duplicated")
    stalls = [
        _integer(item, f"{label}.xrt.stall_codes[{index}]")
        for index, item in enumerate(_list(xrt["stall_codes"], f"{label}.xrt.stall_codes"))
    ]
    if len(stalls) != completions or any(item == 0 for item in stalls):
        _fail(f"{label}.xrt STALL evidence is incomplete or zero")
    raw_min = _integer(xrt["raw_min"], f"{label}.xrt.raw_min")
    raw_max = _integer(xrt["raw_max"], f"{label}.xrt.raw_max")
    if not (RAW_MIN <= raw_min <= raw_max <= RAW_MAX):
        _fail(f"{label}.xrt raw output range is invalid")
    checked = _integer(
        xrt["reference_checked_components"],
        f"{label}.xrt.reference_checked_components",
        0,
    )
    operations = _integer(xrt["operation_count"], f"{label}.xrt.operation_count", 0)
    if hybrid:
        if submissions <= 0 or checked <= 0 or operations <= 0:
            _fail(f"{label}.xrt must contain physical checked work")
        if any(item <= 0 for item in per_complete):
            _fail(f"{label}.xrt must complete work on all four CUs")
    elif any((submissions, completions, checked, operations, *per_submit, *per_complete)) or request_ids or stalls or raw_min != 0 or raw_max != 0:
        _fail(f"{label}.xrt must contain only zero accounting")
    return {
        "submission_count": submissions,
        "completion_count": completions,
        "per_cu_submissions": per_submit,
        "per_cu_completions": per_complete,
        "request_ids": request_ids,
        "stall_codes": stalls,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "reference_checked_components": checked,
        "operation_count": operations,
    }


def _validate_request_contract(mode, record, prompt_ids):
    contract = _mapping(record["request_contract"], f"{mode}.request_contract")
    expected = {
        "prompt": prompt_ids,
        "n_predict": 32,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
    }
    _expect(contract, expected, f"{mode}.request_contract")


def _validate_semantic_gate(mode, record):
    gate = record["semantic_hardware_gate"]
    if mode == "cuda":
        _expect(gate, None, "cuda.semantic_hardware_gate")
        return None
    gate = _mapping(gate, "hybrid.semantic_hardware_gate")
    _keys(gate, ("routes", "xrt", "gpu_attention_routes"), "hybrid.semantic_hardware_gate")
    routes = _validate_routes("hybrid.semantic_hardware_gate", gate["routes"], True)
    xrt = _validate_xrt("hybrid.semantic_hardware_gate", gate["xrt"], True)
    attention = _integer(
        gate["gpu_attention_routes"],
        "hybrid.semantic_hardware_gate.gpu_attention_routes",
        1,
    )
    return {"routes": routes, "xrt": xrt, "gpu_attention_routes": attention}


def _validate_mode(mode, record, audit_sha256):
    record = _mapping(record, mode)
    required = (
        "schema_version",
        "evidence_kind",
        "mode",
        "model_size",
        "model_sha256",
        "model_audit_sha256",
        "llama_revision",
        "binary_sha256",
        "placement",
        "prompt_tokens",
        "prompt_text",
        "prompt_token_ids",
        "generated_token_ids",
        "semantic",
        "hardware_probe",
        "semantic_hardware_gate",
        "routes",
        "xrt",
        "gpu_attention_routes",
        "measurements",
        "process",
        "device_health",
        "request_contract",
        "warmup_count",
    )
    _keys(record, required, mode)
    _expect(record["schema_version"], 2, f"{mode}.schema_version")
    _expect(record["evidence_kind"], "iq1s", f"{mode}.evidence_kind")
    _expect(record["mode"], mode, f"{mode}.mode")
    _expect(record["model_size"], MODEL_SIZE, f"{mode}.model_size")
    _expect(record["model_sha256"], MODEL_SHA256, f"{mode}.model_sha256")
    _expect(record["model_audit_sha256"], audit_sha256, f"{mode}.model_audit_sha256")
    _expect(record["llama_revision"], LLAMA_REVISION, f"{mode}.llama_revision")
    binary_sha = record["binary_sha256"]
    if not isinstance(binary_sha, str) or not len(binary_sha) == 64:
        _fail(f"{mode}.binary_sha256 must contain 64 characters")

    placement = _mapping(record["placement"], f"{mode}.placement")
    _keys(placement, ("all_layers_on_gpu", "cpu_layers"), f"{mode}.placement")
    _expect(placement["all_layers_on_gpu"], True, f"{mode}.placement.all_layers_on_gpu")
    _expect(placement["cpu_layers"], 0, f"{mode}.placement.cpu_layers")
    _expect(record["prompt_tokens"], 256, f"{mode}.prompt_tokens")
    prompt_text = record["prompt_text"]
    if not isinstance(prompt_text, str) or not prompt_text:
        _fail(f"{mode}.prompt_text must be nonempty")
    prompt_ids = [
        _integer(item, f"{mode}.prompt_token_ids[{index}]", 0)
        for index, item in enumerate(
            _list(record["prompt_token_ids"], f"{mode}.prompt_token_ids")
        )
    ]
    if len(prompt_ids) != 256:
        _fail(f"{mode}.prompt_token_ids must contain exactly 256 IDs")
    generated_ids = [
        _integer(item, f"{mode}.generated_token_ids[{index}]", 0)
        for index, item in enumerate(
            _list(record["generated_token_ids"], f"{mode}.generated_token_ids")
        )
    ]
    if len(generated_ids) != 32:
        _fail(f"{mode}.generated_token_ids must contain exactly 32 IDs")
    semantic = _mapping(record["semantic"], f"{mode}.semantic")
    _keys(semantic, ("text", "token_ids"), f"{mode}.semantic")
    _expect(semantic["text"], "OK", f"{mode}.semantic.text")
    semantic_ids = [
        _integer(item, f"{mode}.semantic.token_ids[{index}]", 0)
        for index, item in enumerate(
            _list(semantic["token_ids"], f"{mode}.semantic.token_ids")
        )
    ]
    if len(semantic_ids) != 1:
        _fail(f"{mode}.semantic.token_ids must contain exactly one ID")
    hardware_probe = _mapping(record["hardware_probe"], f"{mode}.hardware_probe")
    _keys(hardware_probe, ("token_ids", "n_predict"), f"{mode}.hardware_probe")
    _expect(hardware_probe["n_predict"], 2, f"{mode}.hardware_probe.n_predict")
    probe_ids = [
        _integer(item, f"{mode}.hardware_probe.token_ids[{index}]", 0)
        for index, item in enumerate(
            _list(hardware_probe["token_ids"], f"{mode}.hardware_probe.token_ids")
        )
    ]
    if len(probe_ids) != 2:
        _fail(f"{mode}.hardware_probe.token_ids must contain exactly two IDs")
    _validate_request_contract(mode, record, prompt_ids)
    _expect(record["warmup_count"], 1, f"{mode}.warmup_count")

    process = _mapping(record["process"], f"{mode}.process")
    _keys(process, ("exit_code",), f"{mode}.process")
    _expect(process["exit_code"], 0, f"{mode}.process.exit_code")
    _validate_health(mode, record)

    hybrid = mode == "hybrid"
    routes = _validate_routes(mode, record["routes"], hybrid)
    xrt = _validate_xrt(mode, record["xrt"], hybrid)
    attention = _integer(record["gpu_attention_routes"], f"{mode}.gpu_attention_routes", 0)
    if hybrid and attention <= 0:
        _fail("hybrid.gpu_attention_routes must be positive")
    if not hybrid and attention != 0:
        _fail("cuda.gpu_attention_routes must be zero")
    semantic_gate = _validate_semantic_gate(mode, record)
    if hybrid:
        if routes["handled"] < semantic_gate["routes"]["handled"]:
            _fail("hybrid final routes do not include the semantic hardware gate")
        if xrt["completion_count"] < semantic_gate["xrt"]["completion_count"]:
            _fail("hybrid final XRT evidence does not include the semantic hardware gate")

    return {
        "binary_sha256": binary_sha,
        "prompt_text": prompt_text,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_ids,
        "semantic_token_ids": semantic_ids,
        "routes": routes,
        "xrt": xrt,
        "semantic_hardware_gate": semantic_gate,
        "gpu_attention_routes": attention,
        "metrics": _summarize_measurements(mode, record["measurements"]),
    }


def _validate_numerical(record):
    record = _mapping(record, "numerical")
    _keys(record, ("schema_version", "status", "cases"), "numerical")
    _expect(record["schema_version"], 1, "numerical.schema_version")
    _expect(record["status"], "pass", "numerical.status")
    cases = _mapping(record["cases"], "numerical.cases")
    _keys(cases, ("single_tile", "tiled"), "numerical.cases")
    normalized = {}
    for name in ("single_tile", "tiled"):
        case = _mapping(cases[name], f"numerical.cases.{name}")
        _keys(case, ("status", "max_absolute_error"), f"numerical.cases.{name}")
        _expect(case["status"], "pass", f"numerical.cases.{name}.status")
        normalized[name] = {
            "status": "pass",
            "max_absolute_error": _number(
                case["max_absolute_error"], f"numerical.cases.{name}.max_absolute_error"
            ),
        }
    return {"schema_version": 1, "status": "pass", "cases": normalized}


def validate_proof(proof_root):
    root = Path(proof_root)
    audit, audit_sha256 = _validate_audit(root / "model-tensor-audit.json")
    cuda = _validate_mode("cuda", _read_json(root / "cuda.json"), audit_sha256)
    hybrid = _validate_mode("hybrid", _read_json(root / "hybrid.json"), audit_sha256)
    numerical = _validate_numerical(_read_json(root / "numerical" / "summary.json"))
    if cuda["binary_sha256"] != hybrid["binary_sha256"]:
        _fail("CUDA and hybrid binaries differ")
    if cuda["prompt_text"] != hybrid["prompt_text"]:
        _fail("CUDA and hybrid prompt text differs")
    if cuda["prompt_token_ids"] != hybrid["prompt_token_ids"]:
        _fail("CUDA and hybrid prompt token IDs differ")
    if cuda["generated_token_ids"] != hybrid["generated_token_ids"]:
        _fail("CUDA and hybrid generated token IDs differ")
    if cuda["semantic_token_ids"] != hybrid["semantic_token_ids"]:
        _fail("CUDA and hybrid semantic token IDs differ")
    eligible = hybrid["routes"]["eligible"]
    handled = hybrid["routes"]["handled"]
    return {
        "schema_version": 2,
        "status": "pass",
        "model": {
            "size": MODEL_SIZE,
            "sha256": MODEL_SHA256,
            "architecture": "qwen35moe",
            "llama_revision": LLAMA_REVISION,
            "binary_sha256": cuda["binary_sha256"],
        },
        "model_audit": audit,
        "model_audit_sha256": audit_sha256,
        "modes": {
            "cuda": {
                "measurements": MEASUREMENT_COUNT,
                "metrics": cuda["metrics"],
                "routes": cuda["routes"],
                "xrt": cuda["xrt"],
                "gpu_attention_routes": cuda["gpu_attention_routes"],
            },
            "hybrid": {
                "measurements": MEASUREMENT_COUNT,
                "metrics": hybrid["metrics"],
                "routes": hybrid["routes"],
                "xrt": hybrid["xrt"],
                "semantic_hardware_gate": hybrid["semantic_hardware_gate"],
                "gpu_attention_routes": hybrid["gpu_attention_routes"],
            },
        },
        "numerical": numerical,
        "token_ids_match": True,
        "eligible_route_coverage": handled / eligible,
        "tensor_eligibility_coverage": audit["routed_expert_types"]["IQ1_S"]
        / audit["routed_expert_count"],
        "all_cus_active": all(
            value > 0 for value in hybrid["xrt"]["per_cu_completions"]
        ),
    }


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    try:
        if len(arguments) != 1:
            _fail("usage: validate_qwen35_iq1s_au250_proof.py PROOF_DIR")
        result = validate_proof(arguments[0])
    except (ProofInvalid, OSError, ValueError, TypeError) as error:
        print(f"QWEN_IQ1S_PROOF_INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
