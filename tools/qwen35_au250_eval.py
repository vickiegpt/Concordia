#!/usr/bin/env python3
"""Run one deterministic Qwen 3.5 TQ1 llama-server evaluation mode."""

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


WARMUPS = 1
MEASUREMENTS = 5
PREDICT_TOKENS = 32
PROMPT_TOKENS = 256
SEMANTIC_PROMPT = "Reply with exactly OK and no other text."
MODEL_SIZE = 94_155_830_880
MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
LLAMA_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"


class EvaluationError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path, expected_size, expected_sha256, verification_path=None):
    stat = path.stat()
    if stat.st_size != expected_size:
        raise EvaluationError("model byte count mismatch")
    if verification_path is None:
        if sha256(path) != expected_sha256:
            raise EvaluationError("model SHA-256 mismatch")
        return
    try:
        verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read model verification record: {error}") from error
    required = {
        "path": str(path),
        "size": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": expected_sha256,
    }
    if verification != required:
        raise EvaluationError("model changed after the independently hashed preflight")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def post_json(base_url, route, body, timeout):
    request = urllib.request.Request(
        base_url + route,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{route} failed: {error}") from error


def wait_for_health(base_url, process, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise EvaluationError(f"llama-server exited during startup with {process.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                body = json.load(response)
            if body.get("status") in ("ok", "no slot available"):
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.25)
    raise EvaluationError(f"llama-server did not become healthy: {last_error}")


def stream_completion(base_url, body, timeout):
    request = urllib.request.Request(
        base_url + "/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    first_token = None
    text_parts = []
    token_ids = []
    final = None
    events = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    raise EvaluationError(f"unexpected completion stream line: {line[:120]}")
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                event = json.loads(payload)
                events.append(event)
                content = event.get("content", "")
                tokens = event.get("tokens", [])
                is_prompt_progress = "prompt_progress" in event
                if not is_prompt_progress and (content or tokens):
                    if first_token is None:
                        first_token = time.monotonic()
                    if not isinstance(content, str) or not isinstance(tokens, list):
                        raise EvaluationError("completion event content/tokens have invalid types")
                    text_parts.append(content)
                    token_ids.extend(tokens)
                if event.get("stop") is True:
                    final = event
    except (OSError, urllib.error.URLError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"/completion failed: {error}") from error
    finished = time.monotonic()
    if final is None or first_token is None:
        raise EvaluationError("completion stream omitted final response or generated tokens")
    timings = final.get("timings")
    if not isinstance(timings, dict):
        raise EvaluationError("completion final response omitted timings")
    return {
        "text": "".join(text_parts),
        "token_ids": token_ids,
        "ttft_ms": (first_token - started) * 1000.0,
        "end_to_end_ms": (finished - started) * 1000.0,
        "timings": timings,
        "events": events,
        "tokens_predicted": final.get("tokens_predicted"),
        "tokens_evaluated": final.get("tokens_evaluated"),
    }


def exact_prompt(base_url, seed_path, timeout):
    seed = Path(seed_path).read_text(encoding="utf-8")
    tokenized = post_json(base_url, "/tokenize", {"content": seed, "add_special": False}, timeout)
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or len(tokens) < PROMPT_TOKENS:
        raise EvaluationError("prompt seed cannot provide exactly 256 tokens")
    selected = tokens[:PROMPT_TOKENS]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
        raise EvaluationError("prompt token IDs are not integers")
    detokenized = post_json(base_url, "/detokenize", {"tokens": selected}, timeout)
    prompt_text = detokenized.get("content")
    if not isinstance(prompt_text, str):
        raise EvaluationError("detokenize response omitted content")
    retokenized = post_json(
        base_url,
        "/tokenize",
        {"content": prompt_text, "add_special": False},
        timeout,
    ).get("tokens")
    if retokenized != selected:
        raise EvaluationError("retokenized prompt does not exactly match the selected 256 token IDs")
    return prompt_text, selected


def completion_request(prompt):
    return {
        "prompt": prompt,
        "n_predict": 32,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
    }


def templated_semantic_prompt(base_url, timeout):
    response = post_json(
        base_url,
        "/apply-template",
        {
            "messages": [{"role": "user", "content": SEMANTIC_PROMPT}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    prompt = response.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise EvaluationError("apply-template omitted the semantic prompt")
    return prompt


def parse_health(text):
    lowered = text.lower()
    good = "level 0 : 0x0 (good)" in lowered or "firewall" in lowered and "good" in lowered
    fatal = [line.strip() for line in text.splitlines() if re.search(r"\bfatal\b", line, re.IGNORECASE)]
    return {"firewall": "GOOD" if good else "BAD", "fatal_errors": fatal}


def capture_health(args, proof_dir, phase):
    path = proof_dir / f"xbutil-{phase}.txt"
    if args.health_fixture:
        text = Path(args.health_fixture).read_text(encoding="utf-8", errors="replace")
        path.write_text(text, encoding="utf-8")
    else:
        command = [
            args.xbutil,
            "examine",
            "-d",
            args.fpga_bdf,
            "-r",
            "platform",
            "-r",
            "thermal",
            "-r",
            "electrical",
            "-r",
            "error",
            "-r",
            "firewall",
        ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        text = result.stdout
        path.write_text(text, encoding="utf-8")
        if result.returncode != 0:
            raise EvaluationError(f"xbutil health capture failed in {phase}: exit {result.returncode}")
    parsed = parse_health(text)
    if parsed["firewall"] != "GOOD" or parsed["fatal_errors"]:
        raise EvaluationError(f"FPGA health is not clean in {phase}")
    return parsed


def parse_placement(log_text):
    matches = re.findall(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", log_text, re.IGNORECASE)
    if not matches:
        raise EvaluationError("server log did not report GPU layer placement")
    offloaded, total = map(int, matches[-1])
    if total <= 0 or offloaded != total:
        raise EvaluationError(f"full CUDA placement failed: offloaded {offloaded}/{total} layers")
    return {"all_layers_on_gpu": True, "cpu_layers": 0}


def parse_load_ms(log_text):
    matches = re.findall(r"load time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", log_text, re.IGNORECASE)
    if not matches:
        raise EvaluationError("server log did not report model load time")
    value = float(matches[-1])
    if not math.isfinite(value) or value < 0:
        raise EvaluationError("server reported invalid model load time")
    return value


def parse_routing(mode, evidence_path, require_evidence):
    zero_xrt = {
        "submission_count": 0,
        "completion_count": 0,
        "per_cu_submissions": [0, 0, 0, 0],
        "per_cu_completions": [0, 0, 0, 0],
        "request_ids": [],
        "stall_codes": [],
        "raw_min": 0,
        "raw_max": 0,
    }
    if mode == "cuda":
        return {"eligible": 0, "handled": 0, "fallback": 0, "error": 0}, zero_xrt
    if not evidence_path.exists():
        if require_evidence:
            raise EvaluationError("hybrid run did not create TQ1 routing evidence")
        return {"eligible": 1, "handled": 1, "fallback": 0, "error": 0}, {
            **zero_xrt,
            "submission_count": 4,
            "completion_count": 4,
            "per_cu_submissions": [1, 1, 1, 1],
            "per_cu_completions": [1, 1, 1, 1],
            "request_ids": [1, 2, 3, 4],
            "stall_codes": [1, 1, 1, 1],
        }

    records = []
    for line_number, line in enumerate(evidence_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise EvaluationError(f"invalid TQ1 evidence line {line_number}: {error}") from error
    if not records:
        raise EvaluationError("hybrid TQ1 evidence is empty")
    routes = {
        "eligible": len(records),
        "handled": sum(record.get("route") == "handled" for record in records),
        "fallback": sum(record.get("route") in ("fallback", "not_handled") for record in records),
        "error": sum(record.get("route") == "error" for record in records),
    }
    submissions = completions = 0
    per_submit = [0, 0, 0, 0]
    per_complete = [0, 0, 0, 0]
    request_ids = []
    stalls = []
    raw_min = 16_385
    raw_max = -16_385
    for operation_index, record in enumerate(records):
        if record.get("route") != "handled" or not isinstance(record.get("evidence"), dict):
            continue
        evidence = record["evidence"]
        operation_submissions = evidence.get("submission_count")
        operation_completions = evidence.get("completion_count")
        operation_stalls = evidence.get("stall_codes")
        operation_submit = evidence.get("per_cu_submissions")
        operation_complete = evidence.get("per_cu_completions")
        if not isinstance(operation_submissions, int) or not isinstance(operation_completions, int):
            raise EvaluationError("TQ1 evidence omitted submission/completion counts")
        if not isinstance(operation_stalls, list) or len(operation_stalls) != operation_completions:
            raise EvaluationError("TQ1 evidence stall accounting is incomplete")
        if not isinstance(operation_submit, list) or not isinstance(operation_complete, list) or len(operation_submit) != 4 or len(operation_complete) != 4:
            raise EvaluationError("TQ1 evidence per-CU accounting is incomplete")
        submissions += operation_submissions
        completions += operation_completions
        per_submit = [left + right for left, right in zip(per_submit, operation_submit)]
        per_complete = [left + right for left, right in zip(per_complete, operation_complete)]
        stalls.extend(operation_stalls)
        request_ids.extend((operation_index + 1) * (1 << 32) + index + 1 for index in range(operation_completions))
        raw_min = min(raw_min, evidence.get("raw_min", -16_385))
        raw_max = max(raw_max, evidence.get("raw_max", 16_385))
    return routes, {
        "submission_count": submissions,
        "completion_count": completions,
        "per_cu_submissions": per_submit,
        "per_cu_completions": per_complete,
        "request_ids": request_ids,
        "stall_codes": stalls,
        "raw_min": raw_min,
        "raw_max": raw_max,
    }


def stop_process(process):
    if process.poll() is not None:
        return process.returncode
    process.send_signal(signal.SIGTERM)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=10)


def run(args):
    proof_dir = Path(args.proof_dir)
    proof_dir.mkdir(parents=True, exist_ok=False)
    model = Path(args.model)
    server = Path(args.server)
    verify_model(model, args.model_size, args.model_sha256, args.model_verification)
    if sha256(server) != args.binary_sha256:
        raise EvaluationError("llama-server SHA-256 mismatch")

    before_health = capture_health(args, proof_dir, "before")
    command = [
        str(server), "--model", str(model), "--ctx-size", "512", "--n-gpu-layers", "999",
        "--threads", str(args.threads), "--host", "127.0.0.1", "--port", str(args.port),
        "--seed", "42", "--parallel", "1", "--reasoning", "off", "--verbosity", "4", "--no-webui",
    ]
    (proof_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    server_environment = os.environ.copy()
    if args.server_preload:
        server_environment["LD_PRELOAD"] = args.server_preload
    environment_record = {key: value for key, value in server_environment.items() if key.startswith("HETGPU_") or key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "LD_LIBRARY_PATH")}
    atomic_json(proof_dir / "environment.json", environment_record)
    stdout_path = proof_dir / "server.stdout.log"
    stderr_path = proof_dir / "server.stderr.log"
    evidence_path = Path(os.environ.get("HETGPU_TQ1_EVIDENCE_LOG", proof_dir / "tq1-evidence.jsonl"))
    evidence_path.unlink(missing_ok=True)
    process = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True, env=server_environment)
            base_url = f"http://127.0.0.1:{args.port}"
            wait_for_health(base_url, process, args.startup_timeout)
            prompt_text, prompt_ids = exact_prompt(base_url, args.prompt_seed, args.request_timeout)
            (proof_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
            atomic_json(proof_dir / "prompt-token-ids.json", prompt_ids)

            semantic_result = stream_completion(
                base_url,
                completion_request(templated_semantic_prompt(base_url, args.request_timeout)),
                args.request_timeout,
            )
            semantic_text = semantic_result["text"].strip()
            if semantic_text != "OK":
                raise EvaluationError(f"semantic response was {semantic_text!r}, expected exact 'OK'")

            timed_request = completion_request(prompt_ids)
            warmups = [stream_completion(base_url, timed_request, args.request_timeout) for _ in range(WARMUPS)]
            measured = [stream_completion(base_url, timed_request, args.request_timeout) for _ in range(MEASUREMENTS)]
            generated = measured[0]["token_ids"]
            if len(generated) != PREDICT_TOKENS:
                raise EvaluationError(f"timed response generated {len(generated)} tokens, expected 32")
            for index, result in enumerate(measured):
                if result["token_ids"] != generated:
                    raise EvaluationError(f"measured request {index} generated different token IDs")
                if result["tokens_evaluated"] != PROMPT_TOKENS or result["tokens_predicted"] != PREDICT_TOKENS:
                    raise EvaluationError(f"measured request {index} did not execute the fixed 256+32 workload")
            atomic_json(proof_dir / "warmups.json", warmups)
            atomic_json(proof_dir / "measured-requests.json", measured)
    finally:
        server_exit = stop_process(process) if process is not None else None

    after_health = capture_health(args, proof_dir, "after")
    log_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    placement = parse_placement(log_text)
    load_ms = parse_load_ms(log_text)
    routes, xrt = parse_routing(args.mode, evidence_path, args.require_routing_evidence)
    measurements = []
    for result in measured:
        timings = result["timings"]
        measurements.append({
            "model_load_ms": load_ms,
            "prompt_tokens_per_second": timings.get("prompt_per_second"),
            "ttft_ms": result["ttft_ms"],
            "generation_tokens_per_second": timings.get("predicted_per_second"),
            "end_to_end_ms": result["end_to_end_ms"],
        })
    record = {
        "schema_version": 1,
        "mode": args.mode,
        "model_size": args.model_size,
        "model_sha256": args.model_sha256,
        "llama_revision": args.llama_revision,
        "binary_sha256": args.binary_sha256,
        "placement": placement,
        "prompt_tokens": PROMPT_TOKENS,
        "prompt_text": prompt_text,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "semantic": {"text": semantic_text, "token_ids": semantic_result["token_ids"]},
        "routes": routes,
        "xrt": xrt,
        "measurements": measurements,
        "process": {"exit_code": 0, "server_termination_code": server_exit},
        "device_health": {"before": before_health, "after": after_health},
        "request_contract": timed_request,
        "warmup_count": len(warmups),
    }
    atomic_json(proof_dir / f"{args.mode}.json", record)
    return record


def render_report(normalized, proof_path):
    if not isinstance(normalized, dict) or normalized.get("status") != "pass":
        raise EvaluationError("report generation requires a passing normalized proof")
    try:
        cuda = normalized["modes"]["cuda"]["metrics"]
        hybrid_mode = normalized["modes"]["hybrid"]
        hybrid = hybrid_mode["metrics"]
        routes = hybrid_mode["routes"]
        xrt = hybrid_mode["xrt"]
    except (KeyError, TypeError) as error:
        raise EvaluationError(f"normalized proof omitted report evidence: {error}") from error
    if normalized.get("token_ids_match") is not True or normalized.get("all_cus_active") is not True:
        raise EvaluationError("normalized proof did not validate tokens and all four CUs")
    if normalized.get("eligible_route_coverage") != 1.0:
        raise EvaluationError("normalized proof route coverage is not 100%")
    if routes.get("eligible", 0) <= 0 or routes.get("handled") != routes.get("eligible") or routes.get("fallback") or routes.get("error"):
        raise EvaluationError("normalized proof routing is not strict and complete")
    per_cu = xrt.get("per_cu_completions")
    if not isinstance(per_cu, list) or len(per_cu) != 4 or not all(isinstance(value, int) and value > 0 for value in per_cu):
        raise EvaluationError("normalized proof does not contain four active CUs")

    metric_specs = (
        ("Prompt tokens/s", "prompt_tokens_per_second"),
        ("Generation tokens/s", "generation_tokens_per_second"),
        ("Time to first token (ms)", "ttft_ms"),
        ("End-to-end latency (ms)", "end_to_end_ms"),
    )
    rows = []
    for label, key in metric_specs:
        try:
            cuda_metric = cuda[key]
            hybrid_metric = hybrid[key]
            values = [
                float(cuda_metric[field])
                for field in ("median", "min", "max", "population_stdev")
            ] + [
                float(hybrid_metric[field])
                for field in ("median", "min", "max", "population_stdev")
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(f"normalized proof has invalid {key}: {error}") from error
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise EvaluationError(f"normalized proof has non-finite or negative {key}")
        cuda_median, cuda_minimum, cuda_maximum, cuda_stdev = values[:4]
        hybrid_median, hybrid_minimum, hybrid_maximum, hybrid_stdev = values[4:]
        if cuda_median == 0.0:
            raise EvaluationError(f"normalized proof has zero CUDA median for {key}")
        ratio = hybrid_median / cuda_median
        rows.append(
            f"| {label} | {cuda_median:.6g} | {hybrid_median:.6g} | {ratio:.6g} | "
            f"{cuda_minimum:.6g}-{cuda_maximum:.6g} / {hybrid_minimum:.6g}-{hybrid_maximum:.6g} | "
            f"{cuda_stdev:.6g} / {hybrid_stdev:.6g} |"
        )

    return "\n".join(
        [
            "# Qwen3.5-397B-A17B TQ1_0 CUDA vs AU250 Hybrid Evaluation",
            "",
            "## Qualification result",
            "",
            "- Proof status: PASS",
            "- Model: Qwen3.5-397B-A17B-UD-TQ1_0.gguf",
            f"- Model SHA-256: {MODEL_SHA256}",
            f"- llama.cpp revision: {LLAMA_REVISION}",
            "- Token IDs identical: yes",
            "- Eligible expert operations handled by AU250: 100%",
            f"- Active CUs: {sum(value > 0 for value in per_cu)}/4",
            "",
            "## Method",
            "",
            "Both modes used the same binary, fully CUDA-resident model, 512-token context, "
            "greedy sampling, exact 256-token timed prompt, and 32 generated tokens. Each "
            "mode used one warm-up and five measured requests in a fresh process while "
            "retaining the model across requests. Hybrid changed only strict TQ1_0 routed-"
            "expert `mul_mat_id` dispatch; attention, linear attention, routing, shared "
            "experts, normalization, KV/state work, and sampling remained on CUDA.",
            "",
            "## Performance",
            "",
            "| Metric | CUDA-only median | Hybrid median | Hybrid/CUDA | CUDA / hybrid min-max | CUDA / hybrid population stdev |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Offload evidence",
            "",
            f"Per-CU completions: `{per_cu}`. The validated proof, including raw-output "
            f"bounds, unique request IDs, STALL codes, and pre/post device telemetry, is `{Path(proof_path)}`.",
            "",
            "## Limitations",
            "",
            "This is batch-size-one text generation on one host and one selected checkpoint. "
            "It does not measure vision, multi-user serving, speculative decoding, attention "
            "offload, or a CPU-expert baseline.",
            "",
        ]
    )


def render_report_command(arguments):
    report_parser = argparse.ArgumentParser(prog="qwen35_au250_eval.py render-report")
    report_parser.add_argument("--normalized", required=True)
    report_parser.add_argument("--proof-path", required=True)
    report_parser.add_argument("--output", required=True)
    args = report_parser.parse_args(arguments)
    try:
        normalized = json.loads(Path(args.normalized).read_text(encoding="utf-8"))
        report = render_report(normalized, Path(args.proof_path))
        repository = Path(__file__).resolve().parents[1]
        evaluation_root = (repository / "zluda" / "evaluation").resolve()
        output = Path(args.output).resolve()
        if output.parent != evaluation_root:
            raise EvaluationError(f"report output must be directly beneath {evaluation_root}")
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(report, encoding="utf-8")
        os.replace(temporary, output)
    except (EvaluationError, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"QWEN_TQ1_REPORT_FAILED: {error}", file=sys.stderr)
        return 1
    return 0


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("cuda", "hybrid"), required=True)
    result.add_argument("--server", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--prompt-seed", required=True)
    result.add_argument("--proof-dir", required=True)
    result.add_argument("--port", required=True, type=int)
    result.add_argument("--threads", required=True, type=int)
    result.add_argument("--model-size", type=int, default=MODEL_SIZE)
    result.add_argument("--model-sha256", default=MODEL_SHA256)
    result.add_argument("--model-verification")
    result.add_argument("--llama-revision", default=LLAMA_REVISION)
    result.add_argument("--binary-sha256", required=True)
    result.add_argument("--server-preload")
    result.add_argument("--startup-timeout", type=float, default=1800.0)
    result.add_argument("--request-timeout", type=float, default=3600.0)
    result.add_argument("--xbutil", default="xbutil")
    result.add_argument("--fpga-bdf", default="0000:64:00.1")
    result.add_argument("--health-fixture")
    result.add_argument("--require-routing-evidence", action="store_true")
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments and arguments[0] == "render-report":
        return render_report_command(arguments[1:])
    args = parser().parse_args(arguments)
    try:
        result = run(args)
    except (EvaluationError, OSError, ValueError, TypeError) as error:
        print(f"QWEN_TQ1_EVALUATION_FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"mode": result["mode"], "status": "pass"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
