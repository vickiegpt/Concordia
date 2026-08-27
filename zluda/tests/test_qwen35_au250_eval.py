#!/usr/bin/env python3
import json
import hashlib
import math
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import importlib.util


EVALUATOR = Path(__file__).parents[2] / "tools" / "qwen35_au250_eval.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("qwen35_eval", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FAKE_SERVER = r'''#!/usr/bin/env python3
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()
mode = os.environ["FAKE_MODE"]
requests_path = os.environ["FAKE_REQUESTS"]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def body(self):
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size))

    def send_json(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        body = self.body()
        if self.path == "/tokenize":
            content = body["content"]
            if content.startswith("seed "):
                tokens = list(range(300))
            elif content == "roundtrip-256":
                tokens = list(range(256))
            else:
                tokens = [701, 702]
            self.send_json({"tokens": tokens})
            return
        if self.path == "/detokenize":
            assert body["tokens"] == list(range(256))
            self.send_json({"content": "roundtrip-256"})
            return
        if self.path == "/apply-template":
            assert body["messages"] == [{"role": "user", "content": "Reply with exactly OK and no other text."}]
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            self.send_json({"prompt": "templated semantic prompt"})
            return
        if self.path != "/completion":
            self.send_error(404)
            return
        with open(requests_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"mode": mode, "body": body}, sort_keys=True) + "\n")
        semantic = isinstance(body["prompt"], str)
        tokens = [777] if semantic else list(range(1000, 1032))
        pieces = ["OK"] if semantic else [f"t{index}" for index in range(32)]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for processed in (0, 128, 256):
            progress = {
                "content": "",
                "tokens": [0],
                "stop": False,
                "tokens_predicted": 0,
                "tokens_evaluated": 2 if semantic else 256,
                "prompt_progress": {"total": 256, "cache": 0, "processed": processed, "time_ms": processed},
            }
            self.wfile.write(("data: " + json.dumps(progress) + "\n\n").encode())
        for index, (token, piece) in enumerate(zip(tokens, pieces)):
            payload = {
                "content": piece,
                "tokens": [token],
                "stop": False,
                "tokens_predicted": index + 1,
                "tokens_evaluated": 2 if semantic else 256,
            }
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
        final = {
            "content": "",
            "tokens": [],
            "stop": True,
            "tokens_predicted": len(tokens),
            "tokens_evaluated": 2 if semantic else 256,
            "timings": {
                "prompt_ms": 1280.0,
                "prompt_per_second": 200.0,
                "predicted_ms": 3200.0,
                "predicted_per_second": 10.0,
            },
        }
        self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())

print("llama_model_load_tensors: offloaded 65/65 layers to GPU", file=sys.stderr, flush=True)
print("llama_perf_context_print:        load time =    1234.00 ms", file=sys.stderr, flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_mode(tmp_path, mode, server, requests):
    proof = tmp_path / mode
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    env = os.environ.copy()
    env.update({"FAKE_MODE": mode, "FAKE_REQUESTS": str(requests)})
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--mode", mode,
            "--server", str(server),
            "--model", str(model),
            "--prompt-seed", str(tmp_path / "seed.txt"),
            "--proof-dir", str(proof),
            "--port", str(free_port()),
            "--threads", "4",
            "--model-size", "5",
            "--model-sha256", hashlib.sha256(model.read_bytes()).hexdigest(),
            "--llama-revision", "925e1179947ea0c0ebfb0032df18af3a729822be",
            "--binary-sha256", hashlib.sha256(server.read_bytes()).hexdigest(),
            "--health-fixture", str(tmp_path / "health.txt"),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((proof / f"{mode}.json").read_text())


def test_fake_server_preserves_identical_requests_and_fixed_counts(tmp_path):
    server = tmp_path / "fake-server.py"
    make_executable(server, FAKE_SERVER)
    (tmp_path / "seed.txt").write_text("seed " * 300, encoding="utf-8")
    (tmp_path / "health.txt").write_text("Level 0 : 0x0 (GOOD)\n", encoding="utf-8")
    requests = tmp_path / "requests.jsonl"

    cuda = run_mode(tmp_path, "cuda", server, requests)
    hybrid = run_mode(tmp_path, "hybrid", server, requests)

    assert len(cuda["measurements"]) == 5
    assert len(hybrid["measurements"]) == 5
    assert cuda["prompt_token_ids"] == hybrid["prompt_token_ids"] == list(range(256))
    assert cuda["generated_token_ids"] == hybrid["generated_token_ids"] == list(range(1000, 1032))
    assert cuda["semantic"] == hybrid["semantic"] == {"text": "OK", "token_ids": [777]}
    assert cuda["placement"] == {"all_layers_on_gpu": True, "cpu_layers": 0}

    records = [json.loads(line) for line in requests.read_text().splitlines()]
    assert len(records) == 14  # semantic + warm-up + five measured, for each mode
    cuda_bodies = [item["body"] for item in records if item["mode"] == "cuda"]
    hybrid_bodies = [item["body"] for item in records if item["mode"] == "hybrid"]
    assert cuda_bodies == hybrid_bodies
    timed = cuda_bodies[1:]
    assert len(timed) == 6
    assert all(body["prompt"] == list(range(256)) for body in timed)
    assert all(body["n_predict"] == 32 for body in timed)
    assert all(body["temperature"] == 0.0 and body["seed"] == 42 for body in timed)
    assert all(body["cache_prompt"] is False for body in timed)


def test_rejects_non_roundtripping_prompt(tmp_path):
    evaluator_source = EVALUATOR.read_text(encoding="utf-8")
    assert "retoken" in evaluator_source.lower()
    assert "exactly 256" in evaluator_source.lower()


def test_parse_load_ms_uses_verbose_server_timestamps():
    evaluator = load_evaluator()
    log = "\n".join(
        [
            "0.00.120.675 I srv    load_model: loading model '/models/qwen.gguf'",
            "1.50.407.110 I srv  llama_server: model loaded",
        ]
    )

    assert evaluator.parse_load_ms(log) == pytest.approx(110_286.435)


def metric(median, minimum, maximum, stdev):
    return {
        "median": median,
        "min": minimum,
        "max": maximum,
        "population_stdev": stdev,
        "cv": stdev / median,
    }


def test_render_report_uses_only_validated_metrics_and_cu_counts(tmp_path):
    evaluator = load_evaluator()
    cuda_metrics = {
        "prompt_tokens_per_second": metric(20.0, 19.0, 21.0, 0.5),
        "generation_tokens_per_second": metric(5.0, 4.5, 5.5, 0.2),
        "ttft_ms": metric(100.0, 90.0, 110.0, 4.0),
        "end_to_end_ms": metric(7000.0, 6900.0, 7100.0, 50.0),
        "model_load_ms": metric(1000.0, 1000.0, 1000.0, 0.0),
    }
    hybrid_metrics = {
        "prompt_tokens_per_second": metric(10.0, 9.0, 11.0, 0.4),
        "generation_tokens_per_second": metric(4.0, 3.5, 4.5, 0.1),
        "ttft_ms": metric(200.0, 190.0, 210.0, 5.0),
        "end_to_end_ms": metric(8000.0, 7900.0, 8100.0, 60.0),
        "model_load_ms": metric(1000.0, 1000.0, 1000.0, 0.0),
    }
    normalized = {
        "schema_version": 1,
        "status": "pass",
        "token_ids_match": True,
        "eligible_route_coverage": 1.0,
        "all_cus_active": True,
        "modes": {
            "cuda": {"measurements": 5, "metrics": cuda_metrics},
            "hybrid": {
                "measurements": 5,
                "metrics": hybrid_metrics,
                "routes": {"eligible": 8, "handled": 8, "fallback": 0, "error": 0},
                "xrt": {"per_cu_completions": [4, 3, 2, 1]},
            },
        },
    }
    report = evaluator.render_report(normalized, tmp_path / "proof")
    assert "Active CUs: 4/4" in report
    assert "| Prompt tokens/s | 20 | 10 | 0.5 |" in report
    assert "| Time to first token (ms) | 100 | 200 | 2 |" in report
    assert "Eligible expert operations handled by AU250: 100%" in report
    table_rows = [line for line in report.splitlines() if line.startswith("| ")][2:]
    assert len(table_rows) == 4
    for row in table_rows:
        numeric_cells = [cell.strip() for cell in row.split("|")[2:5]]
        assert all(math.isfinite(float(cell)) for cell in numeric_cells)


def test_render_report_refuses_nonpassing_proof():
    evaluator = load_evaluator()
    with pytest.raises(evaluator.EvaluationError):
        evaluator.render_report({"status": "fail"}, Path("proof"))
