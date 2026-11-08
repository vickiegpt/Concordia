#!/usr/bin/env python3
"""Four isolated real-model requests; publish throughput only after all gates pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path("/srv/lanxin/qwen35-a3b")

def aggregate(results, profiles):
    qualified = (len(results) == len(profiles) == 4 and
                 all(r.get("all_correct") is True and r.get("ngl") == 99 and
                     not r.get("trace_only") and len(r.get("generated_tokens", [])) == r.get("requested_tokens")
                     for r in results) and
                 all(p.get("capture_valid") is True and p.get("build_matches") is True for p in profiles) and
                 {p.get("device") for p in profiles} == {0, 1, 2, 3})
    report = {"all_correct": qualified, "aggregate_output_tps": None, "aggregate_decode_tps": None,
              "scope": "short four-request correctness smoke; not sustained or large-batch qualification"}
    if qualified:
        start = min(r["started_monotonic_us"] for r in results)
        finish = max(r["finished_monotonic_us"] for r in results)
        decode_start = min(r["decode_started_monotonic_us"] for r in results)
        if not start < decode_start < finish:
            raise ValueError("invalid shared monotonic timing window")
        count = sum(len(r["generated_tokens"]) for r in results)
        report.update(aggregate_output_tps=count * 1e6 / (finish - start),
                      aggregate_decode_tps=(count - 4) * 1e6 / (finish - decode_start),
                      generated_tokens=count, request_window_s=(finish - start) / 1e6,
                      decode_window_s=(finish - decode_start) / 1e6)
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix")
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--build", default="0x20036")
    p.add_argument("--timeout", type=int, default=3600)
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.prefix):
        p.error("invalid prefix")
    reference = json.loads(a.reference.read_text())
    count = reference["requested_tokens"]
    if (reference.get("ngl") != 0 or reference.get("trace_only") or not 2 <= count <= 16 or
            len(reference.get("generated_tokens", [])) != count):
        p.error("reference must be a completed, non-diagnostic CPU run")
    for device in range(4):
        identity = json.loads((a.qualification / f"pacc{device}.json").read_text())
        if (not identity.get("capture_valid") or not identity.get("build_matches") or
                identity.get("device") != device or
                set(identity.get("submission_builds", {}).values()) != {int(a.build, 0)}):
            p.error(f"PACC{device} has no matching numerical/firmware qualification")
    destination = ROOT / "logs" / a.prefix
    destination.mkdir(exist_ok=False)
    env = os.environ.copy()
    env["CONCORDIA_RELEASE"] = str(a.runtime.resolve())
    for key in list(env):
        if key.startswith("QWEN_TRACE_"):
            del env[key]
    hashes = {}
    for file in [a.runtime / "libnvcuda.so", a.runtime / "libhetgpu_cuda_shim.so", ROOT / "bin/qwen-ngl-four"]:
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    (destination / "artifacts.json").write_text(json.dumps(hashes, indent=2) + "\n")
    flag_names = ("HETGPU_SIFIVE_QK_LOCAL_BATCH", "HETGPU_SIFIVE_QK_EXPERT_BATCH",
                  "HETGPU_SIFIVE_QK_VECTOR_PACK", "HETGPU_SIFIVE_QK_PROFILE",
                  "HETGPU_SIFIVE_ELF_PROFILE", "HETGPU_SIFIVE_LOG_KERNEL_LAUNCHES",
                  "HETGPU_SIFIVE_KERNEL_POST_DOORBELL_IRQ_SLEEP_US")
    (destination / "optimization-flags.json").write_text(json.dumps(
        {name:env.get(name, "1000" if name.endswith("IRQ_SLEEP_US") else
                      "1" if name.endswith("LOG_KERNEL_LAUNCHES") else "0")
         for name in flag_names}, indent=2)+"\n")
    processes, logs = [], []
    started = time.monotonic()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for device in range(4):
            log = (destination / f"pacc{device}-driver.log").open("x")
            logs.append(log)
            command = ["bash", str(ROOT / "bin/run-pacc-isolated.sh"), str(device), "--",
                       "python3", str(ROOT / "bin/profile-jobd-phases-four.py"),
                       "--device", str(device), "--require-build", a.build, "--sample-ms", "1",
                       "--timeout", str(a.timeout), "--output", str(destination / f"pacc{device}-phases.json"),
                       "--", str(ROOT / "bin/qwen-ngl-four"),
                       str(ROOT / "models/Qwen3.5-35B-A3B-Q4_K_M.gguf"), "99", str(count),
                       reference["prompt"], str(destination / f"pacc{device}-result.json"), str(a.reference.resolve())]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            time.sleep(0.5)
        codes = [process.wait(timeout=15) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
    def load(suffix):
        paths = [destination / f"pacc{i}-{suffix}.json" for i in range(4)]
        return [json.loads(path.read_text()) for path in paths if path.exists()]
    results, profiles = load("result"), load("phases")
    report = aggregate(results, profiles)
    if any(codes):
        report.update(all_correct=False, aggregate_output_tps=None, aggregate_decode_tps=None)
    report.update(returncodes=codes, elapsed_including_load_s=time.monotonic()-started,
                  requests=results, profile_files=[f"pacc{i}-phases.json" for i in range(4)])
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "requests"}, indent=2), flush=True)
    return 0 if report["all_correct"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
