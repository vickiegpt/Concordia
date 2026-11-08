#!/usr/bin/env python3
"""Archive small reproducible evidence, never huge raw profiler rings/models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--run")
    a = p.parse_args()
    out, logs = a.destination, a.root / "logs"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    def copy(source, name=None):
        target = out / (name or source.name)
        shutil.copy2(source, target)
        manifest[str(source)] = dict(sha256=digest(source), archived=target.name)
    comparisons = []
    for variant in ("baseline", "local", "vector", "final"):
        stem = f"elf-profile-layer3-{variant}"
        copy(logs / f"{stem}.log")
        copy(logs / f"{stem}-summary.json")
        for layer in range(4):
            source = logs / f"{stem}-tensors" / f"l_out-{layer}.0.f32"
            baseline = logs / "elf-profile-layer3-baseline-tensors" / source.name
            row = dict(variant=variant, layer=layer, source=str(source),
                       bytes=source.stat().st_size, sha256=digest(source),
                       baseline_sha256=digest(baseline))
            row["byte_equal"] = source.read_bytes() == baseline.read_bytes()
            comparisons.append(row)
    if not all(r["byte_equal"] for r in comparisons):
        raise ValueError("diagnostic tensor differs from PACC baseline")
    (out / "layer-output-equality.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    for name in ("xm-local-vector-q4-v5.log", "xm-local-vector-q5-v5.log",
                 "xm-local-vector-q8-v5.log", "xm-local-vector-distinct-y-v3.log",
                 "xm-local-vector-distinct-y-v5.log"):
        copy(logs / name)
    runtime = a.root / "runtime-xm-local-batch-v5" / "libnvcuda.so"
    manifest[str(runtime)] = dict(sha256=digest(runtime), archived=None)
    if a.run:
        run = logs / a.run
        report = json.loads((run / "summary.json").read_text())
        # Archive failures as failures; never convert them into throughput.
        for name in ("summary.json", "artifacts.json", "optimization-flags.json"):
            copy(run / name, "four-"+name)
        if report["all_correct"]:
            tools = Path(__file__).resolve().parent
            for device in range(4):
                copy(run / f"pacc{device}-result.json")
                qualification = logs / (a.run + "-qualification")
                copy(qualification / f"pacc{device}-driver.log", f"qualification-pacc{device}.log")
                subprocess.run([sys.executable, str(tools / "compact-phase-evidence.py"),
                                str(run / f"pacc{device}-phases.json"),
                                str(out / f"pacc{device}-phases-compact.json")], check=True)
                subprocess.run([sys.executable, str(tools / "compact-phase-evidence.py"),
                                str(qualification / f"pacc{device}.json"),
                                str(out / f"qualification-pacc{device}-compact.json")], check=True)
            subprocess.run([sys.executable, str(tools / "audit-four-pacc-staging.py"),
                            str(run), "--output", str(out / "partition-audit.json")], check=True)
        baseline = logs / "four-ngl99-v54-v1" / "summary.json"
        copy(baseline, "four-baseline-summary.json")
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(output=str(out), files=len(manifest),
                          diagnostic_tensors_equal=True)))


if __name__ == "__main__":
    main()
