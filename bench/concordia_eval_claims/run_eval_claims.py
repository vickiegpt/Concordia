#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CSV_FIELDS = [
    "claim_id",
    "status",
    "source",
    "expected",
    "observed",
    "unit",
    "runner",
    "artifact",
    "message",
]


@dataclass
class CommandResult:
    name: str
    status: str
    observed: str
    artifact: str
    message: str


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    paper = Path(args.paper).resolve()
    claims_path = Path(args.claims).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    paper_text = paper.read_text(encoding="utf-8")
    runner = Runner(repo_root, work_dir, args.static_only)

    rows = []
    had_failure = False
    for claim in claims:
        results = [check_source(claim, paper, paper_text)]
        for command in claim.get("commands", []):
            results.append(runner.run(command))
        row = row_for_claim(claim, results)
        rows.append(row)
        had_failure = had_failure or row["status"] == "fail"

    write_csv(Path(args.csv), rows)
    write_jsonl(Path(args.jsonl), rows)
    write_markdown(Path(args.markdown), rows, paper, claims_path)
    print(f"[eval-claims] CSV: {args.csv}")
    print(f"[eval-claims] JSONL: {args.jsonl}")
    print(f"[eval-claims] Markdown: {args.markdown}")
    return 1 if had_failure else 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-run evidence for claims in Concordia 05_eval.tex"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run only compile/static/fixture checks and mark live hardware claims blocked.",
    )
    return parser.parse_args()


def check_source(claim, paper: Path, paper_text: str) -> CommandResult:
    pattern = claim.get("source_regex", "")
    if not pattern:
        return CommandResult(
            "source",
            "fail",
            "missing_source_regex",
            str(paper),
            "claim has no source_regex",
        )
    try:
        found = re.search(pattern, paper_text) is not None
    except re.error as err:
        return CommandResult(
            "source",
            "fail",
            "invalid_source_regex",
            str(paper),
            f"{err}",
        )
    if found:
        return CommandResult(
            "source",
            "pass",
            "claim text found",
            str(paper),
            f"{paper.name}:{claim.get('source', '')}",
        )
    return CommandResult(
        "source",
        "fail",
        "claim text not found",
        str(paper),
        f"missing pattern {pattern}",
    )


class Runner:
    def __init__(self, repo_root: Path, work_dir: Path, static_only: bool):
        self.repo_root = repo_root
        self.work_dir = work_dir
        self.static_only = static_only
        self.cache = {}

    def run(self, name: str) -> CommandResult:
        if name not in self.cache:
            self.cache[name] = self._run_uncached(name)
        return self.cache[name]

    def _run_uncached(self, name: str) -> CommandResult:
        if name == "claim_audit":
            return self._run_command(
                name,
                ["bash", "bench/concordia_claim_audit/test_claim_audit.sh"],
                self.work_dir / "claim_audit.log",
            )
        if name == "delta_static":
            env = {"CONCORDIA_BENCH_STATIC_ONLY": "1"}
            return self._run_command(
                name,
                ["bash", "bench/concordia_delta_checkpoint/test_smoke.sh"],
                self.work_dir / "delta_static.log",
                env=env,
            )
        if name == "delta_real":
            if self.static_only:
                return self._blocked(name, "static_only", "live cuda-oxide delta run skipped")
            if self._gpu_count() < 1:
                return self._blocked(name, "no_cuda_device", "no visible CUDA GPU")
            return self._run_delta_real()
        if name == "kimi_parser":
            return self._run_command(
                name,
                ["bash", "bench/kimi_k26_tps/test_parser.sh"],
                self.work_dir / "kimi_parser.log",
            )
        if name == "kimi_tps":
            if self.static_only:
                return self._blocked(name, "static_only", "real Kimi TPS run skipped")
            return self._run_kimi_tps()
        if name == "sass_dry_run":
            env = {
                "HETGPU_SASS_PROOF_WORKDIR": str(self.work_dir / "sass_dry_run"),
                "HETGPU_SASS_PROOF_KEEP": "1",
            }
            result = self._run_command(
                name,
                ["bash", "zluda/tests/sass_roundtrip_bench/run_correctness_suite.sh", "--dry-run"],
                self.work_dir / "sass_dry_run.log",
                env=env,
            )
            if result.status == "pass":
                result.artifact = str(self.work_dir / "sass_dry_run" / "sass_lifter_correctness.csv")
            return result
        if name == "needs_one_gpu":
            count = self._gpu_count()
            if count >= 1:
                return CommandResult(name, "pass", f"gpu_count={count}", "", "visible GPU")
            return self._blocked(name, "no_cuda_device", "requires at least one visible GPU")
        if name == "needs_two_gpus":
            count = self._gpu_count()
            if count >= 2:
                return CommandResult(name, "pass", f"gpu_count={count}", "", "visible GPUs")
            return self._blocked(name, f"gpu_count={count}", "requires at least two visible GPUs")
        if name == "needs_cross_arch_devices":
            return self._blocked(
                name,
                "missing_heterogeneous_targets",
                "requires NVIDIA+AMD/Intel/Tenstorrent live targets",
            )
        return CommandResult(name, "fail", "unknown_command", "", f"unknown command {name}")

    def _run_kimi_tps(self) -> CommandResult:
        kimi_work = self.work_dir / "kimi_tps"
        env = {
            "KIMI_TPS_WORKDIR": str(kimi_work),
            "KIMI_TPS_KEEP": "1",
            "KIMI_TPS_BUILD_ZLUDA": os.environ.get("KIMI_TPS_BUILD_ZLUDA", "0"),
        }
        result = self._run_command(
            "kimi_tps",
            ["bash", "bench/kimi_k26_tps/run_kimi_k26_tps.sh"],
            self.work_dir / "kimi_tps.log",
            env=env,
            timeout=600,
        )
        csv_path = kimi_work / "kimi_k26_tps.csv"
        result.artifact = str(csv_path)
        if result.status != "pass" or not csv_path.exists():
            return result
        statuses = kimi_statuses(csv_path)
        if statuses and all(status == "pass" for status in statuses):
            result.observed = ",".join(statuses)
            result.message = f"real Kimi TPS rows pass; csv:{csv_path}"
            return result
        result.status = "blocked"
        result.observed = ",".join(statuses) if statuses else "missing_rows"
        result.message = f"Kimi TPS did not produce pass rows; csv:{csv_path}"
        return result

    def _run_delta_real(self) -> CommandResult:
        binary = os.environ.get("CONCORDIA_DELTA_BENCH_BINARY", "")
        default_binary = (
            self.repo_root
            / "bench"
            / "concordia_delta_checkpoint"
            / "target"
            / "release"
            / "concordia_delta_checkpoint_bench"
        )
        if binary:
            command = [
                binary,
                "--warmup",
                os.environ.get("CONCORDIA_EVAL_DELTA_WARMUP", "1"),
                "--iters",
                os.environ.get("CONCORDIA_EVAL_DELTA_ITERS", "3"),
            ]
        elif default_binary.exists():
            command = [
                str(default_binary),
                "--warmup",
                os.environ.get("CONCORDIA_EVAL_DELTA_WARMUP", "1"),
                "--iters",
                os.environ.get("CONCORDIA_EVAL_DELTA_ITERS", "3"),
            ]
        else:
            command = ["bash", "bench/concordia_delta_checkpoint/test_smoke.sh"]
        return self._run_command(
            "delta_real",
            command,
            self.work_dir / "delta_real.log",
            timeout=300,
        )

    def _run_command(
        self,
        name: str,
        command: list[str],
        log_path: Path,
        env: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> CommandResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                env=merged_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as err:
            log_path.write_text(err.stdout or "", encoding="utf-8")
            return CommandResult(name, "fail", "timeout", str(log_path), f"timeout_{timeout}s")
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode == 0:
            return CommandResult(name, "pass", "exit_0", str(log_path), "ok")
        return CommandResult(
            name,
            "fail",
            f"exit_{completed.returncode}",
            str(log_path),
            f"exit_{completed.returncode}; log:{log_path}",
        )

    def _blocked(self, name: str, observed: str, message: str) -> CommandResult:
        return CommandResult(name, "blocked", observed, "", message)

    def _gpu_count(self) -> int:
        if "gpu_count" in self.cache:
            return int(self.cache["gpu_count"].observed)
        forced = os.environ.get("CONCORDIA_FORCE_GPU_COUNT")
        if forced:
            try:
                count = int(forced)
            except ValueError:
                count = 0
            self.cache["gpu_count"] = CommandResult(
                "gpu_count", "pass", str(max(0, count)), "", "forced"
            )
            return max(0, count)
        try:
            completed = subprocess.run(
                ["nvidia-smi", "-L"],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
            )
            count = sum(1 for line in completed.stdout.splitlines() if line.startswith("GPU "))
        except (OSError, subprocess.TimeoutExpired):
            count = 0
        self.cache["gpu_count"] = CommandResult("gpu_count", "pass", str(count), "", "probe")
        return count


def kimi_statuses(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [row.get("status", "") for row in csv.DictReader(handle)]


def row_for_claim(claim, results: list[CommandResult]):
    non_source = [result for result in results if result.name != "source"]
    status_values = [result.status for result in non_source] or [results[0].status]
    if any(status == "fail" for status in status_values) or results[0].status == "fail":
        status = "fail"
    elif any(status == "blocked" for status in status_values):
        status = "partial" if any(status == "pass" for status in status_values) else "blocked"
    elif all(status == "pass" for status in status_values):
        status = "pass"
    else:
        status = "partial"

    return {
        "claim_id": claim.get("id", ""),
        "status": status,
        "source": claim.get("source", ""),
        "expected": claim.get("expected", ""),
        "observed": "; ".join(f"{r.name}:{r.observed}" for r in results),
        "unit": claim.get("unit", ""),
        "runner": ";".join(result.name for result in results),
        "artifact": ";".join(result.artifact for result in results if result.artifact),
        "message": "; ".join(f"{r.name}:{r.message}" for r in results),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown(path: Path, rows, paper: Path, claims_path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Concordia 05_eval Claim Rerun",
        "",
        f"- paper: `{paper}`",
        f"- claims: `{claims_path}`",
        "",
        "| claim | status | expected | observed | artifact |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {claim_id} | {status} | {expected} | {observed} | {artifact} |".format(
                claim_id=escape_md(row["claim_id"]),
                status=escape_md(row["status"]),
                expected=escape_md(row["expected"]),
                observed=escape_md(row["observed"]),
                artifact=escape_md(row["artifact"] or "-"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
