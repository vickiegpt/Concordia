#!/usr/bin/env python3
"""Validate an FU900/PACC XM hardware-image delivery and live evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "tools" / "xsfmm_v0p6p6_xm_requirements.json"
DEFAULT_FIT = pathlib.Path("/mnt/probe_nvme0n1p4/pacc/u-boot.itb")
DEFAULT_SOURCE = (
    ROOT / "ext" / "pacc_runtime-sys" / "pacc_linux_jobd" / "xsfmm_native_bf16.c"
)


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.passes.append(message)
        else:
            self.failures.append(message)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def get_path(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for component in dotted.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(dotted)
        current = current[component]
    return current


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_delivery(
    check: Check, requirements: dict[str, Any], delivery: dict[str, Any]
) -> pathlib.Path | None:
    for field in requirements["required_delivery_fields"]:
        try:
            value = get_path(delivery, field)
            check.require(value not in (None, ""), f"delivery field {field}")
        except KeyError:
            check.require(False, f"delivery field {field}")

    if check.failures:
        return None

    check.require(
        delivery["schema"] == "hetgpu.xsfmm-xm-delivery/v1",
        "delivery schema",
    )
    hardware = delivery["hardware"]
    target = requirements["target"]
    check.require(
        hardware["soc"] == target["soc"],
        f"hardware SoC is {target['soc']}",
    )
    check.require(
        hardware["xsfmm_version"] == requirements["specification"]["version"],
        f"Xsfmm version is {requirements['specification']['version']}",
    )
    check.require(
        delivery["activation"]["requires_host_reboot"] is False,
        "candidate activates without a main-host reboot",
    )
    for key in (
        "pacc_instances",
        "harts_per_pacc",
        "vlen_bits",
        "te_for_tew32",
    ):
        check.require(
            hardware[key] == target[key],
            f"hardware {key} is {target[key]}",
        )

    delivered_extensions = {str(item).lower() for item in hardware["extensions"]}
    for extension in requirements["required_extensions"]:
        check.require(
            extension.lower() in delivered_extensions,
            f"hardware extension {extension}",
        )

    image_path = pathlib.Path(delivery["image"]["path"]).expanduser()
    check.require(image_path.is_file(), f"image exists: {image_path}")
    if not image_path.is_file():
        return None
    check.require(
        image_path.stat().st_size == delivery["image"]["bytes"],
        "image byte size matches manifest",
    )
    check.require(
        sha256(image_path).lower() == delivery["image"]["sha256"].lower(),
        "image SHA-256 matches manifest",
    )
    return image_path


def validate_source(
    check: Check, requirements: dict[str, Any], source_path: pathlib.Path
) -> None:
    check.require(source_path.is_file(), f"runtime source exists: {source_path}")
    if not source_path.is_file():
        return
    source = source_path.read_text(encoding="utf-8", errors="replace").lower()
    for instruction, word in requirements["required_instruction_words"].items():
        check.require(
            word.lower() in source,
            f"runtime word {word} ({instruction})",
        )


def validate_fit(
    check: Check, requirements: dict[str, Any], fit_path: pathlib.Path
) -> None:
    check.require(fit_path.is_file(), f"PACC FIT exists: {fit_path}")
    if not fit_path.is_file():
        return
    try:
        strings = subprocess.run(
            ["strings", str(fit_path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.lower()
    except (OSError, subprocess.CalledProcessError) as error:
        check.require(False, f"read PACC FIT strings: {error}")
        return
    for extension in requirements["required_extensions"]:
        advertised = extension.lower() in strings
        if extension.lower() == "zve32f":
            # A full V implementation with F implies the Zve32f subset. FU900
            # DTBs use the full base-ISA spelling instead of listing Zve32f.
            advertised = advertised or bool(
                re.search(r"rv(?:32|64)[a-z]*f[a-z]*v", strings)
            )
        check.require(
            advertised,
            f"PACC FIT advertises {extension}",
        )


def validate_live_evidence(
    check: Check, requirements: dict[str, Any], evidence_dir: pathlib.Path
) -> None:
    check.require(evidence_dir.is_dir(), f"evidence directory exists: {evidence_dir}")
    if not evidence_dir.is_dir():
        return
    files = [path for path in evidence_dir.rglob("*") if path.is_file()]
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in files
    )
    lower = text.lower()
    check.require("completion timeout" not in lower, "no completion timeout")
    check.require("submit failed" not in lower, "no submit failure")
    check.require("host fallback" not in lower, "no host fallback")

    devices = requirements["acceptance"]["required_pacc_devices"]
    for device in devices:
        patterns = (
            rf"pacc{device}:\s+ok",
            rf"device[= ]{device}.*mismatches=0",
            rf"dev[= ]{device}.*mismatches=0",
        )
        check.require(
            any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns),
            f"PACC{device} completed with zero mismatches",
        )

    state_files = [path for path in files if path.name == "state.txt"]
    check.require(bool(state_files), "host boot-ID state evidence exists")
    for path in state_files:
        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
        before = values.get("before_boot_id")
        after = values.get("after_boot_id")
        check.require(
            bool(before) and before == after,
            f"host boot ID preserved in {path}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a candidate FU900/PACC XM hardware image. A DTB extension "
            "string alone never produces a PASS."
        )
    )
    parser.add_argument(
        "--requirements",
        type=pathlib.Path,
        default=DEFAULT_REQUIREMENTS,
    )
    parser.add_argument("--delivery", type=pathlib.Path)
    parser.add_argument("--fit", type=pathlib.Path, default=DEFAULT_FIT)
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--evidence-dir", type=pathlib.Path)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="check requirements, delivery, image, FIT, and runtime source only",
    )
    args = parser.parse_args()

    requirements = load_json(args.requirements)
    check = Check()
    check.require(
        requirements.get("schema") == "hetgpu.xsfmm-xm-requirements/v1",
        "requirements schema",
    )
    validate_source(check, requirements, args.source)
    validate_fit(check, requirements, args.fit)

    if args.delivery is None:
        check.require(False, "candidate delivery manifest supplied")
    else:
        check.require(args.delivery.is_file(), f"delivery exists: {args.delivery}")
        if args.delivery.is_file():
            delivery = load_json(args.delivery)
            validate_delivery(check, requirements, delivery)

    if not args.static_only:
        if args.evidence_dir is None:
            check.require(False, "live completion evidence supplied")
        else:
            validate_live_evidence(check, requirements, args.evidence_dir)

    for message in check.passes:
        print(f"PASS: {message}")
    for message in check.failures:
        print(f"FAIL: {message}", file=sys.stderr)

    if check.failures:
        print(
            f"RESULT: FAIL ({len(check.failures)} failed, "
            f"{len(check.passes)} passed)",
            file=sys.stderr,
        )
        return 1
    print(f"RESULT: PASS ({len(check.passes)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
