#!/usr/bin/env python3
"""Read-only safety gate; passing does not certify accelerator correctness."""
from pathlib import Path
import sys

UNSAFE_BOOTS = {"e20135df-d2f6-425f-a22e-86a60a19a00a"}
BOOT_HELPERS = {"hetgpu_pacc_boot", "hetgpu_pacc_boot_diag"}


def check(boot_id, modules):
    errors = []
    if not boot_id or boot_id in UNSAFE_BOOTS:
        errors.append("boot has an unresolved firmware-to-System-RAM write incident")
    loaded = {line.split()[0] for line in modules.splitlines() if line.strip()}
    for name in sorted(loaded & BOOT_HELPERS):
        errors.append(f"unqualified boot helper is loaded: {name}")
    return errors


def main():
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        modules = Path("/proc/modules").read_text()
    except OSError as exc:
        print(f"PACC preflight refused: cannot inspect host: {exc}", file=sys.stderr)
        return 1
    errors = check(boot_id, modules)
    for error in errors:
        print(f"PACC preflight refused: {error}", file=sys.stderr)
    if errors:
        print("No device writes made. Maintenance recovery requires explicit approval; "
              "do not unload/reset helpers to bypass this gate.", file=sys.stderr)
        return 1
    print(f"PACC safety preflight passed: boot_id={boot_id}; "
          "firmware identity, completion and numerical correctness remain unverified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
