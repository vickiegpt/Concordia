#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 <vendor-lx500_pacc.bin> <hetgpu_pacc_jobd> <out.bin> [jobs.conf]" >&2
    exit 2
fi

src="$1"
jobd="$2"
out="$3"
jobs_conf="${4:-}"

python3 - "$src" "$jobd" "$out" "$jobs_conf" <<'PY'
from pathlib import Path
import os
import sys

src = Path(sys.argv[1])
jobd = Path(sys.argv[2])
out = Path(sys.argv[3])
jobs_conf = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None

image = bytearray(src.read_bytes())
jobd_bytes = jobd.read_bytes()
conf_bytes = b""
if jobs_conf:
    conf_bytes = jobs_conf.read_bytes()

MAGIC = b"070701"

def align4(x: int) -> int:
    return (x + 3) & ~3

def parse_newc(buf: bytes, base: int = 0, size: int | None = None):
    limit = len(buf) if size is None else base + size
    pos = buf.find(MAGIC, base, limit)
    if pos < 0:
        raise SystemExit("no newc cpio magic found")
    entries = {}
    while pos + 110 <= limit and buf[pos:pos + 6] == MAGIC:
        hdr = buf[pos:pos + 110]
        fields = [int(hdr[i:i + 8], 16) for i in range(6, 110, 8)]
        mode = fields[1]
        size = fields[6]
        namesize = fields[11]
        name_start = pos + 110
        name_end = name_start + namesize
        name = bytes(buf[name_start:name_end - 1]).decode("utf-8", "replace")
        data = align4(name_end)
        end = align4(data + size)
        entries[name] = {
            "hdr": pos,
            "data": data,
            "end": end,
            "size": size,
            "mode": mode,
        }
        pos = end
        if name == "TRAILER!!!":
            break
    return entries

def patch_payload(entries, label: str, name: str, payload: bytes, pad: int = 0, required: bool = True):
    if name not in entries:
        print(f"skip {label}:{name}: missing")
        return False
    entry = entries[name]
    size = entry["size"]
    if len(payload) > size:
        msg = f"{label}:{name}: payload {len(payload)} exceeds existing size {size}"
        if required:
            raise SystemExit(msg)
        print(f"skip {msg}")
        return False
    start = entry["data"]
    image[start:start + size] = payload + bytes([pad]) * (size - len(payload))
    print(f"patched {label}:{name}: {len(payload)} bytes into {size}-byte slot")
    return True

jobd_threads = os.environ.get("PACC_JOBD_KERNEL_THREADS", "4").strip() or "4"
jobd_trace = os.environ.get("PACC_JOBD_TRACE", "0").strip() or "0"
jobd_kmsg_present = "PACC_JOBD_KMSG" in os.environ
jobd_kmsg = os.environ.get("PACC_JOBD_KMSG", "0").strip() or "0"
jobd_progress_status = os.environ.get("PACC_JOBD_PROGRESS_STATUS", "0").strip() or "0"
jobd_beacon = os.environ.get("PACC_JOBD_BEACON", "0").strip() or "0"
jobd_mbox_poll_present = "PACC_JOBD_MBOX_POLL" in os.environ
jobd_mbox_poll = os.environ.get("PACC_JOBD_MBOX_POLL", "0").strip() or "0"
jobd_loop_trace = os.environ.get("PACC_JOBD_LOOP_TRACE", os.environ.get("HETGPU_PACC_JOBD_LOOP_TRACE", "")).strip()
jobd_poll_timeout_ms = os.environ.get("PACC_JOBD_POLL_TIMEOUT_MS", "").strip()
jobd_idle_sleep_us = os.environ.get("PACC_JOBD_IDLE_SLEEP_US", "").strip()
jobd_arg_wait_us = os.environ.get("PACC_JOBD_ARG_WAIT_US", "").strip()
jobd_force_elf = os.environ.get("PACC_JOBD_FORCE_ELF", "0").strip() or "0"
jobd_full_ddr_map_present = "PACC_JOBD_FULL_DDR_MAP" in os.environ
jobd_full_ddr_map = os.environ.get("PACC_JOBD_FULL_DDR_MAP", "0").strip() or "0"
jobd_full_ddr_map_bytes = os.environ.get("PACC_JOBD_FULL_DDR_MAP_BYTES", "").strip()
jobd_pacc_id = os.environ.get("PACC_JOBD_PACC_ID", "").strip()
jobd_claim_id = os.environ.get("PACC_JOBD_CLAIM_ID", "0").strip() or "0"
jobd_force_pread = os.environ.get("PACC_JOBD_FORCE_PREAD", "0").strip() or "0"
jobd_force_devmem = os.environ.get("PACC_JOBD_FORCE_DEVMEM", "0").strip() or "0"
jobd_status_control_window = os.environ.get("PACC_JOBD_STATUS_CONTROL_WINDOW", "").strip()
jobd_status_pwrite = os.environ.get("PACC_JOBD_STATUS_PWRITE", "0").strip()
jobd_msync = os.environ.get("PACC_JOBD_MSYNC", "1").strip()
jobd_cbo_inval = os.environ.get("PACC_JOBD_CBO_INVAL", "0").strip() or "0"
jobd_cbo_flush = os.environ.get("PACC_JOBD_CBO_FLUSH", "0").strip() or "0"
jobd_notify_irq = os.environ.get("PACC_JOBD_NOTIFY_IRQ", "").strip()
jobd_heartbeat = os.environ.get("PACC_JOBD_HEARTBEAT", "0").strip() or "0"
jobd_boot_marker = os.environ.get("PACC_JOBD_BOOT_MARKER", "0").strip() or "0"
jobd_early_devmem_marker = os.environ.get("PACC_JOBD_EARLY_DEVMEM_MARKER", os.environ.get("HETGPU_PACC_JOBD_EARLY_DEVMEM_MARKER", "")).strip()
jobd_seed_current_jobs = os.environ.get("PACC_JOBD_SEED_CURRENT_JOBS", "").strip()
jobd_shared_ddr_user_off = os.environ.get("PACC_JOBD_SHARED_DDR_USER_OFF", "").strip()
jobd_shared_ddr_fd_user_off = os.environ.get("PACC_JOBD_SHARED_DDR_FD_USER_OFF", "").strip()
jobd_rope_local_max = os.environ.get("PACC_JOBD_ROPE_LOCAL_MAX_BYTES", "").strip()
jobd_mmvf_local_x_max = os.environ.get("PACC_JOBD_MMVF_LOCAL_X_MAX_BYTES", "").strip()
jobd_mmvf_local_y_max = os.environ.get("PACC_JOBD_MMVF_LOCAL_Y_MAX_BYTES", "").strip()
jobd_kernel_slot_map = os.environ.get("PACC_JOBD_KERNEL_SLOT_MAP", os.environ.get("HETGPU_PACC_JOBD_KERNEL_SLOT_MAP", "")).strip()
jobd_kernel_slot_map_bytes = os.environ.get("PACC_JOBD_KERNEL_SLOT_MAP_BYTES", os.environ.get("HETGPU_PACC_JOBD_KERNEL_SLOT_MAP_BYTES", "")).strip()
jobd_kernel_slot_map_off = os.environ.get("PACC_JOBD_KERNEL_SLOT_MAP_OFF", os.environ.get("HETGPU_PACC_JOBD_KERNEL_SLOT_MAP_OFF", "")).strip()
jobd_xsfmm_smoke = os.environ.get("PACC_JOBD_XSFMM_SMOKE", os.environ.get("HETGPU_PACC_JOBD_XSFMM_SMOKE", "")).strip()
jobd_xsfmm_gemm = os.environ.get("PACC_JOBD_XSFMM_GEMM", os.environ.get("HETGPU_PACC_JOBD_XSFMM_GEMM", "")).strip()
jobd_xsfmm_max_n = os.environ.get("PACC_JOBD_XSFMM_MAX_N", os.environ.get("HETGPU_PACC_JOBD_XSFMM_MAX_N", "")).strip()
jobd_gemm_strict_visible = os.environ.get("PACC_JOBD_GEMM_STRICT_VISIBLE", os.environ.get("HETGPU_PACC_JOBD_GEMM_STRICT_VISIBLE", "")).strip()
jobd_status_mmap_fallback = os.environ.get("PACC_JOBD_STATUS_MMAP_FALLBACK", os.environ.get("HETGPU_PACC_JOBD_STATUS_MMAP_FALLBACK", "")).strip()
jobd_arg_slot_scan = os.environ.get("PACC_JOBD_ARG_SLOT_SCAN", os.environ.get("HETGPU_PACC_JOBD_ARG_SLOT_SCAN", "")).strip()
jobd_arg_slot_scan_all = os.environ.get("PACC_JOBD_ARG_SLOT_SCAN_ALL", os.environ.get("HETGPU_PACC_JOBD_ARG_SLOT_SCAN_ALL", "")).strip()

rcs_lines = [
    "#!/bin/sh",
]
if jobd_pacc_id:
    rcs_lines.append(f"export HETGPU_PACC_ID={jobd_pacc_id}")
if jobd_threads != "4":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_KERNEL_THREADS={jobd_threads}")
if jobd_trace != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_TRACE={jobd_trace}")
if jobd_kmsg_present:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_KMSG={jobd_kmsg}")
if jobd_progress_status != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_PROGRESS_STATUS={jobd_progress_status}")
if jobd_beacon != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_BEACON={jobd_beacon}")
if jobd_mbox_poll_present or jobd_mbox_poll != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_MBOX_POLL={jobd_mbox_poll}")
if jobd_loop_trace:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_LOOP_TRACE={jobd_loop_trace}")
if jobd_poll_timeout_ms:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_POLL_TIMEOUT_MS={jobd_poll_timeout_ms}")
if jobd_idle_sleep_us:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_IDLE_SLEEP_US={jobd_idle_sleep_us}")
if jobd_arg_wait_us:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_ARG_WAIT_US={jobd_arg_wait_us}")
if jobd_force_elf != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_FORCE_ELF={jobd_force_elf}")
if jobd_full_ddr_map_present or jobd_full_ddr_map != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_FULL_DDR_MAP={jobd_full_ddr_map}")
if jobd_full_ddr_map_bytes:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_FULL_DDR_MAP_BYTES={jobd_full_ddr_map_bytes}")
if jobd_claim_id != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_CLAIM_ID={jobd_claim_id}")
if jobd_force_pread != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_FORCE_PREAD={jobd_force_pread}")
if jobd_force_devmem != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_FORCE_DEVMEM={jobd_force_devmem}")
if jobd_status_control_window:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_STATUS_CONTROL_WINDOW={jobd_status_control_window}")
if jobd_status_pwrite:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_STATUS_PWRITE={jobd_status_pwrite}")
if jobd_msync:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_MSYNC={jobd_msync}")
if jobd_cbo_inval != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_CBO_INVAL={jobd_cbo_inval}")
if jobd_cbo_flush != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_CBO_FLUSH={jobd_cbo_flush}")
if jobd_notify_irq:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_NOTIFY_IRQ={jobd_notify_irq}")
if jobd_heartbeat != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_HEARTBEAT={jobd_heartbeat}")
if jobd_boot_marker != "0":
    rcs_lines.append(f"export HETGPU_PACC_JOBD_BOOT_MARKER={jobd_boot_marker}")
if jobd_early_devmem_marker:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_EARLY_DEVMEM_MARKER={jobd_early_devmem_marker}")
if jobd_seed_current_jobs:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_SEED_CURRENT_JOBS={jobd_seed_current_jobs}")
if jobd_shared_ddr_user_off:
    rcs_lines.append(f"export HETGPU_PACC_SHARED_DDR_USER_OFF={jobd_shared_ddr_user_off}")
if jobd_shared_ddr_fd_user_off:
    rcs_lines.append(f"export HETGPU_PACC_SHARED_DDR_FD_USER_OFF={jobd_shared_ddr_fd_user_off}")
if jobd_rope_local_max:
    rcs_lines.append(f"export PACC_JOBD_ROPE_LOCAL_MAX_BYTES={jobd_rope_local_max}")
if jobd_mmvf_local_x_max:
    rcs_lines.append(f"export PACC_JOBD_MMVF_LOCAL_X_MAX_BYTES={jobd_mmvf_local_x_max}")
if jobd_mmvf_local_y_max:
    rcs_lines.append(f"export PACC_JOBD_MMVF_LOCAL_Y_MAX_BYTES={jobd_mmvf_local_y_max}")
if jobd_kernel_slot_map:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_KERNEL_SLOT_MAP={jobd_kernel_slot_map}")
if jobd_kernel_slot_map_bytes:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_KERNEL_SLOT_MAP_BYTES={jobd_kernel_slot_map_bytes}")
if jobd_kernel_slot_map_off:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_KERNEL_SLOT_MAP_OFF={jobd_kernel_slot_map_off}")
if jobd_xsfmm_smoke:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_XSFMM_SMOKE={jobd_xsfmm_smoke}")
if jobd_xsfmm_gemm:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_XSFMM_GEMM={jobd_xsfmm_gemm}")
if jobd_xsfmm_max_n:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_XSFMM_MAX_N={jobd_xsfmm_max_n}")
if jobd_gemm_strict_visible:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_GEMM_STRICT_VISIBLE={jobd_gemm_strict_visible}")
if jobd_status_mmap_fallback:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_STATUS_MMAP_FALLBACK={jobd_status_mmap_fallback}")
if jobd_arg_slot_scan:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_ARG_SLOT_SCAN={jobd_arg_slot_scan}")
if jobd_arg_slot_scan_all:
    rcs_lines.append(f"export HETGPU_PACC_JOBD_ARG_SLOT_SCAN_ALL={jobd_arg_slot_scan_all}")
shared_ddr_base = os.environ.get("PACC_JOBD_SHARED_DDR_BASE", "").strip()
shared_ddr_size = os.environ.get("PACC_JOBD_SHARED_DDR_SIZE", "").strip()
shared_ddr_pacc_base = os.environ.get("PACC_JOBD_SHARED_DDR_PACC_BASE", os.environ.get("HETGPU_PACC_SHARED_DDR_PACC_BASE", "")).strip()
if shared_ddr_base:
    rcs_lines.append(f"export HETGPU_PACC_SHARED_DDR_BASE={shared_ddr_base}")
if shared_ddr_size:
    rcs_lines.append(f"export HETGPU_PACC_SHARED_DDR_BYTES={shared_ddr_size}")
if shared_ddr_pacc_base:
    rcs_lines.append(f"export HETGPU_PACC_SHARED_DDR_PACC_BASE={shared_ddr_pacc_base}")
rcs_lines.append("exec /home/root/pacc_skl_test --mbox=/dev/mbox </dev/console >/dev/console 2>&1")
rcs = ("\n".join(rcs_lines) + "\n").encode()

default_conf = b"""gemm 2 2 2 0x20000800 0x20000840 0x20002800 2 2 2 0 0 1
softmax 0x20000900 0x20002900 1 4 4
rmsnorm 0x20000a00 0x20000a40 0x20002a00 1 4 0.00001
"""

outer = parse_newc(image)
patched = 0
patched += patch_payload(outer, "outer", "home/root/pacc_skl_test", jobd_bytes, 0)
patched += patch_payload(outer, "outer", "etc/init.d/rcS", rcs, ord("\n"))
patched += patch_payload(outer, "outer", "etc/skel/.bashrc", conf_bytes or default_conf, ord("\n"))

inner_name = "core-image-minimal-qemuriscv64.cpio"
if inner_name in outer:
    inner = outer[inner_name]
    inner_entries = parse_newc(image, inner["data"], inner["size"])
    patched += patch_payload(inner_entries, "inner", "home/root/pacc_skl_test", jobd_bytes, 0)
    patched += patch_payload(inner_entries, "inner", "etc/init.d/rcS", rcs, ord("\n"))
    patched += patch_payload(inner_entries, "inner", "etc/skel/.bashrc", conf_bytes or default_conf, ord("\n"))

if patched == 0:
    raise SystemExit("no payloads patched")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(image)
print(f"wrote {out} ({len(image)} bytes)")
PY
