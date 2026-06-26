#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 <vendor-lx500_sifive.bin> <hetgpu_sifive_jobd> <out.bin> [jobs.conf]" >&2
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

def parse_u64_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return int(value, 0)

def env_true(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value not in ("0", "false", "no", "off")

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

jobd_threads_present = "SIFIVE_JOBD_KERNEL_THREADS" in os.environ
jobd_threads = os.environ.get("SIFIVE_JOBD_KERNEL_THREADS", "4").strip() or "4"
jobd_trace = os.environ.get("SIFIVE_JOBD_TRACE", "0").strip() or "0"
jobd_log_present = "SIFIVE_JOBD_LOG" in os.environ
jobd_log = os.environ.get("SIFIVE_JOBD_LOG", "0").strip() or "0"
jobd_kmsg_present = "SIFIVE_JOBD_KMSG" in os.environ
jobd_kmsg = os.environ.get("SIFIVE_JOBD_KMSG", "0").strip() or "0"
jobd_progress_status = os.environ.get("SIFIVE_JOBD_PROGRESS_STATUS", "0").strip() or "0"
jobd_progress_completion = os.environ.get("SIFIVE_JOBD_PROGRESS_COMPLETION", "0").strip() or "0"
jobd_loop_trace = os.environ.get("SIFIVE_JOBD_LOOP_TRACE", "0").strip() or "0"
jobd_diag_ring_present = "SIFIVE_JOBD_DIAG_RING" in os.environ
jobd_diag_ring = os.environ.get("SIFIVE_JOBD_DIAG_RING", "1").strip() or "1"
jobd_beacon = os.environ.get("SIFIVE_JOBD_BEACON", "0").strip() or "0"
jobd_sifive_id = os.environ.get("SIFIVE_JOBD_SIFIVE_ID", "").strip()
jobd_mbox_poll_present = "SIFIVE_JOBD_MBOX_POLL" in os.environ
jobd_mbox_poll = os.environ.get("SIFIVE_JOBD_MBOX_POLL", "0").strip() or "0"
jobd_initial_mbox_poll = os.environ.get("SIFIVE_JOBD_INITIAL_MBOX_POLL", "").strip()
jobd_poll_timeout_ms = os.environ.get("SIFIVE_JOBD_POLL_TIMEOUT_MS", "").strip()
jobd_initial_scan_us = os.environ.get("SIFIVE_JOBD_INITIAL_SCAN_US", "").strip()
jobd_idle_sleep_us = os.environ.get("SIFIVE_JOBD_IDLE_SLEEP_US", "").strip()
jobd_arg_wait_us = os.environ.get("SIFIVE_JOBD_ARG_WAIT_US", "").strip()
jobd_force_elf = os.environ.get("SIFIVE_JOBD_FORCE_ELF", "0").strip() or "0"
jobd_elf_fallback = os.environ.get("SIFIVE_JOBD_ELF_FALLBACK", "").strip()
jobd_generic_noop = os.environ.get("SIFIVE_JOBD_GENERIC_NOOP", "").strip()
jobd_preloaded_noop = os.environ.get("SIFIVE_JOBD_PRELOADED_NOOP", "0").strip() or "0"
jobd_gemm_noop = os.environ.get("SIFIVE_JOBD_GEMM_NOOP", "0").strip() or "0"
jobd_gemm_tiled = os.environ.get("SIFIVE_JOBD_GEMM_TILED", "").strip()
jobd_gemm_copy_io = os.environ.get("SIFIVE_JOBD_GEMM_COPY_IO", "").strip()
jobd_gemm_single_thread_ops = os.environ.get("SIFIVE_JOBD_GEMM_SINGLE_THREAD_OPS", "").strip()
jobd_shared_ddr_payload_pwrite = os.environ.get("SIFIVE_JOBD_SHARED_DDR_PAYLOAD_PWRITE", "").strip()
jobd_full_ddr_map_present = "SIFIVE_JOBD_FULL_DDR_MAP" in os.environ
jobd_full_ddr_map = os.environ.get("SIFIVE_JOBD_FULL_DDR_MAP", "0").strip() or "0"
jobd_full_ddr_map_bytes = os.environ.get("SIFIVE_JOBD_FULL_DDR_MAP_BYTES", "").strip()
jobd_claim_id_present = "SIFIVE_JOBD_CLAIM_ID" in os.environ
jobd_claim_id = os.environ.get("SIFIVE_JOBD_CLAIM_ID", "0").strip() or "0"
jobd_sifive_id_ioctl = os.environ.get("SIFIVE_JOBD_SIFIVE_ID_IOCTL", "0").strip() or "0"
jobd_force_pread = os.environ.get("SIFIVE_JOBD_FORCE_PREAD", "0").strip() or "0"
jobd_control_pread = os.environ.get("SIFIVE_JOBD_CONTROL_PREAD", "").strip()
jobd_control_window_read = os.environ.get("SIFIVE_JOBD_CONTROL_WINDOW_READ", "").strip()
jobd_force_devmem = os.environ.get("SIFIVE_JOBD_FORCE_DEVMEM", "0").strip() or "0"
jobd_devmem_direct = os.environ.get("SIFIVE_JOBD_DEVMEM_DIRECT", os.environ.get("HETGPU_SIFIVE_JOBD_DEVMEM_DIRECT", "")).strip()
jobd_devmem_direct_status = os.environ.get("SIFIVE_JOBD_DEVMEM_DIRECT_STATUS", os.environ.get("HETGPU_SIFIVE_JOBD_DEVMEM_DIRECT_STATUS", "")).strip()
jobd_mbox_status_mmap = os.environ.get("SIFIVE_JOBD_MBOX_STATUS_MMAP", os.environ.get("HETGPU_SIFIVE_JOBD_MBOX_STATUS_MMAP", "")).strip()
jobd_shared_ddr_payload_sync = os.environ.get("SIFIVE_JOBD_SHARED_DDR_PAYLOAD_SYNC", os.environ.get("HETGPU_SIFIVE_JOBD_SHARED_DDR_PAYLOAD_SYNC", "")).strip()
jobd_status_control_window = os.environ.get("SIFIVE_JOBD_STATUS_CONTROL_WINDOW", "").strip()
jobd_status_mmap_fallback = os.environ.get("SIFIVE_JOBD_STATUS_MMAP_FALLBACK", "").strip()
jobd_status_pwrite_present = "SIFIVE_JOBD_STATUS_PWRITE" in os.environ
jobd_status_pwrite = os.environ.get("SIFIVE_JOBD_STATUS_PWRITE", "0").strip()
jobd_completion_mirror_off = os.environ.get("SIFIVE_JOBD_COMPLETION_MIRROR_OFF", "").strip()
jobd_aligned_completion_record = os.environ.get("SIFIVE_JOBD_ALIGNED_COMPLETION_RECORD", "").strip()
jobd_dual_offset_write = os.environ.get("SIFIVE_JOBD_DUAL_OFFSET_WRITE", "").strip()
jobd_msync_present = "SIFIVE_JOBD_MSYNC" in os.environ
jobd_msync = os.environ.get("SIFIVE_JOBD_MSYNC", "0").strip()
jobd_status_msync = os.environ.get("SIFIVE_JOBD_STATUS_MSYNC", "").strip()
jobd_rms_debug = os.environ.get("SIFIVE_JOBD_RMS_DEBUG", "").strip()
jobd_rms_local_copy = os.environ.get("SIFIVE_JOBD_RMS_LOCAL_COPY", "").strip()
jobd_rms_rvv = os.environ.get("SIFIVE_JOBD_RMS_RVV", "").strip()
jobd_rms_output_pwrite = os.environ.get("SIFIVE_JOBD_RMS_OUTPUT_PWRITE", "").strip()
jobd_rms_write_attempts = os.environ.get("SIFIVE_JOBD_RMS_WRITE_ATTEMPTS", "").strip()
jobd_rms_write_chunk_bytes = os.environ.get("SIFIVE_JOBD_RMS_WRITE_CHUNK_BYTES", "").strip()
jobd_repair_writeback = os.environ.get("SIFIVE_JOBD_REPAIR_WRITEBACK", "").strip()
jobd_repair_writeback_attempts = os.environ.get("SIFIVE_JOBD_REPAIR_WRITEBACK_ATTEMPTS", "").strip()
jobd_repair_writeback_sleep_us = os.environ.get("SIFIVE_JOBD_REPAIR_WRITEBACK_SLEEP_US", "").strip()
jobd_repair_writeback_chunk_bytes = os.environ.get("SIFIVE_JOBD_REPAIR_WRITEBACK_CHUNK_BYTES", "").strip()
jobd_sync_write_chunks = os.environ.get("SIFIVE_JOBD_SYNC_WRITE_CHUNKS", "").strip()
jobd_cbo_inval = os.environ.get("SIFIVE_JOBD_CBO_INVAL", "0").strip() or "0"
jobd_cbo_flush = os.environ.get("SIFIVE_JOBD_CBO_FLUSH", "0").strip() or "0"
jobd_cbo_block_bytes = os.environ.get("SIFIVE_JOBD_CBO_BLOCK_BYTES", "").strip()
jobd_cbo_op = os.environ.get("SIFIVE_JOBD_CBO_OP", "").strip()
jobd_evict_after_write = os.environ.get("SIFIVE_JOBD_EVICT_AFTER_WRITE_BYTES", "").strip()
jobd_notify_irq = os.environ.get("SIFIVE_JOBD_NOTIFY_IRQ", "").strip()
jobd_heartbeat = os.environ.get("SIFIVE_JOBD_HEARTBEAT", "0").strip() or "0"
jobd_boot_marker = os.environ.get("SIFIVE_JOBD_BOOT_MARKER", "0").strip() or "0"
jobd_early_devmem_marker_present = "SIFIVE_JOBD_EARLY_DEVMEM_MARKER" in os.environ
jobd_early_devmem_marker = os.environ.get("SIFIVE_JOBD_EARLY_DEVMEM_MARKER", "0").strip() or "0"
jobd_seed_current_jobs = os.environ.get("SIFIVE_JOBD_SEED_CURRENT_JOBS", "").strip()
jobd_shared_ddr_mmap_user_off = os.environ.get("SIFIVE_JOBD_SHARED_DDR_MMAP_USER_OFF", "").strip()
jobd_shared_ddr_user_off = os.environ.get("SIFIVE_JOBD_SHARED_DDR_USER_OFF", "").strip()
jobd_shared_ddr_fd_user_off = os.environ.get("SIFIVE_JOBD_SHARED_DDR_FD_USER_OFF", "").strip()
jobd_shared_ddr_dev = os.environ.get("SIFIVE_JOBD_SHARED_DDR_DEV", "").strip()
jobd_bin_bcast_local_max = os.environ.get("SIFIVE_JOBD_BIN_BCAST_LOCAL_MAX_BYTES", "").strip()
jobd_rope_local_max = os.environ.get("SIFIVE_JOBD_ROPE_LOCAL_MAX_BYTES", "").strip()
jobd_mmvf_local_x_max = os.environ.get("SIFIVE_JOBD_MMVF_LOCAL_X_MAX_BYTES", "").strip()
jobd_mmvf_local_y_max = os.environ.get("SIFIVE_JOBD_MMVF_LOCAL_Y_MAX_BYTES", "").strip()
jobd_mmvf_copy_io = os.environ.get("SIFIVE_JOBD_MMVF_COPY_IO", "").strip()
jobd_mmvf_compute = os.environ.get("SIFIVE_JOBD_MMVF_COMPUTE", "").strip()
jobd_arg_slot_scan = os.environ.get("SIFIVE_JOBD_ARG_SLOT_SCAN", os.environ.get("HETGPU_SIFIVE_JOBD_ARG_SLOT_SCAN", "")).strip()
jobd_redispatch_seen_arg_slot = os.environ.get("SIFIVE_JOBD_REDISPATCH_SEEN_ARG_SLOT", os.environ.get("HETGPU_SIFIVE_JOBD_REDISPATCH_SEEN_ARG_SLOT", "")).strip()
jobd_kernel_metadata_first = os.environ.get("SIFIVE_JOBD_KERNEL_METADATA_FIRST", os.environ.get("HETGPU_SIFIVE_JOBD_KERNEL_METADATA_FIRST", "")).strip()
jobd_fork_elf = os.environ.get("SIFIVE_JOBD_FORK_ELF", os.environ.get("HETGPU_SIFIVE_JOBD_FORK_ELF", "")).strip()
jobd_fork_elf_timeout = os.environ.get("SIFIVE_JOBD_FORK_ELF_TIMEOUT_MS", os.environ.get("HETGPU_SIFIVE_JOBD_FORK_ELF_TIMEOUT_MS", "")).strip()
jobd_kernel_slot_map = os.environ.get("SIFIVE_JOBD_KERNEL_SLOT_MAP", os.environ.get("HETGPU_SIFIVE_JOBD_KERNEL_SLOT_MAP", "")).strip()
jobd_kernel_slot_map_bytes = os.environ.get("SIFIVE_JOBD_KERNEL_SLOT_MAP_BYTES", os.environ.get("HETGPU_SIFIVE_JOBD_KERNEL_SLOT_MAP_BYTES", "")).strip()
jobd_kernel_slot_map_off = os.environ.get("SIFIVE_JOBD_KERNEL_SLOT_MAP_OFF", os.environ.get("HETGPU_SIFIVE_JOBD_KERNEL_SLOT_MAP_OFF", "")).strip()
jobd_helper_io_chunk_bytes = os.environ.get("SIFIVE_JOBD_HELPER_IO_CHUNK_BYTES", os.environ.get("HETGPU_SIFIVE_JOBD_HELPER_IO_CHUNK_BYTES", "")).strip()
jobd_xsfmm_smoke = os.environ.get("SIFIVE_JOBD_XSFMM_SMOKE", os.environ.get("HETGPU_SIFIVE_JOBD_XSFMM_SMOKE", "")).strip()
jobd_xsfmm_gemm = os.environ.get("SIFIVE_JOBD_XSFMM_GEMM", os.environ.get("HETGPU_SIFIVE_JOBD_XSFMM_GEMM", "")).strip()
jobd_env_in_bashrc = os.environ.get("SIFIVE_JOBD_ENV_IN_BASHRC", "0").strip() or "0"
jobd_minimal_rcs = env_true("SIFIVE_JOBD_MINIMAL_RCS", False)
jobd_ddr_ko = os.environ.get("SIFIVE_JOBD_DDR_KO", "").strip()
jobd_ddr_ko_args = os.environ.get("SIFIVE_JOBD_DDR_KO_ARGS", "").strip()
jobd_ddr_ioctl = os.environ.get("SIFIVE_JOBD_DDR_IOCTL", os.environ.get("HETGPU_SIFIVE_JOBD_DDR_IOCTL", "")).strip()

rcs_lines = [
    "#!/bin/sh",
]
if jobd_threads_present or jobd_threads != "4":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_KERNEL_THREADS={jobd_threads}")
if jobd_trace != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_TRACE={jobd_trace}")
if jobd_log_present:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_LOG={jobd_log}")
if jobd_kmsg_present:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_KMSG={jobd_kmsg}")
if jobd_progress_status != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_PROGRESS_STATUS={jobd_progress_status}")
if jobd_progress_completion != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_PROGRESS_COMPLETION={jobd_progress_completion}")
if jobd_loop_trace != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_LOOP_TRACE={jobd_loop_trace}")
if jobd_diag_ring_present:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_DIAG_RING={jobd_diag_ring}")
if jobd_beacon != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_BEACON={jobd_beacon}")
if jobd_sifive_id:
    rcs_lines.append(f"export HETGPU_SIFIVE_ID={jobd_sifive_id}")
if jobd_mbox_poll_present or jobd_mbox_poll != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_MBOX_POLL={jobd_mbox_poll}")
if jobd_initial_mbox_poll:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_INITIAL_MBOX_POLL={jobd_initial_mbox_poll}")
if jobd_poll_timeout_ms:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_POLL_TIMEOUT_MS={jobd_poll_timeout_ms}")
if jobd_initial_scan_us:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_INITIAL_SCAN_US={jobd_initial_scan_us}")
if jobd_idle_sleep_us:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_IDLE_SLEEP_US={jobd_idle_sleep_us}")
if jobd_arg_wait_us:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_ARG_WAIT_US={jobd_arg_wait_us}")
if jobd_force_elf != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_FORCE_ELF={jobd_force_elf}")
if jobd_elf_fallback:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_ELF_FALLBACK={jobd_elf_fallback}")
if jobd_generic_noop:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_GENERIC_NOOP={jobd_generic_noop}")
if jobd_preloaded_noop != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_PRELOADED_NOOP={jobd_preloaded_noop}")
if jobd_gemm_noop != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_GEMM_NOOP={jobd_gemm_noop}")
if jobd_gemm_tiled:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_GEMM_TILED={jobd_gemm_tiled}")
if jobd_gemm_copy_io:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_GEMM_COPY_IO={jobd_gemm_copy_io}")
if jobd_gemm_single_thread_ops:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_GEMM_SINGLE_THREAD_OPS={jobd_gemm_single_thread_ops}")
if jobd_shared_ddr_payload_pwrite:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_SHARED_DDR_PAYLOAD_PWRITE={jobd_shared_ddr_payload_pwrite}")
if jobd_full_ddr_map_present or jobd_full_ddr_map != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_FULL_DDR_MAP={jobd_full_ddr_map}")
if jobd_full_ddr_map_bytes:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_FULL_DDR_MAP_BYTES={jobd_full_ddr_map_bytes}")
if jobd_claim_id_present or jobd_claim_id != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_CLAIM_ID={jobd_claim_id}")
if jobd_sifive_id_ioctl != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_SIFIVE_ID_IOCTL={jobd_sifive_id_ioctl}")
if jobd_ddr_ioctl:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_DDR_IOCTL={jobd_ddr_ioctl}")
if jobd_force_pread != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_FORCE_PREAD={jobd_force_pread}")
if jobd_control_pread:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_CONTROL_PREAD={jobd_control_pread}")
if jobd_control_window_read:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_CONTROL_WINDOW_READ={jobd_control_window_read}")
if jobd_force_devmem != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_FORCE_DEVMEM={jobd_force_devmem}")
if jobd_devmem_direct:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_DEVMEM_DIRECT={jobd_devmem_direct}")
if jobd_devmem_direct_status:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_DEVMEM_DIRECT_STATUS={jobd_devmem_direct_status}")
if jobd_mbox_status_mmap:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_MBOX_STATUS_MMAP={jobd_mbox_status_mmap}")
if jobd_shared_ddr_payload_sync:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_SHARED_DDR_PAYLOAD_SYNC={jobd_shared_ddr_payload_sync}")
if jobd_status_control_window:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_STATUS_CONTROL_WINDOW={jobd_status_control_window}")
if jobd_status_mmap_fallback:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_STATUS_MMAP_FALLBACK={jobd_status_mmap_fallback}")
if jobd_status_pwrite_present:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_STATUS_PWRITE={jobd_status_pwrite}")
if jobd_completion_mirror_off:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_COMPLETION_MIRROR_OFF={jobd_completion_mirror_off}")
if jobd_aligned_completion_record:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_ALIGNED_COMPLETION_RECORD={jobd_aligned_completion_record}")
if jobd_dual_offset_write:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_DUAL_OFFSET_WRITE={jobd_dual_offset_write}")
if jobd_msync_present:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_MSYNC={jobd_msync}")
if jobd_status_msync:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_STATUS_MSYNC={jobd_status_msync}")
if jobd_rms_debug:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_RMS_DEBUG={jobd_rms_debug}")
if jobd_rms_local_copy:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_RMS_LOCAL_COPY={jobd_rms_local_copy}")
if jobd_rms_rvv:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_RMS_RVV={jobd_rms_rvv}")
if jobd_rms_output_pwrite:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_RMS_OUTPUT_PWRITE={jobd_rms_output_pwrite}")
if jobd_rms_write_attempts:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_RMS_WRITE_ATTEMPTS={jobd_rms_write_attempts}")
if jobd_rms_write_chunk_bytes:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_RMS_WRITE_CHUNK_BYTES={jobd_rms_write_chunk_bytes}")
if jobd_repair_writeback:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_REPAIR_WRITEBACK={jobd_repair_writeback}")
if jobd_repair_writeback_attempts:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_REPAIR_WRITEBACK_ATTEMPTS={jobd_repair_writeback_attempts}")
if jobd_repair_writeback_sleep_us:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_REPAIR_WRITEBACK_SLEEP_US={jobd_repair_writeback_sleep_us}")
if jobd_repair_writeback_chunk_bytes:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_REPAIR_WRITEBACK_CHUNK_BYTES={jobd_repair_writeback_chunk_bytes}")
if jobd_sync_write_chunks:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_SYNC_WRITE_CHUNKS={jobd_sync_write_chunks}")
if jobd_cbo_inval != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_CBO_INVAL={jobd_cbo_inval}")
if jobd_cbo_flush != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_CBO_FLUSH={jobd_cbo_flush}")
if jobd_cbo_block_bytes:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_CBO_BLOCK_BYTES={jobd_cbo_block_bytes}")
if jobd_cbo_op:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_CBO_OP={jobd_cbo_op}")
if jobd_evict_after_write:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_EVICT_AFTER_WRITE_BYTES={jobd_evict_after_write}")
if jobd_notify_irq:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_NOTIFY_IRQ={jobd_notify_irq}")
if jobd_heartbeat != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_HEARTBEAT={jobd_heartbeat}")
if jobd_boot_marker != "0":
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_BOOT_MARKER={jobd_boot_marker}")
if jobd_early_devmem_marker_present:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_EARLY_DEVMEM_MARKER={jobd_early_devmem_marker}")
if jobd_seed_current_jobs:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_SEED_CURRENT_JOBS={jobd_seed_current_jobs}")
if jobd_shared_ddr_mmap_user_off:
    rcs_lines.append(f"export HETGPU_SIFIVE_SHARED_DDR_MMAP_USER_OFF={jobd_shared_ddr_mmap_user_off}")
if jobd_shared_ddr_user_off:
    rcs_lines.append(f"export HETGPU_SIFIVE_SHARED_DDR_USER_OFF={jobd_shared_ddr_user_off}")
if jobd_shared_ddr_fd_user_off:
    rcs_lines.append(f"export HETGPU_SIFIVE_SHARED_DDR_FD_USER_OFF={jobd_shared_ddr_fd_user_off}")
if jobd_shared_ddr_dev:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_SHARED_DDR_DEV={jobd_shared_ddr_dev}")
if jobd_bin_bcast_local_max:
    rcs_lines.append(f"export SIFIVE_JOBD_BIN_BCAST_LOCAL_MAX_BYTES={jobd_bin_bcast_local_max}")
if jobd_rope_local_max:
    rcs_lines.append(f"export SIFIVE_JOBD_ROPE_LOCAL_MAX_BYTES={jobd_rope_local_max}")
if jobd_mmvf_local_x_max:
    rcs_lines.append(f"export SIFIVE_JOBD_MMVF_LOCAL_X_MAX_BYTES={jobd_mmvf_local_x_max}")
if jobd_mmvf_local_y_max:
    rcs_lines.append(f"export SIFIVE_JOBD_MMVF_LOCAL_Y_MAX_BYTES={jobd_mmvf_local_y_max}")
if jobd_mmvf_copy_io:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_MMVF_COPY_IO={jobd_mmvf_copy_io}")
if jobd_mmvf_compute:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_MMVF_COMPUTE={jobd_mmvf_compute}")
if jobd_arg_slot_scan:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_ARG_SLOT_SCAN={jobd_arg_slot_scan}")
if jobd_redispatch_seen_arg_slot:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_REDISPATCH_SEEN_ARG_SLOT={jobd_redispatch_seen_arg_slot}")
if jobd_kernel_metadata_first:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_KERNEL_METADATA_FIRST={jobd_kernel_metadata_first}")
if jobd_fork_elf:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_FORK_ELF={jobd_fork_elf}")
if jobd_fork_elf_timeout:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_FORK_ELF_TIMEOUT_MS={jobd_fork_elf_timeout}")
if jobd_kernel_slot_map:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_KERNEL_SLOT_MAP={jobd_kernel_slot_map}")
if jobd_kernel_slot_map_bytes:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_KERNEL_SLOT_MAP_BYTES={jobd_kernel_slot_map_bytes}")
if jobd_kernel_slot_map_off:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_KERNEL_SLOT_MAP_OFF={jobd_kernel_slot_map_off}")
if jobd_helper_io_chunk_bytes:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_HELPER_IO_CHUNK_BYTES={jobd_helper_io_chunk_bytes}")
if jobd_xsfmm_smoke:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_XSFMM_SMOKE={jobd_xsfmm_smoke}")
if jobd_xsfmm_gemm:
    rcs_lines.append(f"export HETGPU_SIFIVE_JOBD_XSFMM_GEMM={jobd_xsfmm_gemm}")
shared_ddr_base = os.environ.get("SIFIVE_JOBD_SHARED_DDR_BASE", "").strip()
shared_ddr_size = os.environ.get("SIFIVE_JOBD_SHARED_DDR_SIZE", "").strip()
shared_ddr_sifive_base = os.environ.get("SIFIVE_JOBD_SHARED_DDR_SIFIVE_BASE", os.environ.get("HETGPU_SIFIVE_SHARED_DDR_SIFIVE_BASE", "")).strip()
if shared_ddr_base:
    rcs_lines.append(f"export HETGPU_SIFIVE_SHARED_DDR_BASE={shared_ddr_base}")
if shared_ddr_size:
    rcs_lines.append(f"export HETGPU_SIFIVE_SHARED_DDR_BYTES={shared_ddr_size}")
if shared_ddr_sifive_base:
    rcs_lines.append(f"export HETGPU_SIFIVE_SHARED_DDR_SIFIVE_BASE={shared_ddr_sifive_base}")
rcs_lines.append("exec /home/root/sifive_skl_test --mbox=/dev/mbox")
compact_rcs_lines = [rcs_lines[0]]
exports = [line[len("export "):] for line in rcs_lines[1:] if line.startswith("export ")]
rest = [line for line in rcs_lines[1:] if not line.startswith("export ")]
env_payload = None
if jobd_minimal_rcs:
    compact_rcs_lines = [
        rcs_lines[0],
        *([f"export {' '.join(exports)}"] if exports else []),
        *([f"insmod /home/root/ddr.ko {jobd_ddr_ko_args} || true"] if jobd_ddr_ko else []),
        "exec /home/root/sifive_skl_test --mbox=/dev/mbox",
    ]
elif jobd_env_in_bashrc != "0":
    env_payload = ("\n".join(exports) + "\n").encode()
    compact_rcs_lines = [
        rcs_lines[0],
        *([f"insmod /home/root/ddr.ko {jobd_ddr_ko_args} || true"] if jobd_ddr_ko else []),
        "set -a;. /etc/skel/.bashrc;set +a",
        "exec /home/root/sifive_skl_test --mbox=/dev/mbox --config=/dev/null",
    ]
elif exports:
    compact_rcs_lines.append("export " + " ".join(exports))
if jobd_env_in_bashrc == "0" and not jobd_minimal_rcs:
    compact_rcs_lines.extend(rest)
rcs_lines = compact_rcs_lines
rcs = ("\n".join(rcs_lines) + "\n").encode()

default_conf = b"""gemm 2 2 2 0x20000800 0x20000840 0x20002800 2 2 2 0 0 1
softmax 0x20000900 0x20002900 1 4 4
rmsnorm 0x20000a00 0x20000a40 0x20002a00 1 4 0.00001
"""

outer_offset = parse_u64_env("LX500_SIFIVE_INITRAMFS_OFFSET")
outer_length = parse_u64_env("LX500_SIFIVE_INITRAMFS_LENGTH")
if outer_offset is not None:
    if outer_offset < 0 or outer_offset >= len(image):
        raise SystemExit(f"LX500_SIFIVE_INITRAMFS_OFFSET 0x{outer_offset:x} outside image size 0x{len(image):x}")
    if outer_length is None:
        outer_length = len(image) - outer_offset
    if outer_length < 0 or outer_offset + outer_length > len(image):
        raise SystemExit(
            f"LX500_SIFIVE_INITRAMFS window off=0x{outer_offset:x} len=0x{outer_length:x} "
            f"exceeds image size 0x{len(image):x}"
        )
    print(f"using explicit initramfs window off=0x{outer_offset:x} len=0x{outer_length:x}")
    outer = parse_newc(image, outer_offset, outer_length)
else:
    outer = parse_newc(image)
patched = 0
patched += patch_payload(outer, "outer", "home/root/sifive_skl_test", jobd_bytes, 0)
if jobd_ddr_ko:
    patched += patch_payload(outer, "outer", "home/root/ddr.ko", Path(jobd_ddr_ko).read_bytes(), 0)
patched += patch_payload(outer, "outer", "etc/init.d/rcS", rcs, ord("\n"))
patched += patch_payload(outer, "outer", "etc/skel/.bashrc", env_payload or conf_bytes or default_conf, ord("\n"))

inner_name = "core-image-minimal-qemuriscv64.cpio"
if inner_name in outer:
    inner = outer[inner_name]
    inner_entries = parse_newc(image, inner["data"], inner["size"])
    patched += patch_payload(inner_entries, "inner", "home/root/sifive_skl_test", jobd_bytes, 0)
    if jobd_ddr_ko:
        patched += patch_payload(inner_entries, "inner", "home/root/ddr.ko", Path(jobd_ddr_ko).read_bytes(), 0)
    patched += patch_payload(inner_entries, "inner", "etc/init.d/rcS", rcs, ord("\n"))
    patched += patch_payload(inner_entries, "inner", "etc/skel/.bashrc", env_payload or conf_bytes or default_conf, ord("\n"))

if patched == 0:
    raise SystemExit("no payloads patched")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(image)
print(f"wrote {out} ({len(image)} bytes)")
PY
