#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mask="${1:-${HETGPU_PACC_RECOVER_MASK:-0x5}}"
firmware="${HETGPU_PACC_RECOVER_FIRMWARE:-lanxin/lx500_pacc.bin}"
settle_s="${HETGPU_PACC_RECOVER_SETTLE_S:-45}"
patch_pacc_id_env="${HETGPU_PACC_RECOVER_PATCH_PACC_ID_ENV:-0}"
sys_cold_reset="${HETGPU_PACC_RECOVER_SYS_COLD_RESET:-1}"
force_resetpc_reload="${HETGPU_PACC_RECOVER_FORCE_RESETPC_RELOAD:-1}"
staged_boot="${HETGPU_PACC_RECOVER_STAGED_BOOT:-1}"
pre_reset_mask="${HETGPU_PACC_RECOVER_PRE_RESET_MASK:-}"
pre_reset_settle_s="${HETGPU_PACC_RECOVER_PRE_RESET_SETTLE_S:-1}"

if [[ ! -e "${repo_root}/pacc_boot_helper/hetgpu_pacc_boot.ko" ]]; then
  echo "missing ${repo_root}/pacc_boot_helper/hetgpu_pacc_boot.ko" >&2
  exit 2
fi

sudo -n rmmod hetgpu_pacc_boot 2>/dev/null || true
if [[ -n "${pre_reset_mask}" && "${pre_reset_mask}" != "0" ]]; then
  sudo -n insmod "${repo_root}/pacc_boot_helper/hetgpu_pacc_boot.ko" \
    reset_only_on_load=1 boot_on_load=0 cores_are_wfi=1 \
    pacc_mask="${pre_reset_mask}" core_mask="${HETGPU_PACC_RECOVER_CORE_MASK:-0xf}"
  sleep "${pre_reset_settle_s}"
  sudo -n rmmod hetgpu_pacc_boot 2>/dev/null || true
fi
sudo -n insmod "${repo_root}/pacc_boot_helper/hetgpu_pacc_boot.ko" \
  boot_on_load=1 cores_are_wfi=1 pacc_mask="${mask}" core_mask="${HETGPU_PACC_RECOVER_CORE_MASK:-0xf}" \
  staged_boot="${staged_boot}" \
  firmware_name="${firmware}" \
  load_addr="${HETGPU_PACC_RECOVER_LOAD_ADDR:-0xc0000000}" \
  entry="${HETGPU_PACC_RECOVER_ENTRY:-0xc0000000}" \
  reserved_size="${HETGPU_PACC_RECOVER_RESERVED_SIZE:-0x08000000}" \
  per_pacc_load="${HETGPU_PACC_RECOVER_PER_PACC_LOAD:-1}" \
  per_pacc_entry="${HETGPU_PACC_RECOVER_PER_PACC_ENTRY:-1}" \
  patch_pacc_id_env="${patch_pacc_id_env}" \
  local_doorbell_bit="${HETGPU_PACC_RECOVER_LOCAL_DOORBELL_BIT:-1}" \
  clear_db_status=1 set_nonsecure="${HETGPU_PACC_RECOVER_SET_NONSECURE:-0}" send_base_cmd=1 \
  send_base_after_core_release="${HETGPU_PACC_RECOVER_SEND_BASE_AFTER_RELEASE:-1}" \
  send_base_delay_ms="${HETGPU_PACC_RECOVER_SEND_BASE_DELAY_MS:-20}" \
  sys_cold_reset="${sys_cold_reset}" force_resetpc_reload="${force_resetpc_reload}"

sleep "${settle_s}"
sudo -n rmmod hetgpu_pacc_boot 2>/dev/null || true
sudo -n chmod 666 /dev/pacc* /dev/hetgpu_pacc_mbox* 2>/dev/null || true

echo "PACC-only recovery completed for mask=${mask}, firmware=${firmware}"
