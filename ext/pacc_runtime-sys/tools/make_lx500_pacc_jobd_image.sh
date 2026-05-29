#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor_image="${1:-/lib/firmware/lanxin/lx500_pacc.bin}"
output_image="${2:-${repo_root}/target/lx500_pacc_hetgpu_jobd.bin}"
jobd="${PACC_LINUX_JOBD_OUT:-${repo_root}/target/hetgpu_pacc_jobd}"
inject="$(mktemp -d)"
trap 'rm -rf "${inject}"' EXIT

"${repo_root}/tools/build_pacc_linux_jobd.sh" "${jobd}" >/dev/null

install -d "${inject}/usr/sbin" "${inject}/etc/rcS.d" "${inject}/etc"
install -m 0755 "${jobd}" "${inject}/usr/sbin/hetgpu_pacc_jobd"
cat > "${inject}/etc/hetgpu_pacc_jobs.conf" <<'EOF'
# Preloaded PACC jobs. Host doorbells select only job_id.
# Fill physical DDR addresses before running real Qwen:
# gemm    M N K A_PHYS B_PHYS C_PHYS LDA LDB LDC ALPHA_PHYS BETA_PHYS BATCH
# softmax SRC_PHYS DST_PHYS ROWS COLS STRIDE
# rmsnorm X_PHYS WEIGHT_PHYS Y_PHYS ROWS HIDDEN EPS
EOF
cat > "${inject}/etc/rcS.d/S99hetgpu-pacc-jobd" <<'EOF'
#!/bin/sh
case "$1" in
  start|"")
    echo "Starting hetgpu_pacc_jobd after host probe settles"
    (
      sleep "${HETGPU_PACC_JOBD_START_DELAY:-8}"
      export HETGPU_PACC_JOBD_KERNEL_THREADS="${HETGPU_PACC_JOBD_KERNEL_THREADS:-4}"
      export HETGPU_PACC_JOBD_LOG="${HETGPU_PACC_JOBD_LOG:-1}"
      export HETGPU_PACC_JOBD_KMSG="${HETGPU_PACC_JOBD_KMSG:-1}"
      export HETGPU_PACC_JOBD_TRACE="${HETGPU_PACC_JOBD_TRACE:-0}"
      export HETGPU_PACC_SHARED_DDR_BASE="${HETGPU_PACC_SHARED_DDR_BASE:-0x20100600000}"
      export HETGPU_PACC_SHARED_DDR_BYTES="${HETGPU_PACC_SHARED_DDR_BYTES:-0x100000000}"
      export HETGPU_PACC_SHARED_DDR_PACC_BASE="${HETGPU_PACC_SHARED_DDR_PACC_BASE:-0x80000000}"
      export HETGPU_PACC_JOBD_FORCE_DEVMEM="${HETGPU_PACC_JOBD_FORCE_DEVMEM:-1}"
      export HETGPU_PACC_JOBD_DEVMEM_DIRECT="${HETGPU_PACC_JOBD_DEVMEM_DIRECT:-1}"
      export HETGPU_PACC_JOBD_DEVMEM_DIRECT_STATUS="${HETGPU_PACC_JOBD_DEVMEM_DIRECT_STATUS:-1}"
      export HETGPU_PACC_JOBD_MBOX_STATUS_MMAP="${HETGPU_PACC_JOBD_MBOX_STATUS_MMAP:-0}"
      export HETGPU_PACC_JOBD_SHARED_DDR_PAYLOAD_SYNC="${HETGPU_PACC_JOBD_SHARED_DDR_PAYLOAD_SYNC:-0}"
      export HETGPU_PACC_JOBD_INITIAL_MBOX_POLL="${HETGPU_PACC_JOBD_INITIAL_MBOX_POLL:-0}"
      exec /usr/sbin/hetgpu_pacc_jobd --mbox=/dev/mbox
    ) >/dev/kmsg 2>&1 &
    ;;
  stop)
    killall hetgpu_pacc_jobd 2>/dev/null || true
    ;;
esac
EOF
chmod 0755 "${inject}/etc/rcS.d/S99hetgpu-pacc-jobd"

sudo -n env TMPDIR="${TMPDIR:-/tmp}" "${repo_root}/tools/repack_lx500_pacc_nested_initramfs.sh" \
  "${vendor_image}" "${inject}" "${output_image}"

sha256sum "${output_image}"
