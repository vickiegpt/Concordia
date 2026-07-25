#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-/tmp/lanxin_disagg_eval/xsfmm_image_contract_${STAMP}}"
SOURCE="${ROOT}/ext/pacc_runtime-sys/pacc_linux_jobd/xsfmm_ctx/xsfmm_ctx.c"
mkdir -p "$OUT"

HOST_BOOT_ID="$(cat /proc/sys/kernel/random/boot_id)"
EXPECTED_WORDS=(
    0x508772d7
    0x8416f2d7
    0x8422f357
    0x43e06057
    0x8402f357
    0xf2881077
    0x52567027
)

encoding_ok=1
: >"$OUT/encodings.tsv"
for word in "${EXPECTED_WORDS[@]}"; do
    count="$(grep -Foc "$word" "$SOURCE" || true)"
    printf '%s\t%s\n' "$word" "$count" | tee -a "$OUT/encodings.tsv"
    if [[ "$count" == "0" ]]; then
        encoding_ok=0
    fi
done

fpga_managers="$(find /sys/class/fpga_manager -mindepth 1 -maxdepth 1 2>/dev/null | wc -l || true)"
fpga_regions="$(find /sys/class/fpga_region -mindepth 1 -maxdepth 1 2>/dev/null | wc -l || true)"
fpga_bridges="$(find /sys/class/fpga_bridge -mindepth 1 -maxdepth 1 2>/dev/null | wc -l || true)"
mtd_devices="$(find /dev -maxdepth 1 -name 'mtd*' 2>/dev/null | wc -l || true)"
pacc_nodes="$(find /sys/firmware/devicetree/base/soc -maxdepth 1 -type d -name 'pacc@*' 2>/dev/null | wc -l || true)"
bitstreams="$(
    find /boot /lib/firmware/lanxin \
        -type f \( -name '*.bit' -o -name '*.rbf' -o -name '*.sof' \
        -o -name '*.pof' -o -name '*.jic' -o -name '*.xclbin' \) \
        2>/dev/null | wc -l || true
)"

cat >"$OUT/platform.env" <<EOF
host_boot_id=${HOST_BOOT_ID}
pacc_nodes=${pacc_nodes}
fpga_managers=${fpga_managers}
fpga_regions=${fpga_regions}
fpga_bridges=${fpga_bridges}
mtd_devices=${mtd_devices}
bitstream_files=${bitstreams}
encoding_source_ok=${encoding_ok}
required_abi=0.6.6
required_max_m=2048
required_max_n=32
required_max_k=6144
required_continuous_ops=64
required_completion_p99_us=10
EOF

python3 - "$OUT/platform.env" >"$OUT/platform.json" <<'PY'
import json
import sys

values = {}
for line in open(sys.argv[1], encoding="utf-8"):
    key, value = line.rstrip("\n").split("=", 1)
    if value.isdigit():
        value = int(value)
    values[key] = value
values["linux_reconfiguration_available"] = any(
    int(values[key]) > 0
    for key in ("fpga_managers", "fpga_regions", "mtd_devices", "bitstream_files")
)
print(json.dumps(values, indent=2, sort_keys=True))
PY

cat "$OUT/platform.json"
if [[ "$encoding_ok" != "1" ]]; then
    echo "contract=failed reason=firmware-encoding-source" | tee "$OUT/result.txt"
    exit 1
fi
if [[ "$fpga_managers" == "0" && "$fpga_regions" == "0" &&
      "$mtd_devices" == "0" && "$bitstreams" == "0" ]]; then
    echo "contract=blocked reason=no-linux-xm-image-programming-path" |
        tee "$OUT/result.txt"
    exit 2
fi
echo "contract=programming-path-present active-hardware-gates-required" |
    tee "$OUT/result.txt"
