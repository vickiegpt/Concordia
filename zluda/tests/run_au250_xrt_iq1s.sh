#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"

if [[ ${1:-} == "--inside" ]]; then
    export RUSTUP_HOME=/work/target/au250-runtime/rustup
    export CARGO_HOME=/work/target/au250-runtime/cargo
    export CARGO_TARGET_DIR=/work/target/au250-app215
    export PATH=/work/target/au250-runtime/bin:$CARGO_HOME/bin:$PATH
    export CARGO_INCREMENTAL=0

    xclbin=/au250_xrt/example/MaxCores_370M.xclbin
    xclbin_info=$(xclbinutil --info --input "$xclbin" 2>&1)
    printf "%s\n" "$xclbin_info"
    require_instance_bank() {
        local instance=$1
        local bank=$2
        awk -v instance="$instance" -v bank="$bank" '
            $1 == "Instance:" { active = ($2 == instance) }
            active && $1 == "Memory:" && $2 == bank { found = 1 }
            END { exit !found }
        ' <<<"$xclbin_info" || {
            echo "MaxCores topology missing ${instance}->${bank}" >&2
            exit 1
        }
    }
    require_instance_bank ternip_big_1 bank0
    require_instance_bank ternip_big_2 bank3
    require_instance_bank ternip_big_3 bank2
    require_instance_bank ternip_small_1 bank1

    export HETGPU_XRT_AU250_IQ1S_TEST=1
    export HETGPU_XRT_XCLBIN="$xclbin"
    export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
    export HETGPU_XRT_TIMEOUT_MS=10000
    export HETGPU_XRT_EXECUTION_LOG=/work/target/au250-iq1s-live.jsonl
    rm -f /work/target/au250-iq1s-live.jsonl

    cargo test -p zluda --features nvidia --no-default-features \
        au250_iq1s_two_by_two_tiles_match_reference -- --ignored --nocapture

    python3 - <<'PY'
import json
from pathlib import Path

records = [json.loads(line) for line in Path("/work/target/au250-iq1s-live.jsonl").read_text().splitlines()]
if len(records) != 1:
    raise SystemExit(f"expected one XRT IQ1_S evidence record, got {len(records)}")
evidence = records[0]["evidence"]
if evidence["comparison_status"] != "pass":
    raise SystemExit(f"XRT IQ1_S comparison did not pass: {evidence}")
if sum(value > 0 for value in evidence["per_cu_submissions"]) < 2:
    raise SystemExit(f"fewer than two CUs were used: {evidence}")
if not (-4096 <= evidence["raw_min"] <= evidence["raw_max"] <= 4096):
    raise SystemExit(f"raw component bounds are invalid: {evidence}")
PY

    health_report=$(xbutil examine -d 0000:64:00.1 \
        -r dynamic-regions -r error -r firewall -r thermal 2>&1)
    printf "%s\n" "$health_report"
    grep -Fq "Level 0 : 0x0 (GOOD)" <<<"$health_report" || {
        echo "AU250 health gate failed: firewall is not GOOD" >&2
        exit 1
    }
    for cu in ternip_big_1 ternip_big_2 ternip_big_3 ternip_small_1; do
        grep -Eq "${cu}.*\(DONE\)" <<<"$health_report" || {
            echo "AU250 health gate failed: ${cu} is not DONE" >&2
            exit 1
        }
    done
    if grep -Eiq "(^|[^[:alpha:]])fatal([^[:alpha:]]|$)" <<<"$health_report"; then
        echo "AU250 health gate failed: fatal error reported" >&2
        exit 1
    fi
    exit 0
fi

# env.sh currently reads an unset helper while loading, so source it before
# enabling nounset in callers and use its host-side thermal guard here.
set +u
source /au250_xrt/env.sh >/dev/null
set -u
temperature="$(_au250_fpga_temp)"
awk -v temperature="$temperature" 'BEGIN { exit !(temperature < 85) }' || {
    echo "AU250 temperature guard failed before launch: ${temperature} C" >&2
    exit 1
}

"$repo_root/tools/au250_hybrid_run.sh" \
    bash /work/zluda/tests/run_au250_xrt_iq1s.sh --inside

temperature="$(_au250_fpga_temp)"
awk -v temperature="$temperature" 'BEGIN { exit !(temperature < 85) }' || {
    echo "AU250 temperature guard failed after launch: ${temperature} C" >&2
    exit 1
}
echo "AU250 post-run temperature: ${temperature} C"
