#!/usr/bin/env bash
set -eo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"

source /au250_xrt/env.sh >/dev/null
au250-temp

runtime_dir="$repo_root/target/au250-runtime"
mkdir -p "$runtime_dir/bin"
install -m 0755 /opt/miniconda3/bin/ninja "$runtime_dir/bin/ninja"

au250-run bash -lc '
set -euo pipefail
export RUSTUP_HOME=/work/target/au250-runtime/rustup
export CARGO_HOME=/work/target/au250-runtime/cargo
export CARGO_TARGET_DIR=/work/target/au250-app215
export PATH=/work/target/au250-runtime/bin:$CARGO_HOME/bin:$PATH
export CARGO_INCREMENTAL=0

if [[ ! -x "$CARGO_HOME/bin/cargo" ]]; then
    curl --proto "=https" --tlsv1.2 -fsS https://sh.rustup.rs \
        -o /work/target/au250-runtime/rustup-init.sh
    sh /work/target/au250-runtime/rustup-init.sh \
        -y --profile minimal --default-toolchain 1.92.0
fi

export HETGPU_XRT_AU250_TEST=1
export HETGPU_XRT_XCLBIN=/au250_xrt/example/asym9_bs9_2641toks.xclbin
export HETGPU_XRT_IP_NAME=ternip_big:ternip_big_1
export HETGPU_XRT_INSTANCE=0
export HETGPU_XRT_MEMORY_GROUP=0
export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
export HETGPU_XRT_TIMEOUT_MS=10000

cargo test -p zluda --features nvidia --no-default-features \
    au250_tmatmul_runs_when_requested -- --ignored --nocapture

xbutil examine -d 0000:64:00.1 \
    -r dynamic-regions -r error -r firewall -r thermal
'

au250-temp
