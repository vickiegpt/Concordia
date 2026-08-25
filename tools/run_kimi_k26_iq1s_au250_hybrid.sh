#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

if [[ ${1:-} == "--inside" ]]; then
    proof_dir=$2
    compare_max=$3
    predict=$4
    benchmark=$5
    case "$proof_dir" in
        /work/.proof/kimi-au250-*) ;;
        *) echo "refusing unexpected proof directory: $proof_dir" >&2; exit 2 ;;
    esac
    [[ $compare_max =~ ^[0-9]+$ ]] || { echo "invalid comparison count: $compare_max" >&2; exit 2; }
    [[ $predict =~ ^[1-9][0-9]*$ ]] || { echo "invalid prediction count: $predict" >&2; exit 2; }
    install -d "$proof_dir"

    runner=/work/target/au250-bitnet-cuda130/bin/llama-cli
    tokenizer=/work/target/au250-bitnet-cuda130/bin/llama-tokenize
    model=/models/kimi/moonshotai_Kimi-K2.6-IQ1_S-00001-of-00006.gguf
    xclbin=/au250_xrt/example/MaxCores_370M.xclbin
    libnvcuda=/work/target/au250-app215/debug/libnvcuda.so
    for required in "$runner" "$tokenizer" "$model" "$xclbin" "$libnvcuda"; do
        [[ -e $required ]] || { echo "missing hybrid proof input: $required" >&2; exit 1; }
    done

    export LD_PRELOAD="$libnvcuda"
    export LD_LIBRARY_PATH=/work/target/au250-bitnet-cuda130/3rdparty/llama.cpp/src:/work/target/au250-bitnet-cuda130/3rdparty/llama.cpp/ggml/src:/work/target/au250-bitnet-cuda130/3rdparty/llama.cpp/ggml/src/ggml-cuda:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
    export HETGPU_TMATMUL_BACKEND=xrt
    export HETGPU_BITNET_DISAGGREGATE=1
    export HETGPU_BITNET_DISAGG_STRICT=1
    export HETGPU_TMATMUL_HARDWARE_MATMUL=1
    export HETGPU_XRT_XCLBIN="$xclbin"
    export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
    export HETGPU_XRT_TIMEOUT_MS=10000
    export HETGPU_XRT_COMPARE_MAX_LAUNCHES="$compare_max"
    export HETGPU_BITNET_CXL_KERNELS=mul_mat_q,mul_mat_vec_q
    export HETGPU_BITNET_GPU_KERNELS=attention,attn,flash,softmax,rope,kq,qk,qkv,query,key,value,kv_cache
    export HETGPU_BITNET_ROUTE_LOG="$proof_dir/routes.jsonl"
    export HETGPU_XRT_EXECUTION_LOG="$proof_dir/xrt.jsonl"
    unset HETGPU_CXL_TMATMUL HETGPU_TMATMUL_CXL HETGPU_TMATMUL_MATRIX_STAGE HETGPU_TMATMUL_IO_STAGE
    rm -f "$proof_dir/routes.jsonl" "$proof_dir/xrt.jsonl"

    nvidia-smi -L >"$proof_dir/nvidia-smi-before.txt" 2>&1
    nvidia-smi --query-gpu=name --format=csv,noheader >"$proof_dir/gpu-name.txt" 2>&1
    xbutil examine -d 0000:64:00.1 -r platform >"$proof_dir/xbutil-platform-before.txt" 2>&1
    prompt='[BOS]<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>Reply with the single word OK.<|im_end|><|im_assistant|>assistant<|im_middle|>'
    printf '%s\n' "$prompt" >"$proof_dir/prompt.txt"
    printf '%q ' "$runner" -m "$model" --seed 42 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --no-display-prompt --simple-io --no-warmup -c 512 -n "$predict" -ngl 99 -p "$prompt" >"$proof_dir/command.txt"
    printf '\n' >>"$proof_dir/command.txt"

    set +e
    "$runner" -m "$model" \
        --seed 42 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 \
        --no-display-prompt --simple-io --no-warmup -c 512 -n "$predict" -ngl 99 \
        -p "$prompt" >"$proof_dir/stdout.txt" 2>"$proof_dir/stderr.log"
    exit_code=$?
    set -e
    printf '%s\n' "$exit_code" >"$proof_dir/exit-code.txt"

    if [[ $exit_code -eq 0 && -s $proof_dir/stdout.txt ]]; then
        set +e
        "$tokenizer" -m "$model" --ids --no-bos --log-disable \
            -f "$proof_dir/stdout.txt" >"$proof_dir/token-ids.json" 2>"$proof_dir/tokenizer-stderr.log"
        tokenize_exit=$?
        set -e
        [[ $tokenize_exit -eq 0 ]] || exit_code=$tokenize_exit
    else
        printf '[]\n' >"$proof_dir/token-ids.json"
    fi

    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv >"$proof_dir/nvidia-smi-after.csv" 2>&1
    xbutil examine -d 0000:64:00.1 -r dynamic-regions -r error -r firewall -r thermal >"$proof_dir/xbutil-after.txt" 2>&1
    sha256sum "$model" | awk '{print $1}' >"$proof_dir/model.sha256"
    sha256sum "$xclbin" | awk '{print $1}' >"$proof_dir/xclbin.sha256"
    sha256sum "$libnvcuda" | awk '{print $1}' >"$proof_dir/libnvcuda.sha256"
    sha256sum "$runner" | awk '{print $1}' >"$proof_dir/runner.sha256"

    PROOF_DIR="$proof_dir" RUN_EXIT_CODE="$exit_code" BENCHMARK_MODE="$benchmark" python3 - <<'PY'
import json
import os
import re
from pathlib import Path

proof = Path(os.environ["PROOF_DIR"])
stderr = (proof / "stderr.log").read_text(errors="replace")
health = (proof / "xbutil-after.txt").read_text(errors="replace")
try:
    token_ids = json.loads((proof / "token-ids.json").read_text())
except (OSError, json.JSONDecodeError):
    token_ids = []
if not isinstance(token_ids, list):
    token_ids = []

def rate(label):
    pattern = re.compile(
        rf"{label}\s+time\s*=\s*[0-9.]+\s*ms\s*/\s*[0-9]+\s*(?:tokens|runs)?\s*\(\s*([0-9.]+)\s+tokens per second\s*\)",
        re.IGNORECASE,
    )
    matches = pattern.findall(stderr)
    return float(matches[-1]) if matches else 0.0

temperature_matches = re.findall(r"^\s*FPGA\s*:\s*([0-9.]+)\s*C", health, re.MULTILINE | re.IGNORECASE)
fatal_errors = [line.strip() for line in health.splitlines() if re.search(r"\bfatal\b", line, re.IGNORECASE)]
summary = {
    "model_sha256": (proof / "model.sha256").read_text().strip(),
    "xclbin_sha256": (proof / "xclbin.sha256").read_text().strip(),
    "libnvcuda_sha256": (proof / "libnvcuda.sha256").read_text().strip(),
    "runner_sha256": (proof / "runner.sha256").read_text().strip(),
    "exit_code": int(os.environ["RUN_EXIT_CODE"]),
    "generated_token_ids": token_ids,
    "generated_text": (proof / "stdout.txt").read_text(errors="replace"),
    "prompt_tokens_per_second": rate("prompt eval"),
    "generation_tokens_per_second": rate("eval"),
    "gpu_name": (proof / "gpu-name.txt").read_text(errors="replace").splitlines()[0].strip(),
    "fpga_bdf": "0000:64:00.1",
    "firewall_status": "GOOD" if "Level 0 : 0x0 (GOOD)" in health else "BAD",
    "fatal_errors": fatal_errors,
    "fpga_temperature_c": float(temperature_matches[-1]) if temperature_matches else 999.0,
    "benchmark_mode": os.environ["BENCHMARK_MODE"] == "1",
}
(proof / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

    python3 /work/zluda/tests/validate_au250_hybrid_proof.py "$proof_dir" >"$proof_dir/validation.json"
    cat "$proof_dir/validation.json"
    exit 0
fi

compare_max=${HETGPU_XRT_COMPARE_MAX_LAUNCHES:-0}
predict=${N_PREDICT:-1}
benchmark=${HETGPU_AU250_BENCHMARK:-0}
[[ $compare_max =~ ^[0-9]+$ ]] || { echo "HETGPU_XRT_COMPARE_MAX_LAUNCHES must be a nonnegative integer" >&2; exit 2; }
[[ $predict =~ ^[1-9][0-9]*$ ]] || { echo "N_PREDICT must be positive" >&2; exit 2; }
case "$benchmark" in 0|1) ;; *) echo "HETGPU_AU250_BENCHMARK must be 0 or 1" >&2; exit 2;; esac

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
proof_rel=".proof/kimi-au250-${timestamp}"
proof_dir="$repo_root/$proof_rel"
install -d "$proof_dir" "$repo_root/target"
printf '%s\n' "$proof_rel" >"$repo_root/target/au250-last-proof-path"

"$repo_root/tools/build_au250_kimi_runtime.sh"
"$repo_root/tools/au250_hybrid_run.sh" bash \
    /work/tools/run_kimi_k26_iq1s_au250_hybrid.sh \
    --inside "/work/$proof_rel" "$compare_max" "$predict" "$benchmark"
python3 "$repo_root/zluda/tests/validate_au250_hybrid_proof.py" "$proof_dir"
echo "Kimi AU250 hybrid proof: $proof_rel"
