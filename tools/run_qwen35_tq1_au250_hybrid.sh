#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
validator="${repo_root}/zluda/tests/validate_qwen35_tq1_au250_proof.py"

if [[ "${1:-}" == "--inside" ]]; then
    [[ $# -eq 3 ]] || { echo "usage: $0 --inside PROOF_DIR XCLBIN" >&2; exit 2; }
    proof_dir=$2
    xclbin=$3
    case "${proof_dir}" in
        /work/.proof/qwen35-tq1-*) ;;
        *) echo "refusing unexpected proof directory ${proof_dir}" >&2; exit 2 ;;
    esac

    model=/models/qwen/Qwen3.5-397B-A17B-UD-TQ1_0.gguf
    manifest=/qwen-build/manifest.json
    llama_server=/qwen-build/llama-build/bin/llama-server
    libnvcuda=/qwen-build/hetgpu-target/release/libnvcuda.so
    oracle=/qwen-build/tq1_upstream_reference
    evaluator=/work/tools/qwen35_au250_eval.py
    prompt_seed=/work/zluda/evaluation/fixtures/qwen35_prompt_seed.txt
    model_size=94155830880
    model_sha256=0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568
    llama_revision=925e1179947ea0c0ebfb0032df18af3a729822be
    fpga_bdf=0000:64:00.1
    cu_config='{"version":1,"cus":[{"ip_name":"ternip_big:ternip_big_1","memory_group":0,"lanes":9},{"ip_name":"ternip_big:ternip_big_2","memory_group":3,"lanes":9},{"ip_name":"ternip_big:ternip_big_3","memory_group":2,"lanes":9},{"ip_name":"ternip_small:ternip_small_1","memory_group":1,"lanes":6}]}'

    for required in "${model}" "${manifest}" "${llama_server}" "${libnvcuda}" "${oracle}" "${evaluator}" "${prompt_seed}" "${xclbin}" "${validator}"; do
        [[ -f "${required}" ]] || { echo "missing Qwen evaluation input ${required}" >&2; exit 1; }
    done
    install -d "${proof_dir}"

    actual_size="$(stat -c %s "${model}")"
    [[ "${actual_size}" == "${model_size}" ]] || { echo "Qwen model size mismatch" >&2; exit 1; }
    actual_model_sha="$(sha256sum "${model}" | awk '{print $1}')"
    [[ "${actual_model_sha}" == "${model_sha256}" ]] || { echo "Qwen model SHA-256 mismatch" >&2; exit 1; }
    MODEL="${model}" MODEL_SHA256="${model_sha256}" OUTPUT="${proof_dir}/model-verification.json" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["MODEL"])
stat = path.stat()
record = {
    "path": str(path),
    "size": stat.st_size,
    "device": stat.st_dev,
    "inode": stat.st_ino,
    "mtime_ns": stat.st_mtime_ns,
    "ctime_ns": stat.st_ctime_ns,
    "sha256": os.environ["MODEL_SHA256"],
}
Path(os.environ["OUTPUT"]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

    LLAMA_SERVER="${llama_server}" LIBNVCUDA="${libnvcuda}" ORACLE="${oracle}" \
    MANIFEST="${manifest}" LLAMA_REVISION="${llama_revision}" python3 - <<'PY'
import hashlib
import json
import os

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

manifest = json.load(open(os.environ["MANIFEST"], encoding="utf-8"))
if manifest.get("schema_version") != 1 or manifest.get("llama_revision") != os.environ["LLAMA_REVISION"]:
    raise SystemExit("build manifest revision/schema mismatch")
for name, variable in (("llama_server", "LLAMA_SERVER"), ("libnvcuda", "LIBNVCUDA"), ("tq1_upstream_reference", "ORACLE")):
    artifact = manifest.get("artifacts", {}).get(name, {})
    if artifact.get("path") != os.environ[variable] or artifact.get("sha256") != digest(os.environ[variable]):
        raise SystemExit(f"build manifest artifact mismatch: {name}")
PY
    server_sha256="$(sha256sum "${llama_server}" | awk '{print $1}')"
    libnvcuda_sha256="$(sha256sum "${libnvcuda}" | awk '{print $1}')"
    xclbin_sha256="$(sha256sum "${xclbin}" | awk '{print $1}')"

    xclbin_info="$(xclbinutil --info --input "${xclbin}" 2>&1)"
    printf '%s\n' "${xclbin_info}" > "${proof_dir}/xclbin-info.txt"
    for cu in ternip_big_1 ternip_big_2 ternip_big_3 ternip_small_1; do
        grep -Fq "Instance:        ${cu}" <<<"${xclbin_info}" || {
            echo "xclbin is missing expected compute unit ${cu}" >&2
            exit 1
        }
    done

    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits \
        > "${proof_dir}/nvidia-memory-preflight.csv"
    MODEL_SIZE="${model_size}" python3 - "${proof_dir}/nvidia-memory-preflight.csv" <<'PY'
import os
import pathlib
import sys

free_mib = 0
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    fields = [field.strip() for field in line.split(",")]
    free_mib += int(fields[-1])
required = int(os.environ["MODEL_SIZE"]) + 2 * 1024**3
if free_mib * 1024**2 < required:
    raise SystemExit(f"insufficient aggregate free GPU memory: {free_mib} MiB free, {required} bytes required")
PY

    xbutil examine -d "${fpga_bdf}" -r dynamic-regions -r error -r firewall -r thermal \
        > "${proof_dir}/xbutil-preflight.txt" 2>&1
    grep -Fq 'Level 0 : 0x0 (GOOD)' "${proof_dir}/xbutil-preflight.txt" || {
        echo "AU250 firewall is not GOOD during preflight" >&2
        exit 1
    }
    if grep -Eiq '(^|[^[:alpha:]])fatal([^[:alpha:]]|$)' "${proof_dir}/xbutil-preflight.txt"; then
        echo "AU250 reported a fatal preflight error" >&2
        exit 1
    fi

    {
        printf 'model_sha256=%s\n' "${actual_model_sha}"
        printf 'llama_server_sha256=%s\n' "${server_sha256}"
        printf 'libnvcuda_sha256=%s\n' "${libnvcuda_sha256}"
        printf 'xclbin_sha256=%s\n' "${xclbin_sha256}"
        printf 'llama_revision=%s\n' "${llama_revision}"
        printf 'threads=%s\n' "${QWEN35_THREADS:-$(nproc)}"
    } > "${proof_dir}/artifact-hashes.txt"
    git -C /work status --porcelain=v1 --untracked-files=normal --ignore-submodules=all \
        > "${proof_dir}/repository-status.txt"
    git -C /work diff --binary --ignore-submodules=all HEAD | sha256sum | awk '{print $1}' \
        > "${proof_dir}/repository-diff.sha256"
    nvidia-smi -L > "${proof_dir}/nvidia-gpus.txt"

    export LD_LIBRARY_PATH="/qwen-build/llama-build/bin:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
    export HETGPU_XRT_XCLBIN="${xclbin}"
    export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
    export HETGPU_XRT_TIMEOUT_MS=10000
    export HETGPU_XRT_CLOCK_HZ=300000000
    export HETGPU_XRT_CU_CONFIG="${cu_config}"
    threads="${QWEN35_THREADS:-$(nproc)}"

    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        > "${proof_dir}/cuda-compute-apps-before.csv"
    (
        export HETGPU_QWEN_TQ1_XRT=0
        export HETGPU_QWEN_TQ1_STRICT=0
        unset HETGPU_TQ1_EVIDENCE_LOG HETGPU_XRT_EXECUTION_LOG
        python3 "${evaluator}" \
            --mode cuda --server "${llama_server}" --server-preload "${libnvcuda}" \
            --model "${model}" --prompt-seed "${prompt_seed}" \
            --model-verification "${proof_dir}/model-verification.json" \
            --proof-dir "${proof_dir}/cuda-mode" --port 18080 --threads "${threads}" \
            --model-size "${model_size}" --model-sha256 "${model_sha256}" \
            --llama-revision "${llama_revision}" --binary-sha256 "${server_sha256}" \
            --fpga-bdf "${fpga_bdf}"
    )
    cp "${proof_dir}/cuda-mode/cuda.json" "${proof_dir}/cuda.json"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        > "${proof_dir}/cuda-compute-apps-after.csv"

    HETGPU_TQ1_EVALUATION_LIBRARY="${libnvcuda}" \
    HETGPU_XRT_XCLBIN="${xclbin}" \
        bash /work/zluda/tests/run_au250_xrt_tq1.sh "${oracle}" "${proof_dir}/numerical"

    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        > "${proof_dir}/hybrid-compute-apps-before.csv"
    (
        export HETGPU_QWEN_TQ1_XRT=1
        export HETGPU_QWEN_TQ1_STRICT=1
        export HETGPU_TQ1_EVIDENCE_LOG="${proof_dir}/hybrid-mode/tq1-evidence.jsonl"
        export HETGPU_XRT_EXECUTION_LOG="${proof_dir}/hybrid-mode/xrt-execution.jsonl"
        python3 "${evaluator}" \
            --mode hybrid --server "${llama_server}" --server-preload "${libnvcuda}" \
            --model "${model}" --prompt-seed "${prompt_seed}" \
            --model-verification "${proof_dir}/model-verification.json" \
            --proof-dir "${proof_dir}/hybrid-mode" --port 18081 --threads "${threads}" \
            --model-size "${model_size}" --model-sha256 "${model_sha256}" \
            --llama-revision "${llama_revision}" --binary-sha256 "${server_sha256}" \
            --fpga-bdf "${fpga_bdf}" --require-routing-evidence
    )
    cp "${proof_dir}/hybrid-mode/hybrid.json" "${proof_dir}/hybrid.json"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        > "${proof_dir}/hybrid-compute-apps-after.csv"

    python3 "${validator}" "${proof_dir}" | tee "${proof_dir}/summary.json"
    exit 0
fi

[[ $# -eq 4 ]] || {
    echo "usage: $0 MODEL BUILD_MANIFEST XCLBIN OUTPUT_DIR" >&2
    exit 2
}
model=$1
manifest=$2
xclbin=$3
proof_dir=$4

for required in "${model}" "${manifest}" "${xclbin}"; do
    [[ -f "${required}" ]] || { echo "missing evaluation input ${required}" >&2; exit 1; }
done
case "$(realpath -m "${proof_dir}")" in
    "${repo_root}"/.proof/qwen35-tq1-*) ;;
    *) echo "output directory must be ${repo_root}/.proof/qwen35-tq1-*" >&2; exit 2 ;;
esac
[[ ! -e "${proof_dir}" ]] || { echo "output directory already exists: ${proof_dir}" >&2; exit 1; }
[[ "$(realpath "${model}")" == /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf ]] || {
    echo "runner requires the pinned model path" >&2
    exit 1
}
[[ "$(realpath "${manifest}")" == /root/qwen35-au250-build/manifest.json ]] || {
    echo "runner requires the pinned build manifest path" >&2
    exit 1
}
case "$(realpath "${xclbin}")" in
    /au250_xrt/*) ;;
    *) echo "xclbin must be mounted beneath /au250_xrt" >&2; exit 1 ;;
esac

proof_rel="$(realpath -m --relative-to="${repo_root}" "${proof_dir}")"
source /au250_xrt/env.sh >/dev/null
temperature="$(_au250_fpga_temp)"
[[ -z "${temperature}" || "${temperature}" -lt 85 ]] || {
    echo "AU250 temperature ${temperature}C exceeds 85C guard" >&2
    exit 1
}
cd "${repo_root}"
"${repo_root}/tools/au250_qwen35_run.sh" bash /work/tools/run_qwen35_tq1_au250_hybrid.sh \
    --inside "/work/${proof_rel}" "$(realpath "${xclbin}")"
python3 "${validator}" "${proof_dir}" | tee "${proof_dir}/summary.json"
