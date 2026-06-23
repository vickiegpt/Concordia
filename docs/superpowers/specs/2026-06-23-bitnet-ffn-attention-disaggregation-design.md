# BitNet FFN and Attention Disaggregation

**Date:** 2026-06-23
**Status:** Approved for spec review
**Scope:** Add a hetGPU-side runtime routing layer that sends BitNet FFN
ternary matmul work to the CXL tmatmul path while keeping attention and other
GPU-friendly kernels on the native NVIDIA path.

## Goals

1. Route BitNet FFN, MLP, MoE, and ternary linear matmul kernels to the
   existing CXL tmatmul launch path when explicitly enabled.
2. Keep attention, RoPE, KV-cache, softmax, and QKV kernels on the GPU path.
3. Support both lightweight environment marker lists and an optional JSON route
   manifest for reproducible experiments.
4. Emit machine-readable per-launch route logs for correctness proof and
   benchmarking.
5. Preserve the current fallback behavior by default, with a strict mode for
   debugging ambiguous or unsafe offload candidates.

## Non-Goals

- No BitNet or llama.cpp source patch in this phase.
- No rebasing `/home/eabban/BitNet` onto a newer llama.cpp.
- No new CXL tmatmul hardware ABI.
- No arbitrary FFN graph partitioning inside ggml in this phase.
- No offload of attention, RoPE, softmax, KV-cache, or QKV kernels to CXL.
- No dtype conversion from full-precision GPU tensors into ternary CXL payloads
  beyond the data layouts already supported by the existing tmatmul fallback.

## Existing Context

hetGPU already has the pieces needed for a first runtime split:

- `zluda/src/impl/function.rs` intercepts CUDA kernel launches and has a
  tmatmul named fallback path.
- `zluda/src/impl/function.rs` already recognizes
  `HETGPU_BITNET_DISAGGREGATE`, `HETGPU_BITNET_FFN_CXL`, and
  `HETGPU_TMATMUL_BITNET_DISAGGREGATE`.
- `zluda/src/impl/function.rs` already contains basic FFN and attention kernel
  marker checks.
- `zluda/src/impl/cxl_tmatmul.rs` owns CXL tmatmul staging, assembly encoding,
  allocation validation, and `RUN_CSR_ONLY` submission.
- `tools/run_kimi_k26_iq1m_bitnet.sh` is the local Kimi K2.6 BitNet runner
  wrapper for end-to-end smoke runs.

The sibling BitNet checkout exposes ternary matmul through ggml-level helpers
such as `ggml_bitnet_can_mul_mat`, but that layer does not reliably label
whether a matmul belongs to FFN or attention. The first implementation should
therefore classify launches in hetGPU using kernel names, explicit route
markers, and optional manifest entries.

## Architecture

Add a small routing module inside the existing Intel/tmatmul launch boundary:

```text
BitNet llama-cli
  -> CUDA Driver API launch
  -> hetGPU LD_PRELOAD hook
  -> BitNet route classifier
       -> CXL tmatmul for FFN ternary matmul candidates
       -> native GPU path for attention and explicit GPU routes
       -> fallback/native path for unknown kernels
```

The routing module should be local to `zluda/src/impl/function.rs` at first, or
split into `zluda/src/impl/bitnet_disagg.rs` if the implementation grows beyond
simple helpers and tests. It must reuse `execute_tmatmul_hardware_matmul_fallback`
and `submit_cxl_hardware_matmul_fallback` instead of creating a second CXL
submit path.

## Route Decision Model

Each launch is classified into one of four route decisions:

| Decision | Meaning |
| --- | --- |
| `CxlTmatmul` | Try CXL tmatmul staging and submit for an FFN ternary matmul. |
| `GpuNative` | Leave the launch on the native GPU path. |
| `Fallback` | Do not offload; allow existing hetGPU fallback behavior. |
| `Reject` | In strict mode, fail or skip the unsafe offload instead of masking it. |

Classification order:

1. If BitNet disaggregation is disabled, return `Fallback`.
2. Normalize the kernel name to lowercase.
3. Apply explicit GPU deny markers from `HETGPU_BITNET_GPU_KERNELS`.
4. Apply manifest entries, if `HETGPU_BITNET_ROUTE_MANIFEST` is set.
5. Apply explicit CXL allow markers from `HETGPU_BITNET_CXL_KERNELS`.
6. Apply built-in attention/GPU markers.
7. Apply built-in FFN/CXL markers.
8. Return `Fallback` or `Reject` depending on strict mode and candidate status.

Attention markers include:

```text
attention, attn, flash, softmax, soft_max, rope, kq, qk, qkv,
query, key, value, kv_cache
```

FFN/CXL markers include:

```text
ffn, feed_forward, mlp, gate_proj, up_proj, down_proj, expert, moe,
mul_mat_q, mul_mat_vec_q, mul_mat_f
```

Explicit GPU markers always override CXL markers. This prevents accidental
offload of attention matmuls whose names also include generic `mul_mat`.

## Manifest Format

The optional manifest is a small JSON file used for reproducible runs:

```json
{
  "version": 1,
  "default": "fallback",
  "routes": [
    {
      "match": "ffn_gate",
      "route": "cxl_tmatmul",
      "reason": "BitNet FFN gate projection"
    },
    {
      "match": "flash_attn",
      "route": "gpu",
      "reason": "attention remains on GPU"
    }
  ]
}
```

Fields:

- `version` must be `1`.
- `default` may be `fallback`, `gpu`, or `reject`.
- `routes[].match` is a lowercase substring match after kernel-name
  normalization.
- `routes[].route` may be `cxl_tmatmul`, `gpu`, `fallback`, or `reject`.
- `routes[].reason` is optional and is copied into route logs.

Malformed manifests should not crash the process in default mode. They should
log one error, disable manifest routing, and continue with built-in/env
classification. In strict mode, a malformed manifest returns `Reject`.

## Environment

The feature is opt-in:

```text
HETGPU_BITNET_DISAGGREGATE=1
HETGPU_BITNET_FFN_CXL=1
HETGPU_TMATMUL_BITNET_DISAGGREGATE=1
```

Any of these enables the router.

Additional controls:

```text
HETGPU_BITNET_ROUTE_MANIFEST=/path/bitnet_routes.json
HETGPU_BITNET_CXL_KERNELS=ffn_gate,mlp_up,down_proj
HETGPU_BITNET_GPU_KERNELS=attention,rope,softmax
HETGPU_BITNET_DISAGG_STRICT=1
HETGPU_BITNET_ROUTE_LOG=/tmp/bitnet_routes.jsonl
```

CXL submission still requires the existing tmatmul controls:

```text
HETGPU_TMATMUL_HARDWARE_MATMUL=1
HETGPU_CXL_TMATMUL=1
HETGPU_CXL_TMATMUL_DEV=/dev/cxl_tmatmul3b000
HETGPU_CXL_TMATMUL_DAX=/dev/dax0.0
```

If `HETGPU_CXL_TMATMUL` is absent, `CxlTmatmul` route decisions should still
emit route logs and use the existing emulator/named fallback behavior rather
than silently pretending that real CXL hardware was used.

## Route Logging

When `HETGPU_BITNET_ROUTE_LOG` is set, each classified launch appends one JSONL
record:

```json
{
  "kernel": "_z13mul_mat_vec_q_bitnet_ffn_gate",
  "route": "cxl_tmatmul",
  "source": "builtin_ffn_marker",
  "matched": "ffn",
  "strict": false,
  "cxl_enabled": true,
  "hardware_matmul_enabled": true,
  "message": "FFN ternary matmul candidate"
}
```

The log should be best-effort. File-open or write failures must not crash the
process in default mode. In strict mode, route-log failures may produce
`Reject` only when the launch would otherwise attempt CXL offload.

## Error Handling

Default mode is conservative:

- Unknown kernels stay on the existing path.
- Explicit GPU routes stay on the native GPU path.
- CXL submit failures log the failure and return to the existing fallback
  behavior.
- Missing or malformed manifest disables manifest routing after one diagnostic.

Strict mode is for debugging:

- A malformed manifest rejects candidate offloads.
- A kernel selected for `cxl_tmatmul` but lacking a supported tmatmul layout is
  rejected instead of silently falling through.
- A route-log failure can reject a candidate offload if the caller requested
  logging as part of the evidence trail.

Strict mode must not reject kernels explicitly routed to GPU.

## Testing

Pure Rust tests should cover:

1. Env gate parsing for all three enable variables.
2. GPU marker precedence over CXL marker matches.
3. Built-in attention markers stay on GPU.
4. Built-in FFN markers select CXL tmatmul.
5. Manifest route selection and default fallback.
6. Malformed manifest behavior in default and strict mode.
7. Route JSONL formatting for a synthetic kernel.
8. Existing `mul_mat_q` parameter layout tests remain passing.

Runtime smoke checks on an NVIDIA host:

```bash
HETGPU_BITNET_DISAGGREGATE=1 \
HETGPU_TMATMUL_HARDWARE_MATMUL=1 \
HETGPU_BITNET_ROUTE_LOG=/tmp/bitnet_routes.jsonl \
LD_PRELOAD=target/debug/libnvcuda.so \
tools/run_kimi_k26_iq1m_bitnet.sh "用一句中文说明你已经启动。"
```

Hardware CXL smoke adds:

```bash
HETGPU_CXL_TMATMUL=1 \
HETGPU_CXL_TMATMUL_DEV=/dev/cxl_tmatmul3b000 \
HETGPU_CXL_TMATMUL_DAX=/dev/dax0.0
```

The expected evidence is a route log with attention-like kernels routed to GPU
and FFN-like ternary matmul kernels routed to `cxl_tmatmul` candidates. Hardware
success additionally requires CXL staging and `RUN_CSR_ONLY` status logs.

## Acceptance Criteria

- The router is disabled by default.
- With disaggregation enabled, attention markers route to GPU.
- With disaggregation enabled, FFN ternary matmul markers route to CXL tmatmul
  candidates.
- Explicit GPU marker lists override CXL marker lists and built-in FFN markers.
- An optional manifest can reproduce a routing policy without recompilation.
- Route logs are valid JSONL and include enough source/match data to audit the
  split.
- Existing tmatmul hardware matmul tests continue to pass.
- No BitNet or llama.cpp source files are modified in this phase.
