# GLM-5.2 RTX 5090 Attention and CPU FFN Tuning

## Stable Configuration

Use the dedicated split mode:

```sh
SPLIT_ATTN_GPU_FFN_CPU=1 tools/run_glm52_pacc_gpu_server.sh
```

This mode assigns the attention layers to CUDA0, overrides every `ffn_*`
tensor to the CPU backend, disables PACC boot and CPU/PACC hooks, uses 16 CPU
threads, keeps model mmap enabled, and disables the server prompt-state RAM
cache. Flash Attention remains disabled because the current CUDA shim does not
support that graph; the regular attention kernels stay on the CUDA backend.

Do not combine this 207 GiB CPU tensor placement with `--no-mmap` on the
251 GiB Lanxin host. The anonymous CPU copy leaves too little headroom for the
RISC-V fake-CUDA allocation path and can push RM into long direct reclaim.

## 2026-08-02 Measurements

All rows use RTX 5090 attention, CPU FFN, `ctx=1024`, `batch=128`,
`ubatch=32`, deterministic temperature zero, and real server timing fields.

| Threads | Generation length | Generation TPS | Prompt TPS | Notes |
| ---: | ---: | ---: | ---: | --- |
| 32 | 32 | 0.390 | 2.056 | First long request after load |
| 16 | 32 | 0.589 | 1.667 | One four-token warmup request |
| 12 | 32 | 0.578 | 1.352 | One four-token warmup request |

The prior eight-token 32-thread reference was 0.478 TPS. The 16-thread result
is 23% faster than that reference and is the best stable measured setting.
The 12-thread result shows that reducing worker count further starts to lose
RVV throughput.

The 32-thread profile averaged about 11 active CPU cores, 1.10 instructions per
cycle, and zero major faults. Page-fault address sampling placed nearly all
minor faults in GGUF model shards rather than NVIDIA device mappings. This is
consistent with MoE expert selection touching different file-backed weight
pages on each token.

Raw logs are stored on Lanxin under:

```text
/mnt/probe_nvme0n1p4/models/.lanxin-build/logs/glm52-attn5090-cpuffn-opt-t32-20260802
/mnt/probe_nvme0n1p4/models/.lanxin-build/logs/glm52-attn5090-cpuffn-opt-t16-20260802b
/mnt/probe_nvme0n1p4/models/.lanxin-build/logs/glm52-attn5090-cpuffn-opt-t12-20260802
```
