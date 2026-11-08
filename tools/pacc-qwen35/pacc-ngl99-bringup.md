# Qwen3.5-35B-A3B full-layer PACC bring-up

## Scope and current gate

Target: Qwen3.5-35B-A3B Q4_K_M, `ngl=99`, one PACC0, batch=256,
ubatch=1, flash attention disabled. This is full-layer compute offload with
host-backed weights staged through the owned 2 MiB payload aperture, not a
claim that all model weights reside in PACC RAM. Host-side quantization and
existing pointwise shims remain. GDN, softmax and top-k run on the PACC core;
quantized integer matrix dots use XM. No host matrix fallback is enabled.

**End-to-end gate: pending.** Do not quote a full-offload TPS until the
request JSON confirms the expected text and the actual generated-token count.
The CPU reference produces `2` for raw prompt `1+1=` (4 input tokens).
One output token is only a correctness smoke, not sustained decode throughput.

## Repairs

- Correct concat argument counts and sizes for the current GGML CUDA ABI.
- Add device-side gated delta, top-k and masked attention softmax paths.
- Preserve exact lazily registered CUDA function identity before compilation.
- Tile FP32 MMVF within actual shared-DDR capacity and preserve FP32 inputs.
- Handle multihead attention strides and broadcast without host matmul.
- Add Q4_K, Q5_K and Q8_0 XM integer-dot tiles, expert selection, gate fusion,
  bias application, and small-K row grouping.
- Preserve untouched KV rows and size scatter output from actual indices.
- Keep preloaded and ELF completion ownership separate; retain completed ELF
  results so a cleared status record cannot cause duplicate execution.
- Replace unbounded retained ELF mappings with a 64-entry, 32 MiB total-budget
  cache. Lookup checks symbol and all ELF bytes, not only a hash. Restore writable
  PT_LOAD data and BSS from the relocated initial image before reuse. Exhaustion
  fails explicitly rather than growing memory or selecting CPU fallback.
- Add persistent phase, mapping-count and fault diagnostics in the reserved,
  disabled-telemetry area. Fault PC/address are meaningful only if signal != 0.

## Loader failure evidence

v52, using the old retained-mapping policy plus diagnostics, failed the repeated
cell probe at zero-based iteration 4085. The last stage was 0x5104, before ELF
execution, with 4086 retained mappings totaling 0x3fd8000 bytes. A later tiny
kernel could not complete either. This isolates a loader/resource-lifetime
failure; it is not evidence of a new XM opcode mismatch.

v53 passed 6000 hardware submissions, including a mutable static counter that
must restart at one for every fresh invocation. With the identity probe included,
only two ELF mappings (32 KiB) remained. Host cache tests additionally cover
10000 hits, writable/BSS restoration, different ELF bytes, and bounded exhaustion.

Firmware: `v53_bounded_elf`, marker `0x20035`.
SHA256: `b6e1b133a71c34820c55250a6cab341a0de9ec4c784980791606061e4adf9129`.
No host reboot. Only PACC0 was rebound through the vendor driver. The canonical
vendor image was restored and its SHA checked after every load.

## Reproduction

These paths and the loader's boot/allocation checks are specific to lanxin2.
Deploy the tools into `/srv/lanxin/qwen35-a3b/bin` and run with PACC0 idle:

```sh
CONCORDIA_RELEASE=/srv/lanxin/qwen35-a3b/runtime-ngl99-release \
  bash /srv/lanxin/qwen35-a3b/bin/run-ngl99-validation.sh UNIQUE_PREFIX 0x20035
```

The runner first verifies firmware identity and numerical results, then tests
the added CUDA ABIs, then makes the full-offload request. Failure stops the run.
It does not load firmware or reboot anything. The matching firmware must already
be qualified. Tests using actual PTX require the existing llama.cpp PTX cache.

Release builds use the LLVM submodule at the recorded gitlink, LLVM 20, and the
evaluation environment's Cargo.lock. The lock is not normally tracked by this
repository; preserve its hash with the evidence when reproducing a build.

## Remaining boundaries

- General CUDA warp reductions are not universally implemented by these named
  kernel specializations. Unsupported combinations return errors.
- Multi-token MoE CUDA kernels and large-batch throughput are not qualified by
  the ubatch=1 test. PACC0 is not four PACC devices.
- Board VLEN=1024 code is not executed on the host's VLEN=128 CPU.
- Do not reuse the owned shared-DDR address after a host reboot.
- The default installed runtime is unchanged; candidate libraries are separate.
