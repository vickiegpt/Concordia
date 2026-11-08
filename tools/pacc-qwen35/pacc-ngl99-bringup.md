# Qwen3.5-35B-A3B full-layer PACC bring-up

## Scope and current gate

Target: Qwen3.5-35B-A3B Q4_K_M, `ngl=99`, one PACC0, batch=256,
ubatch=1, flash attention disabled. This is full-layer compute offload with
host-backed weights staged through the owned 2 MiB payload aperture, not a
claim that all model weights reside in PACC RAM. Host-side quantization and
existing pointwise shims remain. GDN, softmax and top-k run on the PACC core;
quantized integer matrix dots use XM. No host matrix fallback is enabled.

**End-to-end gate: FAILED numerical validation.** The clean-release two-token
run for raw prompt `1+1=` completed but produced `[17, 1046]` (`2 gr`), versus
the CPU reference `[17, 220]` (`2 `). All logits were finite. Prefill took
1187607.727 ms and the following decode took 265996.446 ms; these are failed-run
diagnostic timings, not a valid throughput result. `aggregate_output_tps` is null.
The phase recorder observed 8720 completed preloaded submissions and no incomplete
ones, but those records do not cover all ELF kernel execution. Do not infer an
end-to-end bottleneck percentage from the preloaded phase sum.

The first token matching does not validate recurrent state or sustained decode.
No host reboot or PACC1/2/3 programming was used for this result.

## First numerical defect found

The four-layer diagnostic prefix preserves 76 original tensors byte-for-byte.
Its CPU layer-0 and layer-3 outputs for the first input token also exactly match
the corresponding full-model CPU outputs. It is not a replacement benchmark.

With the old runtime, the first token's `state_predelta-1` should be all zero
but contains 2048 nonzero trailing floats (max absolute value 0.084170967).
The host data-copy shim treats `k_get_rows_float_vec`'s `ne00v` as a scalar
float count, although GGML passes a count of 16-byte vectors. It copies only
one quarter of an FP32 recurrent state, leaving the rest stale. This defect
affects both initial scratch-buffer reuse and state restoration on later tokens.

The fix multiplies the vector width by four while preserving scalar-element
strides, shares that width calculation with staged buffer metadata, and rejects
unsupported types, negative indices, and out-of-range source rows. No-op gather
can no longer return success. This is a host data-copy repair, not a host matmul
fallback or an XM speedup.

`test-getrows-vector-abi.py` reproduces the old failure, then passes scalar,
vector, 524288-float state, guard preservation, negative-index and short-buffer
checks with the repaired library. Full-model post-fix generation is not yet
qualified. The single-token layer-31 residual comparison also amplifies small
upstream differences; that threshold crossing alone is not proof of another bug.

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

For direct generation without the server's two-token sequence-removal probe,
use `run-ngl99-direct.sh UNIQUE_PREFIX 0x20035`. It still computes all prompt and
decode steps, checks every output logit is finite, and compares generated text
and token IDs with an actual CPU run. Its default is two output tokens.

## Four-device readiness audit

Read-only inspection found all four `/dev/paccN` and helper nodes present and
bound. However, all four helpers report the same base, `0x2075d000000`, with one
4 MiB allocation. The current PACC1/2/3 beacon fields are zero; this does not
establish their firmware identity, readiness, or numerical correctness.

Do not enable four simultaneous clients with the current single-slot settings:

- Only PACC0 has the qualified v53 image and numerical/completion evidence.
- The current builder and recovery loader are explicitly PACC0-only.
- Four equal payload slots would be 512 KiB each, below some current Q4/Q5
  launches. Tile bounds must use actual per-device capacity, not a fixed 2 MiB.
- The enabled early-devmem startup marker writes completion records for all
  four slots. A new card boot must not overwrite another card's live completion.
- Diagnostic rings use slots 8..11, while RMS debug records start in slot 9.
  That layout must be separated before trusting four-card profiling.
- A single numeric `HETGPU_SIFIVE_VISIBLE_DEVICES` value means a device count;
  explicit physical-device lists require the list syntax. Verify physical
  affinity rather than assuming `1` selects PACC1.

For aggregate throughput, qualify independent requests on four isolated device
queues first, then evaluate expert or tensor parallelism. No four-card TPS or
linear speedup has been measured. This audit did not reset or program PACC1/2/3.

## Remaining boundaries

- General CUDA warp reductions are not universally implemented by these named
  kernel specializations. Unsupported combinations return errors.
- Multi-token MoE CUDA kernels and large-batch throughput are not qualified by
  the ubatch=1 test. PACC0 is not four PACC devices.
- Board VLEN=1024 code is not executed on the host's VLEN=128 CPU.
- Do not reuse the owned shared-DDR address after a host reboot.
- The default installed runtime is unchanged; candidate libraries are separate.
