# PACC0 XM context and Qwen35 performance harness

This directory is a reproducible **experiment harness**, not a new production
default. `hetgpu_pacc_jobd.c` is the exact legacy jobd source used by the running
PACC0 firmware; keeping it here avoids overwriting an independently modified
SiFive runtime checkout. No firmware binaries, kernel images or model weights
are committed. The host runtime ABI must already provide
`hetgpu_sifive_submit_q6k_q8_1_tiled`, included in this branch's SiFive runtime.
The related Q8_1 layout, Qwen RoPE/get-rows ABI and RISC-V dp4a lowering fixes
are included without the original working tree's unrelated CXL routing edits.

## Verified fixes

1. The vendor guest DT advertised `xlxmmbase`, `xlxmm32a8i`,
   `xlxmm32a16f`, `xlxmm32a32f`, `xlxmm64t`. Its Linux 6.7.9+ kernel
   recognizes the corresponding `xsf*` names. `patch-xm-isa-alias.py` uses
   libfdt to change only these existing names in both ISA properties for four
   CPU nodes. It requires an input SHA and changes exactly 80 bytes. It never
   adds a feature, patches a kernel instruction or forces privileged CSR bits.
2. That DT-only change removes the first `sf.vtzero.t` SIGILL. Configuration
   and tile-state probes pass in BF16, F16, F32 and I8 modes.
3. `xsfmm_native_bf16.c` sets e32/w1 before the FP32 tile store; the old e16alt/w2
   store context produced incorrect results without SIGILL. The leaf is internal:
   callers supply valid buffers, 1 <= M,N <= 32 and positive even K <= 128.
4. `q6k-batch-xm.c` uses exact BF16 representations of quantized integers for
   K=16 XM dots, then applies Q6 subblock scales in integer arithmetic and the
   original floating scale order. It does not round weight*scale into BF16.
   The XM build returns errors; it does not select RVV or host matmul on failure.
5. `test-xm-pack-reference.c` is a **host-only test double** for packed layout
   tests. The firmware link command does not include it.
6. v46 adds RVV operand unpacking and exact scale reduction, aligned RVV copies
   only inside the owned payload aperture, and four persistent XM workers inside
   PACC0. This is not four PACC devices. Matrix products still execute on XM.
7. Routine dispatch console output is disabled in the performance image;
   errors and the shared-memory diagnostic ring remain enabled. Restore routine
   output with guest `HETGPU_PACC_JOBD_DISPATCH_LOG=1`. A bounded 2 ms hot-scan
   window returns to the normal idle policy; host busy polling is not enabled.
8. `test-xm-reject.c` injects invalid leaf results in a host-only unit test.
   NaN, infinity, fractional/out-of-range dots and a leaf error fail closed.

## Deployment constraints

These commands target the existing `/srv/lanxin/qwen35-a3b` evaluation tree on
`lanxin2`, not an arbitrary machine. Copy the source/tools into that tree's `bin`
directory. The build uses its existing vendor patcher under
`pacc-runtime-head/ext/pacc_runtime-sys/patch_lx500_pacc_inplace_jobd.sh`, original
vendor firmware backup, Clang 20 and libfdt. Test input dequantization uses the
existing llama.cpp `libggml-base.so`.

`build-xm-batch-jobd.sh` creates v46 and runs host layout, fault-injection and
copy boundary tests before linking. It refuses to overwrite an existing build.
The resulting image is specific to the pinned 4 MiB allocation at
`0x2075d000000`. **Do not reuse that address after a host reboot.** The loader
checks the exact boot ID, owned allocation, image SHA, original firmware SHA,
reserved firmware region and PACC0 clients before using the vendor bind path.
It restores the original canonical firmware file in `finally` and never
reboots the host or unloads the coherency helper. This is a temporary PACC0
deployment; it is not a persistent boot-firmware upgrade.

```sh
python3 bin/test-xm-diagnostics.py
bash bin/build-xm-batch-jobd.sh
sudo python3 bin/load-pacc0-v26-dynamic.py --version v46_xm_reduce
# Only after the dry-run checks and a PACC0 maintenance window:
sudo python3 bin/load-pacc0-v26-dynamic.py --version v46_xm_reduce --apply
```

For hardware tests, use `source bin/pacc-env.sh`, the verified owned DDR base and
size, bootstrap=0, and strict completion (ASYNC_NO_WAIT=0, SKIP_JOB_WAIT=0).
Bootstrap=0 skips an obsolete startup handshake, not job completion.

```sh
bash bin/run-xm-validation.sh xm-validation 0x2002e
bash bin/run-xm-validation.sh xm-worker-tails 0x2002e --hardware-workers
bash bin/run-xm-qwen-benchmark.sh xm-count16-p4 0x2002e 4 count16
bash bin/run-xm-qwen-benchmark.sh xm-count64-p8 0x2002e 8 count64
```

Capability events occur only on the first MMVF after guest boot, so later
captures need the first capture to interpret the startup probes. Build markers
are emitted on every MMVF. Probe stages, faults and 64-call results are at
0x5324..0x532f; format snapshots are at 0x5400..0x5447.

## Evidence boundaries

- Qwen configuration is `ngl=1`: CPU model blocks/preprocessing plus PACC0
  Q6_K output projection and RMSNorm. It is not all-PACC, four-PACC, GPU+PACC,
  or a 500 TPS saturation measurement.
- Native XM validation: four shapes up to 32x32x128, repeated 64 times, exact
  integer-valued reference and sentinel checks. Its elapsed time includes data
  preparation, reference calculation, scheduling and checks, not pure XM time.
- Packed Q6 tests: GGML dequantization reference, padded strides, finite output,
  output guards; includes complete 248320x1x2048 output projection.
- `compute` includes unpacking, BF16 staging, XM leaf calls and scale reduction.
  Do not interpret it as a device matrix-unit busy counter.
- Detailed 0x5500..0x5503 timings sum worker elapsed time and overlap across
  workers. Do not add those values to device wall time. 0x5504 counts XM leaf
  calls and 0x5505 reports the number of workers.
- Completion publish timing excludes diagnostic publication. The host residual
  includes dispatch, polling, scheduling and diagnostics, not just interrupts.
- The telemetry reader uses two identical reads, not a source-side atomic
  snapshot. Captures are checked for per-sequence completeness and index gaps.
- New percentile calculations use nearest rank. Archived v35-v38 micro-test
  files predate that correction and used floor((N-1)*p); use raw observations
  or `max_us` rather than their old small-sample p99 values.
- Token throughput is accepted only after every response matches the expected
  text. The count16 test is token-capped, not a full answer through ten.
- count64 requires exactly 64 output tokens, a length stop, and an exact prefix
  of the requested comma-separated integers. It is a longer single concurrent
  cohort, not a repeated arrival-rate or saturation benchmark.

`docs/pacc-xm-context-qwen35-20261107.md` and `evidence/` preserve the original
v38 measurements. `docs/pacc-xm-performance-qwen35-20261107.md` and
`evidence-performance/` record
v39-v46. Original large raw logs remain under `/srv/lanxin/qwen35-a3b/logs`;
manifests hash them. Do not regenerate the old evidence manifest with new sources.

## Compiler and VLEN experiments

The live host has VLEN=128 bits; the PACC guest reports `vlenb=128` bytes,
which is VLEN=1024 bits. Do not use host `-march=native` for guest kernels.
The existing JIT target already includes `zvl1024b`; this minimum VLEN is not
the same as requesting fixed-length 1024-bit compiler optimization.

`COMPILER_PROFILE=vlen1024` compiles jobd, workers, operand packing/reduction
and payload copying with `-O3 -march=rv64gcv_zvl1024b
-mrvv-vector-bits=1024`. The native XM assembly leaf is unchanged. The build
probes host VLEN and uses separate VLA objects for host-only reference tests
when it is not 1024; those tests do not qualify the fixed-length firmware.
Real PACC numerical/completion tests remain mandatory.

```sh
VERSION=v49_vlen1024 BUILD_ID=0x20031 COMPILER_PROFILE=vlen1024 \
    bash bin/build-xm-batch-jobd.sh
# Existing images are never overwritten. Use a new version for a rebuild.
sudo python3 bin/load-pacc0-v26-dynamic.py --version v49_vlen1024
# After the same maintenance checks as above:
sudo python3 bin/load-pacc0-v26-dynamic.py --version v49_vlen1024 --apply
bash bin/run-xm-validation.sh v49-large 0x20031
bash bin/run-xm-validation.sh v49-workers 0x20031 --hardware-workers
bash bin/run-xm-qwen-benchmark.sh v49-p4 0x20031 4 count16
```

The COMGR change makes final code generation explicitly `-O3`, including the
large-bitcode path that skips preoptimization. Set
`HETGPU_SIFIVE_CODEGEN_OPT_LEVEL=0`, `1`, `2` or `3` for controlled comparisons.
The ELF cache version is bumped and includes this setting. This source fix
was tested through an isolated equivalent `clang-sifive-o3.py` wrapper, not
installed as a rebuilt production runtime; the full release build is blocked
by an absent LLVM submodule. The wrapper uses a distinct compiler path/cache
and records compiler inputs, flags, duration and output size. It does not enable
fast-math. No current throughput improvement is established for JIT O3 alone.

See `docs/pacc-xm-compiler-options-20261107.md` and
`evidence-compiler-options/` for the measured comparisons and limitations.

## Increasing offload depth

The fifth argument to `run-xm-qwen-benchmark.sh` is now `ngl` (default 1).
The Python harness also accepts `--ngl`, records it in the result, refuses
to overwrite prior runs, and saves failure results with null TPS. Its bounded
startup/request timeouts can be set with `--startup-timeout` and
`--request-timeout`. CPU mode always selects ngl=0 and `--device none`.

```sh
bash bin/run-xm-qwen-benchmark.sh ngl2-check 0x20031 4 count16 2
bash bin/run-xm-qwen-benchmark.sh ngl3-check 0x20031 4 count16 3
```

These larger depths are experiments, not validated working configurations.
Both currently fail before readiness: a Q8_0 x Q8_1 MMVQ takes the generic
ELF path and needs 23,549,120 bytes against the 2,097,152-byte owned payload.
The Q6_K-only native tiled interception does not cover this type. Do not
increase the advertised shared size beyond the actual allocation or enable
success-on-error. Real Q8 tiling/ABI and numerical validation are required.

The fixed expected job count applies only to ngl=1. Other depths still require
correct tokens, successful process exit, firmware build matching, complete
observed job-5 phase sets and no ring-index gaps, but have no established total
job count. The profiler tracks job-5 compute phases, not all kernel types;
zero job-5 completions does not imply no other device work executed.
See `docs/pacc-qwen-ngl-sweep-20261107.md` and `evidence-ngl-sweep/`.
