# PACC0 XM context and Qwen35 diagnostic release

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

## Deployment constraints

These commands target the existing `/srv/lanxin/qwen35-a3b` evaluation tree on
`lanxin2`, not an arbitrary machine. Copy the source/tools into that tree's `bin`
directory. The build uses its existing vendor patcher under
`pacc-runtime-head/ext/pacc_runtime-sys/patch_lx500_pacc_inplace_jobd.sh`, original
vendor firmware backup, Clang 20 and libfdt. Test input dequantization uses the
existing llama.cpp `libggml-base.so`.

`build-xm-batch-jobd.sh` creates v38 and runs host layout tests before linking.
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
sudo python3 bin/load-pacc0-v26-dynamic.py --version v38_q6k_xm
# Only after the dry-run checks and a PACC0 maintenance window:
sudo python3 bin/load-pacc0-v26-dynamic.py --version v38_q6k_xm --apply
```

For hardware tests, use `source bin/pacc-env.sh`, the verified owned DDR base and
size, bootstrap=0, and strict completion (ASYNC_NO_WAIT=0, SKIP_JOB_WAIT=0).
Bootstrap=0 skips an obsolete startup handshake, not job completion.

```sh
python3 bin/profile-jobd-phases.py --output logs/xm-validation.json \
  --timeout 180 --expected-submissions 493 --require-build 0x20026 \
  -- bin/test_jobd_q6k --hardware-large
python3 bin/decode-xm-config.py logs/profile-v38-q6k-xm.json
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
- Completion publish timing excludes diagnostic publication. The host residual
  includes dispatch, polling, scheduling and diagnostics, not just interrupts.
- The telemetry reader uses two identical reads, not a source-side atomic
  snapshot. Captures are checked for per-sequence completeness and index gaps.
- New percentile calculations use nearest rank. Archived v35-v38 micro-test
  files predate that correction and used floor((N-1)*p); use raw observations
  or `max_us` rather than their old small-sample p99 values.
- Token throughput is accepted only after every response matches the expected
  text. The count16 test is token-capped, not a full answer through ten.

The report in `docs/pacc-xm-context-qwen35-20261107.md` records the measured
results and remaining bottlenecks. Original large raw logs remain under
`/srv/lanxin/qwen35-a3b/logs`; the evidence manifest hashes them.
