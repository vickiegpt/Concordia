# FU900/PACC Xsfmm v0.6.6 XM Hardware Image Contract

## Scope

This contract describes the hardware artifact required by the Concordia PACC
runtime. It is not a PACC Linux firmware image. Files such as
`lx500_pacc*.bin`, `u-boot.itb`, a DTB, or a jobd executable cannot add the
Tightly Integrated Matrix Engine to a FU900/PACC core.

The deliverable must be produced by the FU900 hardware build flow that owns the
SiFive Bullet core and its attached XM implementation. If the platform is an
emulator, the deliverable is the emulator image/checkpoint plus its load
command. If it is an FPGA, it is the board-specific programming image plus its
programmer command. The loader and active hardware build ID are mandatory parts
of the delivery.

## ISA Compatibility

The target specification is SiFive "Xsfmm Family of Attached Matrix
Extensions", version 0.6.6, dated 2026-01-08.

Official specification:

`https://sifive.cdn.prismic.io/sifive/aitG-KlQnVZVEO9g_xsfmm-matrix-spec-v0p6p6.pdf`

Reference PDF SHA-256 observed on 2026-07-24:

`c74853a1cc799cc053d46b5f936f7938eb14d0c4f200b9f4a88b54e418d5427e`

Versions 0.6.1 through 0.6.6 clarify dependencies, supported combinations, and
exception behavior. They do not replace the instruction encoding introduced by
the ISA-incompatible v0.6 revision. A v0.6 implementation can satisfy v0.6.6
provided it implements the final dependency, supported-combination, `vstart`,
and reserved `altfmt/vtype` behavior.

The image must implement, on every PACC hart:

- `Xsfmmbase`
- `Xsfmm32a16f`
- `Zve32f`
- `Zvfbfmin`
- VLEN of 1024 bits
- TE=32 for TEW=32, matching the current 32x32 jobd tiling
- `mstatus.MS` in bits 30:29
- BF16 operands selected by `SEW=16`, `TWIDEN=2`, `ALTFMT=1`
- FP32 tile accumulation for `sf.mm.f.f`

The runtime's required instruction words are:

| Instruction | Word |
|---|---:|
| `sf.vsettnt t0,a4,e16alt,w2` | `0x508772d7` |
| `sf.vsettm t0,a3` | `0x8416f2d7` |
| `sf.vsettk t1,t0` | `0x8422f357` |
| `sf.vtzero.t mt0` | `0x43e06057` |
| `sf.vsettn t1,t0` | `0x8402f357` |
| `sf.mm.f.f mt0,v8,v16` | `0xf2881077` |
| `sf.vste32 t0,(a2)` | `0x52567027` |
| `sf.vtdiscard` | `0x43c06057` |

## Platform Contract

The target has four PACC instances at:

- `0x38100000`
- `0x38500000`
- `0x39100000`
- `0x39500000`

Each instance has four harts. The image must instantiate and clock an XM for
each PACC instance, preserve the existing shared-DDR and mailbox address map,
and preserve PACC-only reset. The XM clock/reset domain must be released before
the PACC hart executes its first tile-state instruction.

The image delivery manifest must contain:

- hardware image file name, byte size, and SHA-256
- immutable hardware build ID
- FU900 RTL release and XM configuration/release
- Xsfmm specification version
- PACC count, harts per PACC, VLEN, TE, and required extension list
- synthesis/emulation tool and version
- top-level configuration and source revision/checksum
- exact load/program command
- exact active-build-ID readback command
- reset and clock sequence
- rollback image and rollback command
- whether activation requires a host reboot or power cycle

The checked-in requirements file is
`tools/xsfmm_v0p6p6_xm_requirements.json`. A candidate delivery is checked with
`tools/check_xsfmm_v0p6p6_image.py`. The hardware owner can start from
`tools/xsfmm_v0p6p6_delivery.example.json`.

## Acceptance Gates

An image is accepted only when all gates pass:

1. The delivered image and manifest hashes match.
2. Active hardware build ID readback matches the manifest.
3. The live PACC DTB advertises all required extensions. This is metadata only,
   not proof of execution.
4. `mstatus.MS` can be enabled for the submitting task.
5. A minimal 4x4x4 BF16 test completes `sf.vtzero.t`,
   `sf.mm.f.f`, `sf.vste32`, and `sf.vtdiscard` without timeout or illegal
   instruction.
6. The 4x4x4 output has zero mismatches against a host FP32 reference.
7. A 32x32x32 BF16 test has zero mismatches.
8. All four PACC instances pass independently.
9. Four concurrent PACC requests complete with unique per-PACC control slots.
10. The main-host boot ID remains unchanged during PACC-only validation.
11. XM issue and completion counters wrap or drain correctly across at least
    10,000 requests; no fixed-width counter may permanently stop submission.

The 2026-08-02 hardware-only run narrowed the current image failure to the XM
datapath. PACC0 and PACC1 returned exact output for `M=32,N=2,K=6144`, then
stopped after 168-176 cumulative tiles, close to `2^19` matrix instructions
when the 3072 `sf.mm.f.f` operations and stores per tile are counted. Changing
jobd batching, callback lifetime, `mstatus.MS`, and `sf.vtdiscard` only moved
the boundary. PACC2 and PACC3 stopped on their first tile. The replacement
image must explicitly test this counter-wrap case.

Until that image is available, jobd defaults to a 480,000-command estimated
lifetime budget. PACC1 completed 30 consecutive `M=32,N=8,K=4096` requests
with zero mismatches; a 250-iteration stress run failed closed at iteration 133
with status `0xffff1f25` before the hardware hang. The PACC-only recovery left
the main-host boot ID unchanged. This is a recoverability guard, not evidence
that the current image meets the sustained-run gate.

Advertising `xsfmmbase` in a DTB is not sufficient. Returning from
configuration instructions is not sufficient. A completion record without
correct output is not sufficient.

## Current Board Evidence

The current FU900 DTB advertises `xsfmmbase`, `xsfmm32a8i`,
`xsfmm32a16f`, `xsfmm32a32f`, and `xsfmm64t`. It also contains EDAC
descriptions for the Tightly Integrated Matrix Engine tile-state RAM.

The software path has independently verified the v0.6.6 instruction words and
enabled `mstatus.MS` using a PACC-kernel-compatible context module. The
hardware-only 4x4x4 BF16 test then times out on the first tile-state operation
without completion:

`/tmp/lanxin_disagg_eval/xsfmm_mscheck6_20260724_065318`

The fail-closed four-PACC runner was revalidated on 2026-07-25 with the
hardware-only firmware SHA-256
`2d2645cab94f6e492f6922923d403f0f63d2fa1fecc9ff7aa97791de4371cb15`.
PACC0 reached the jobd ready record, but its first 4x4x4 BF16 request returned
no completion within five seconds:

`/tmp/lanxin_disagg_eval/xsfmm_runner_recovery_20260725_020856`

The runner reported `acceptance=failed`, counted zero Xsfmm throughput, cleared
the shared-DDR control window, and restored
`lx500_pacc_jobd_hostbase_idmarker.bin`. All four stable beacons subsequently
reported phase `0x7002`. The main-host boot ID remained
`9508c6db-ddf6-4e8d-8162-bd2e32a03660`.

The canonical Xsfmm+CUDA peak entry point reproduced the same failure on all
four PACC instances concurrently. Every probe exited with submit timeout status
9, the aggregate Xsfmm throughput remained zero, and the model stage was
blocked with exit status 20:

`/tmp/lanxin_disagg_eval/xsfmm_cuda_peak_gate_20260725_022230`

This means the current DTB and the active hardware implementation are not a
passing Xsfmm delivery. The board has no Linux-visible FPGA manager, JTAG/XVC
device, vendor RTL tree, synthesis tool, or hardware image loader. A matching
hardware artifact must therefore come from the FU900 hardware owner/build
environment; it cannot be reconstructed from the PACC Linux firmware.

`tools/run_xsfmm_cuda_peak.sh` enforces the 4x4x4 and 32x32x32 four-PACC gates
before starting a model runner. It accepts an end-to-end TPS result only when
the model runner reports at least one real Xsfmm offload, positive generated
tokens, and zero mismatches. Standalone PACC TOPS and CUDA token throughput are
never added together.

## No-Reboot Rule

Do not activate a candidate that requires a main-host reboot or power cycle
without explicit user approval. PACC-only reset/recovery is allowed. Always
record the main-host boot ID before and after validation and restore the stable
PACC firmware after a failed test.
