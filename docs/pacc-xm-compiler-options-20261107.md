# PACC compiler options and VLEN audit, 2026-11-07

## Scope and result

Machine: lanxin2, host boot ID `89a6c374-059a-40da-8c4f-c00179fd48d4`.
No host reboot. Only PACC0 was rebound through the vendor driver. PACC1-3
and the unrelated CPU server were left alone. Canonical vendor firmware was
restored after each temporary load.

The host measured VLEN=128 bits; PACC capability event 0x5325 reports
`vlenb=128` bytes, or VLEN=1024 bits. The fixed-1024 candidate was executed
only on PACC, never on the 128-bit host. The existing JIT march already had
`zvl1024b`; the jobd/Q6 compilation did not request fixed VLEN.

Compiler settings are a real issue, but not a demonstrated explanation of
the remaining throughput gap. The earlier `sf.vtzero.t` fault was resolved
by the guest ISA-name/context fixes described in the prior context report;
this experiment does not replace that evidence with a compiler-only claim.

The existing RVV unpack loop explicitly requests `vsetvl_e8m1(32-lane)`;
reduction requests `vsetvl_e32m1(rows-row)` with at most 32 rows. VLA code
already uses these lengths on the 1024-bit guest. Setting fixed VLEN does
not widen the algorithm's requested element count or remove strided stores.

## Firmware A/B

All variants use the same jobd/Q6/XM algorithms. No software matmul fallback,
fast-math, firmware protocol changes or new host polling policy was added.
Q6 floating-point contraction remains disabled. Four workers are within
PACC0, not four devices.

| Image | Change from baseline | Full projection runtime ms |
| --- | --- | ---: |
| v46 | Q6 O3 RVV; jobd/workers O2 generic | 2294.033 |
| v47 | jobd/workers O3 RVV | 2323.240 |
| v48 | O3 RVV plus X280 scheduling tune | 2303.242 |
| v49 | O3, rv64gcv_zvl1024b, fixed VLEN1024 | 2243.731 |

The v48 tuning experiment is not a claim that the PACC core is X280.
Each large suite passed 493 submissions with complete captures and no index
gaps. The full projection is M=248320, N=1, K=2048. The numbers are runtime
call durations, not whole test/reference-check wall times. Individual runs
are not a statistical speedup claim.

v49 also passed five uneven-worker/tail submissions. Large-suite aggregate
device compute time was 0.996772s versus v46 1.016407s, approximately 1.9%
lower in this pair. Max absolute full-projection error versus the dequantized
reference remained 0.0012552687. The fixed-length candidate needs these
board tests: the host build tests use separately compiled VLA test objects.

## JIT code-generation fix

Large bitcode skips IR preoptimization, but final BC/IR-to-object codegen
also lacked an explicit optimization level. COMGR now passes validated
`HETGPU_SIFIVE_CODEGEN_OPT_LEVEL` (default 3) to all four final codegen paths,
including assembly and llc fallback. ELF cache v6 includes this environment
setting and invalidates pre-fix objects.

An isolated compiler wrapper tested equivalent final `-O3` behavior without
replacing the active runtime. One observed module was 9,340,108-byte BC,
83,065,876-byte sanitized IR and 5,641,112-byte output. Code generation
completed in 131.759s. Compilation happened during server startup and is
excluded from the cohort throughput below; do not call that startup-free
metric cold-start throughput.

Targeted tests against the actual isolated COMGR source: 8 passed. The full
crate has an existing `sifive_tile_dims_x390` failure, reproduced with the
untouched baseline. The full workspace build is blocked by missing
`ext/llvm-project/llvm-sys/Cargo.toml`. The source fix has not been installed
as a rebuilt production runtime.

## Correct-token Qwen measurements

Qwen3.5-35B-A3B-Q4_K_M, ngl=1, CPU model blocks plus PACC0 Q6 output projection
and RMSNorm. Eight CPU threads; batch=256. This is neither all-PACC nor
GPU+four-PACC. Throughput is completed output tokens divided by cohort wall
time, includes request prefill and excludes server startup. Count16 is a
token-capped exact-text correctness test, not a saturation benchmark.

| Firmware / JIT | Requests x tokens | Correct tokens | Wall s | TPS |
| --- | ---: | ---: | ---: | ---: |
| v46 / original warm | 4 x 16 | 64 | 61.504995 | 1.040566 |
| v46 / O3 fresh cache | 4 x 16 | 64 | 61.239842 | 1.045071 |
| v46 / O3 warm | 4 x 16 | 64 | 61.814569 | 1.035355 |
| v49 / original warm | 4 x 16 | 64 | 61.385922 | 1.042584 |
| v49 / original warm | 8 x 64 | 512 | 260.042295 | 1.968910 |

All four four-request cohorts passed exact-text verification and 8245/8245 submissions
with no telemetry gaps. JIT O3 alone and fixed VLEN1024 have no material
four-request throughput benefit in these measurements.

The eight-request cohort passed exact-prefix, token-count and length-stop
checks, with 31525/31525 submissions and no index gaps. It remains consistent
with the previously measured v46 range of 1.960028-1.969136 TPS, rather than
establishing a new speedup. Device input copy totaled 34.516646s, compute
86.926113s and output copy 10.307137s. Completion publication p99 was 7us,
not an end-to-end request/completion-latency claim. PACC0 was left running
the validated v49 image; its canonical vendor boot file remains restored.

For v49 four-request telemetry, device input copy totaled 8.940938s, compute
18.548144s, output copy 1.358064s. Summed worker detail times were activation
pack 2.769562s, weight unpack 44.307669s, XM leaf 11.246314s, reduction
10.358427s; these overlap across four workers and must NOT be added to wall
time. There were 16,885,760 XM leaf calls. Weight unpacking and repeated
small K=16 leaf calls remain measured optimization targets. Larger K cannot
be substituted blindly because exact Q6 subblock scale arithmetic must be
preserved. Cache/reuse changes need buffer ownership and invalidation first.

Telemetry uses double-read stability, not source-side atomic snapshots.
Build markers, complete per-sequence phases and index-gap checks qualify
these captures; they do not eliminate that instrumentation limitation.

## Reproduction and evidence

Use `tools/pacc-qwen35/README.md` for deployment constraints and commands.
`evidence-compiler-options/` contains checked cohort JSON, microtest logs,
phase summaries, build logs and SHA256 provenance. Large raw traces stay
under `/srv/lanxin/qwen35-a3b/logs` and are hashed rather than committed.
Firmware and model binaries are not committed. Existing performance evidence
is preserved unchanged. Read the new build logs for compiler profiles;
source hashes alone cannot distinguish these compiler-only firmware builds.
