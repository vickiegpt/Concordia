# 2048 independent requests: capacity preparation

The requested workload is 2048 independent active requests across four PACC
workers, 512 sequences per worker. It is not just llama.cpp's `-b 2048` token
capacity. No PACC TPS has been measured for this configuration.

## Live checks

The original built libllama ABI reports `llama_max_parallel_sequences() = 256`.
The existing benchmark also had a 32-sequence check and allocated a fixed
256-entry llama_batch. Merely changing an environment variable cannot run 512
sequences per worker and could overflow that batch allocation if the guard were
removed without fixing the allocation.

An isolated source copy and libllama build now support 512 sequences. The original
source, original build, running CPU llama-server and installed runtime are unchanged.
The sequence limit is compile-time configurable in the copied header. The v4
benchmark checks the actual loaded ABI limit and allocates at least one batch entry
per active sequence. No accelerator numerical code was changed.

Built ABI result: 512. Single-instance CPU context allocation also passed:

| Item | Measurement |
|---|---:|
| Independent sequences | 512 |
| Context tokens per sequence | 256 |
| Total context tokens | 131072 |
| Logical token batch / microbatch | 512 / 8 |
| Peak RSS | 57674508 KiB, about 55.0 GiB |
| Mapped model buffer | 20985.64 MiB |
| Recurrent state | 32160 MiB |
| KV cache | 2560 MiB |
| Compute buffer reservation | 2202.52 MiB |
| Output buffer | 485 MiB |

This was CPU allocation only, with an 80 GiB virtual-address-space limit and
300-second timeout. It did not generate tokens or test PACC memory access. Four
instances would need about 125.6 GiB of recurrent state alone. Multiplying RSS is
not an exact aggregate-memory prediction because file-backed model pages can be
shared. A subsequent four-process barrier test simultaneously held all **2048
independent sequence slots across four CPU contexts** successfully. Each worker held its allocation until
the parent observed all four ready reports; all four then exited with code zero.
Host MemAvailable was initially 259247408 KiB and reached a minimum of 114102192
KiB (about 108.8 GiB). The guard would terminate only the test workers below 20 GiB.
This is not a PSS measurement and does not attribute all host memory changes to
the test. Accelerator execution remains unqualified. All throughput fields in
these capacity reports are null.

The new build also repeated the two-request CPU reference: token sequences
`[24,24]` and `[15,16]`, matching the original reference. This is a small CPU
regression, not 2048-request inference correctness.

## Hardware blocker

The machine has rebooted since the previous hardware qualification, but not as
part of this task. Current boot: `4ca2fa21-0d48-4090-93a5-51206ac1de29`.
The helper now advertises fixed base `0x20110600000`, size 16 MiB, and the user
confirmed that jobd/shared-DDR were not reinitialized after boot. The old tests
used a dynamically allocated, pinned 4 MiB region on a different boot.

The existing isolation wrapper correctly refused to run. Its boot checks were
not bypassed. No writes to either old address, no firmware reload, no device reset,
and no host reboot were performed. Recover owned DDR and per-device jobd, then
requalify identity/completion/numerical output before any PACC run. Permission for
that device-level maintenance was requested separately.

The previous RoPE staging failure (592512 bytes versus 524288 per-device window)
also remains relevant until a new owned partition layout or tiled path is validated.
The new 16 MiB configuration is not evidence of a safe or usable larger aperture.
XM leaf submissions still support at most eight independent columns: a 512-request
logical batch does not become one N=512 matrix operation automatically.

## Reproduce the isolated build

These commands target the existing lanxin2 source/backend snapshot, not arbitrary
upstream versions. The custom CMake target reuses the original GGML CPU/CUDA shared
libraries; it does not rebuild or replace them. Check the original source version
and ABI before reusing this recipe elsewhere.

```bash
ROOT=/srv/lanxin/qwen35-a3b
TOOLS=$ROOT/Concordia-xm-release/tools/pacc-qwen35
mkdir "$ROOT/seq512-source"
cp -a "$ROOT/llama.cpp/src" "$ROOT/llama.cpp/include" "$ROOT/seq512-source/"
patch --batch --fuzz=0 -d "$ROOT/seq512-source" -p1 < "$TOOLS/seq512-llama-max-seq.patch"
cp "$TOOLS/seq512-CMakeLists.txt" "$ROOT/seq512-source/CMakeLists.txt"
cmake -S "$ROOT/seq512-source" -B "$ROOT/build-seq512" \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SOURCE="$ROOT/seq512-source" -DGGML_SOURCE="$ROOT/llama.cpp/ggml" \
  -DGGML_BIN="$ROOT/llama.cpp/build-pacc-cuda-gcc14/bin" \
  -DPROBE_SOURCE="$TOOLS/qwen-capacity-probe.cpp" \
  -DHARNESS_SOURCE="$TOOLS/seq512-qwen-multiseq.cpp" \
  -DJSON_INCLUDE="$ROOT/llama.cpp/vendor/nlohmann"
cmake --build "$ROOT/build-seq512" -j8
```

The directories already exist from this test; do not overwrite them for a new
variant. When running the new executables, prepend `build-seq512/bin` to
LD_LIBRARY_PATH after sourcing the compatibility environment, or its original
libllama search path will select the 256-sequence library again. Bootstrap must
remain disabled until hardware recovery is qualified.
