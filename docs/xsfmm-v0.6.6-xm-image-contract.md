# Xsfmm v0.6.6 XM/VCIX image contract

This contract separates the fixed SoC image from PACC Linux firmware. Replacing
`lx500_pacc_jobd_*.bin` changes PACC software only; it does not replace the XM
decoder or VCIX datapath.

## Required instruction interface

An accepted image must execute the Xsfmm v0.6.6 words used by the checked
firmware:

| Operation | Encoding |
| --- | --- |
| `sf.vsettnt` | `0x508772d7` |
| `sf.vsettm` | `0x8416f2d7` |
| `sf.vsettk` | `0x8422f357` |
| `sf.vtzero.t` | `0x43e06057` |
| `sf.vsettn` | `0x8402f357` |
| `sf.mm.f.f` | `0xf2881077` |
| `sf.vste32` | `0x52567027` |

The implementation must preserve tile accumulator state across consecutive
`sf.mm.f.f` operations until `sf.vste32`, including K loops and adjacent output
tiles. An illegal instruction, silent context reset, stale tile, or missing
completion is a contract failure.

## Capability and identity registers

The SoC integration must expose a read-only XM capability block in the PACC MMIO
window. The exact offset can be supplied by the board device tree, but the block
must contain:

- Magic `XMV066` and a monotonically increasing image build ID.
- Encoding ABI major/minor, with ABI 0.6.6 represented explicitly.
- Maximum supported M, N, K, active contexts, and in-flight tiles.
- Completion mechanism version and cycle-counter frequency.
- Feature bits for persistent tile context, BF16 input, FP32 accumulation,
  `sf.vste32`, completion doorbell, and completion sequence number.

Software must reject an image when the block is absent or reports smaller
limits than the requested operation.

## Acceptance gates

All gates are hardware-only. CPU fallback and software GEMM are disabled.

| Gate | Requirement |
| --- | --- |
| Encoding | All seven v0.6.6 words execute and return correct data |
| Stable base | Four PACC devices pass M=32, N=32, K=2048 twice |
| Large M | Four PACC devices pass M=2048, N=32, K=2048 |
| Large K | Four PACC devices pass M=32, N=32, K=6144 |
| Continuous context | 64 consecutive Xsfmm operations complete without reinitializing XM |
| Completion | Sequence-numbered completion; no stale success; PACC-side p99 below 10 us |
| Sustained run | 10,000 requests with zero mismatch, timeout, or lost completion |

The performance report must preserve instruction cycles, host submit latency,
completion latency, and end-to-end wall time as separate metrics.

## Current board observation

The evaluated LX500 system exposes four fixed `lanxin,pacc-lx500` device-tree
nodes. It exposes no Linux FPGA manager, FPGA region, FPGA bridge, MTD device, or
board bitstream file. Therefore Linux can replace PACC firmware but cannot
replace the XM/VCIX image. Producing the required image needs the board RTL and
vendor synthesis/programming flow, or a vendor-provided signed image and
bootloader flashing procedure.
