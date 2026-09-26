# Text SASS GEU and BF16 packing support

`F2FP.BF16.F32.PACK_AB` supports the four independently compiler-confirmed
combinations of default RN or explicit RZ rounding, with or without RELU.
They map to `cvt.{rn,rz}{.relu}.bf16x2.f32`. The first source becomes the
upper 16-bit lane; the second becomes the lower lane. Plain real GPR
operands and RZ sources are supported; malformed operand counts, other
rounding/saturation/FTZ modes and source transformations are diagnosed.
The existing FP16 path remains unchanged.

Rounding is observable: input `0x3f80c000` becomes BF16 `0x3f81` under RN
and `0x3f80` under RZ. RELU changes a negative input such as -1 to zero.
The unit tests exercise these numeric distinctions, packing order,
destination/source aliasing and ties to even. Four offline NVIDIA ptxas
references emit matching F2FP forms, including RELU+RZ.

`FSETP.GEU.AND Pdst,PT,Ra,Rb,PT` maps to unordered `setp.geu.f32`:
either NaN makes the comparison true. Support is deliberately limited
to a discarded second predicate result, AND with true PT, plain GPR/RZ
comparison operands and no FTZ. Live second results, dynamic/negated
boolean inputs, OR/XOR, FTZ and transformed sources are diagnosed rather
than silently losing semantics. Other comparison paths are unchanged.

The optional test assembles actual lifter output for all four BF16 modes:

```sh
HETGPU_TEST_PTXAS=/path/to/ptxas cargo test -p ptx bf16_generated_ptx_all_modes_assemble_offline -- --ignored
```

It requires ptxas with sm_120 support; it performs no GPU execution.
The internal PTX parser still returns `Todo` for BF16 conversion. That
parser limitation is separate from NVIDIA PTX legality and is not hidden
by a stub parser or a claim that the complete compiler supports BF16.

These fixes cover the text frontend and the external cuobjdump path.
The built-in binary decoder does not preserve all required modifiers and
is not certified by these tests. A workload using default RN and the
simple AND/PT GEU form can pass while previously unsupported RZ, RELU,
FTZ or predicate combinations remain incorrect outside its reached
instruction domain. Such a pass is not general ISA coverage.

References: [NVIDIA conversion and packing semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
and [predicate comparison semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#comparison-and-selection-instructions-setp).
