# Text SASS IMAD.HI.U32 support

The text lifter lowers the confirmed unsigned high-result form as
`high32(u32(a) * u32(b) + u64(c))`, modulo 64 bits before extraction.
The register addend is the pair `Rn:Rn+1`, with `Rn` the low word. PTX
`mad.hi.u32` instead adds a **32-bit** value after extraction and is not
an equivalent replacement.

Independent NVIDIA ptxas compilation of `mad.hi.u32 d,a,b,c` on sm_120
loads `R5=c`, sets `R4=0`, and emits `IMAD.HI.U32 R5,R6,R7,R4`.
For `a=b=0,c=7`, substituting a PTX addend of `R4` would return 0 instead
of 7. The lowering therefore multiplies wide, packs the addend, adds
wide, shifts, and narrows, using block-local temporaries before changing
any source/destination alias. Every instruction retains its execution
predicate.

Supported text operands:

- One real GPR destination; three sources; only `HI` and `U32` modifiers.
- Product sources: plain GPRs, RZ, or unsigned 32-bit immediates.
- Addend: RZ or an aligned real GPR pair starting at R0–R252.

Other types, extra modifiers/carry inputs, missing operands, constants,
negation/absolute/component modifiers, uniform pairs, immediate addends,
and pairs crossing the real-register boundary produce diagnostics.
No operand is silently dropped or replaced by a zero.

The conventional unit tests interpret the generated integer PTX subset
against an independent u128 oracle, including low-word carry, wrapping,
aliases and predication. The optional assembler test can be run with:

```sh
HETGPU_TEST_PTXAS=/path/to/ptxas cargo test -p ptx imad_hi_generated_ptx_assembles_offline -- --ignored
```

It requires a ptxas supporting sm_120 and uses no GPU. A full workspace
build still requires the repository's normal LLVM/submodule dependencies.

This support belongs to `lift_sass_text_to_ptx` and the external
cuobjdump path selected by `HETGPU_SASS_LIFTER_CUOBJDUMP`. The built-in
binary decoder does not yet preserve these modifiers; this change does
not certify that separate path or all NVIDIA SASS architectures.
A successful fixed workload validates its reached inputs and consumers,
not arbitrary addend pairs or unexecuted branches.

Reference: [NVIDIA PTX integer mad semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-mad).
