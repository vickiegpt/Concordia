# SM120 recovery validation: scope and stronger regression tests

The five recovery repairs were designed and experimentally evaluated for
**SM120**. This is their original validity domain, not merely a restriction on
a later test run. Nothing here establishes another SASS architecture's encoding,
register-pair semantics, or compiler behavior. No other architecture adapter,
compatibility policy, target matrix, or dispatch logic is introduced.

Within SM120, correctness is broader than one QKV workload but still bounded by
the implemented operand forms. A successful fixed HBFSim model request does not
prove an untaken tail path or an untested instruction mode. The follow-up tests
exercise instruction semantics independently of model shapes and names.

| Repair | Supported domain tested | Remaining boundary |
|---|---|---|
| FSEL negative qNaN | Confirmed immediate encoding, payload bits, register/zero sources, selection predicate; ordinary source execution predicates | Other immediate execution encodings are diagnosed; no arbitrary binary-decoder proof |
| IMAD.HI.U32 | Complete unsigned GPR-pair addend, carry/wrap, zero/immediate sources, register aliases, execution guards | Signed/X/transformed/constant-bank forms outside implemented domain |
| CS2R SRZ | Every supported even GPR pair, both predicate directions, explicit single-word form | Other special-register identities and built-in binary decoding unverified |
| GEU / BF16 PACK_AB | Simple AND/PT comparison, four RN/RZ with/without RELU conversions, FP classes, rounding boundaries, lanes/aliases/guards | Dynamic boolean/FTZ/live second-predicate forms diagnosed; checked internal parser rejects RELU |
| Register pairs | Supported IADD.64 / WIDE.U32 / ordinary IADD3 forms, integer boundaries, aliases, predicates, non-QKV fixture | Signed/carry/unsupported widths and descriptor atomics retain explicit diagnostics |

## Method and references

[LLVM's regression-testing guidance](https://llvm.org/docs/TestingGuide.html)
motivates small reproducible feature/bug cases and explicit target selection.
Tests follow data dependencies rather than fixing scratch-register names.
This does not import LLVM's backend abstractions or claim LLVM tests ran.

[CUTLASS numeric-conversion tests](https://github.com/NVIDIA/cutlass/blob/main/test/unit/core/numeric_conversion.cu)
motivate checking conversions across numerical types and lane combinations.
Our exhaustive BF16 encoding and midpoint CPU enumeration is an additional,
independently written check; CUTLASS kernels or host conversion code are not
copied or run as an oracle.

[NVIDIA PTX conversion](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
and [predicate comparison](https://docs.nvidia.com/cuda/parallel-thread-execution/#comparison-and-selection-instructions-setp)
semantics define the emitted operations. These PTX references do not alone prove
a SM120 SASS encoding; compiler-produced SM120 evidence remains separate.

## CPU coverage and independence

`generality_integer.rs` interprets the emitted PTX subset and compares against a
whole-operation `u128` reference using the original register state. It checks
3,702 executions across register renaming, aliases, predicates, boundary/seeded
values and immediate/zero/uniform forms. Seven deliberate wrong-emission
executions show the checks detect lost addend high words, a missing pair write,
a dropped predicate and swapped FSEL arms. They do not mutate production code.

`generality_float.rs` reads actual emitted conversion/comparison opcodes,
operands and predicates. Its bit-rounding model is compared against an independent
reference selecting adjacent BF16 values by exact `f64` distance. It enumerates
all 65,536 widened BF16 encodings in four modes and every finite rounding midpoint
with adjacent FP32 values and both signs. Separate cases cover tiny NaN payloads,
subnormals, signed zeros, infinities, overflow, aliasing and register renaming.
NaNs are compared as a class; no unmeasured payload contract is invented.
Mode/lane mutations must be detected. The GEU reference uses integer float-order
keys independently of the interpreted floating comparison.

These are software semantic checks of selected emitted instructions. They are
not a native-GPU differential run, complete PTX interpreter, or exhaustive proof
of all FP32 inputs and SM120 instructions. They complement the existing actual
parser/integration fixtures and optional offline assembler checks.

`generality_sm120.rs` adds an optional offline compiler check: independently
specified HI arithmetic and BF16 RZ/RELU PTX are compiled for SM120, actual SASS
instructions are parsed and lifted, and the selected lifted PTX is assembled.
The independent input kernels store their results so the target operations remain
live. Reassembly of selected lifted instructions checks legality only; it does not
prove whole-kernel recovery, initialized-register behavior, or GPU output equality.
A changed compiler instruction pattern fails explicitly instead of silently skipping.

Validation on the reviewed source: **121 passed, 0 failed, 0 ignored, 0 filtered**
(107 module, 12 integration, 2 pipeline tests). This includes four explicitly
configured optional offline tool tests. It is a scoped harness result, not a full
workspace or GPU result. No production semantics changed in this follow-up.

## Running

Use the portable harness in [tools/SASS_CPU_TESTS.md](../tools/SASS_CPU_TESTS.md).
All new lifts explicitly select SM120. Run ordinary CPU tests first; configure
the optional NVIDIA offline tools explicitly when running ignored tool tests.
An unavailable tool or unsupported form is not a PASS. Existing complete-workspace
build limitations, built-in decoder limits, and RELU downstream-parser rejection
remain separately reported. No GPU or HBFSim frozen artifact is changed.
