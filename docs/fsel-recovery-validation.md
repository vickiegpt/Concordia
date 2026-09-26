# FSEL negative-qNaN recovery validation

The text frontend preserves the immediate payload only for the proved FSEL
immediate form: source slot 1, low encoding bits `0x7808`, and a negative quiet
NaN (`bits & 0xffc00000 == 0xffc00000`). The tests deliberately reject missing
encoding, the register opcode, the wrong source slot, positive NaNs, negative
infinity, signaling NaNs, finite values, and an unproved execution-guarded
immediate encoding. Rejection emits a diagnostic rather than guessing a payload.

Payload fixtures vary destination and source registers (including RZ), payload
bits and the P3/!P3 selection predicate. Conventional register/immediate fixtures
retain both execution and selection predicates. Positive outputs are parsed by
`ptx_parser::parse_module_checked`, not a parser stub. The predicate-defining
instructions make the fixtures complete for the existing declaration collector.
Synthetic encoding variation checks extraction and operand handling; it does not
prove arbitrary hardware encodings or architectures. No binary decoder or GPU
behavior is added by these tests.

CPU validation on 2026-09-26: the isolated actual-source SASS subset, containing
instruction, cubin_parser, disassembler and lifter with the real PTX parser,
passed all 66 tests (none ignored or filtered). The complete workspace compile
check `cargo test --workspace --no-run --offline --locked` independently failed
at the existing Gemmini manifest target `src/bin/test_ttmlir.rs`, which is absent
from this revision. This subset result does not claim workspace success or GPU
execution. No production lifting logic changed in this test extension.
