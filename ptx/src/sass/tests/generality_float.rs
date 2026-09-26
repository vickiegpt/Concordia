//! Bounded SM120 CPU checks of emitted PTX semantics, not GPU differential execution.
//! References (algorithms below independently written; no external code copied):
//! https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
//! https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp
//! https://github.com/NVIDIA/cutlass/blob/main/test/unit/core/numeric_conversion.cu
//! CUTLASS motivates checking numeric conversions across type/lane combinations.
//! Exhaustive small-format and midpoint enumeration below is our added CPU check;
//! external conversion code is NOT used as an oracle for PTX NaN payloads.
use super::*;

fn lift_float(line: &str) -> SassLiftResult {
    let result = lift_sass_text_to_ptx(
        &format!("Function : general_float\n/*0000*/ {line}\n/*0010*/ EXIT ;\n"),
        SassLiftOptions {
            sm_version: 120,
            ..SassLiftOptions::default()
        },
    )
    .unwrap();
    assert!(
        result.diagnostics.is_empty(),
        "{line}: {:?}",
        result.diagnostics
    );
    result
}

#[derive(Clone, Debug)]
struct Instruction {
    guard: Option<String>,
    opcode: String,
    operands: Vec<String>,
}

fn instruction(ptx: &str, prefix: &str) -> Instruction {
    let mut selected = Vec::new();
    for line in ptx.lines().map(str::trim) {
        let mut words = line.split_whitespace();
        let first = match words.next() {
            Some(w) => w,
            None => continue,
        };
        let (guard, op) = if first.starts_with('@') {
            (Some(first.to_owned()), words.next().unwrap_or(""))
        } else {
            (None, first)
        };
        if op.starts_with(prefix) {
            selected.push(Instruction {
                guard,
                opcode: op.to_owned(),
                operands: words
                    .collect::<String>()
                    .trim_end_matches(';')
                    .split(',')
                    .map(str::to_owned)
                    .collect(),
            });
        }
    }
    assert_eq!(selected.len(), 1, "{ptx}");
    selected.remove(0)
}

fn nan32(bits: u32) -> bool {
    bits & 0x7fff_ffff > 0x7f80_0000
}
fn nan16(bits: u16) -> bool {
    bits & 0x7fff > 0x7f80
}

// Instruction model uses bit rounding; the reference below instead measures
// distances between exact f64 representations of adjacent BF16 numbers.
fn convert_model(bits: u32, rn: bool, relu: bool) -> u16 {
    if nan32(bits) {
        return 0x7fff;
    } // payload is deliberately not asserted
    if relu && f32::from_bits(bits) < 0.0 {
        return 0;
    }
    if bits & 0x7fff_ffff == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    let bias = if rn { 0x7fff + ((bits >> 16) & 1) } else { 0 };
    (bits.wrapping_add(bias) >> 16) as u16
}

fn reference(bits: u32, rn: bool, relu: bool) -> u16 {
    if nan32(bits) {
        return 0x7fc0;
    }
    let value = f32::from_bits(bits);
    if relu && value < 0.0 {
        return 0;
    }
    let sign = ((bits >> 16) & 0x8000) as u16;
    if !value.is_finite() || value == 0.0 {
        return (bits >> 16) as u16;
    }
    let magnitude = f64::from(value).abs();
    // Binary search the representable nonnegative finite BF16 values.
    let (mut lo, mut hi) = (0u16, 0x7f7fu16);
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if f64::from(f32::from_bits(u32::from(mid) << 16)) <= magnitude {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if !rn {
        return sign | lo;
    }
    let lower = f64::from(f32::from_bits(u32::from(lo) << 16));
    // Conceptual next finite value 2^128 determines overflow rounding.
    let upper = if lo == 0x7f7f {
        2f64.powi(128)
    } else {
        f64::from(f32::from_bits(u32::from(lo + 1) << 16))
    };
    let rounded = if magnitude - lower > upper - magnitude
        || (magnitude - lower == upper - magnitude && lo & 1 != 0)
    {
        lo + 1
    } else {
        lo
    };
    sign | rounded
}

fn same_bf16(actual: u16, expected: u16, bits: u32) {
    if nan16(expected) {
        assert!(nan16(actual), "NaN input {bits:08x} became {actual:04x}");
    } else {
        assert_eq!(actual, expected, "input {bits:08x}");
    }
}

fn execute_pack(inst: &Instruction, regs: &mut [u32; 256], predicates: &[bool; 8]) {
    if let Some(guard) = &inst.guard {
        let negated = guard.starts_with("@!");
        let index: usize = guard
            .trim_start_matches('@')
            .trim_start_matches('!')
            .trim_start_matches("%p")
            .parse()
            .unwrap();
        if predicates[index] == negated {
            return;
        }
    }
    let rn = match inst.opcode.as_str() {
        "cvt.rn.bf16x2.f32" | "cvt.rn.relu.bf16x2.f32" => true,
        "cvt.rz.bf16x2.f32" | "cvt.rz.relu.bf16x2.f32" => false,
        op => panic!("unexpected opcode {op}"),
    };
    assert_eq!(inst.operands.len(), 3);
    let read = |operand: &str| {
        if operand == "0" {
            0
        } else {
            regs[operand.trim_start_matches("%r").parse::<usize>().unwrap()]
        }
    };
    let (a, b) = (read(&inst.operands[1]), read(&inst.operands[2]));
    let relu = inst.opcode.contains(".relu.");
    let value =
        (u32::from(convert_model(a, rn, relu)) << 16) | u32::from(convert_model(b, rn, relu));
    let dst = inst.operands[0]
        .trim_start_matches("%r")
        .parse::<usize>()
        .unwrap();
    regs[dst] = value;
}

#[test]
fn every_bf16_encoding_widened_and_roundtripped_through_emitted_modes() {
    for (modifier, rn, relu) in [
        ("", true, false),
        (".RZ", false, false),
        (".RELU", true, true),
        (".RELU.RZ", false, true),
    ] {
        let result = lift_float(&format!("F2FP.BF16.F32.PACK_AB{modifier} R19, R8, R27 ;"));
        let inst = instruction(&result.ptx, "cvt.");
        let mut regs = [0; 256];
        for code in 0..=u16::MAX {
            // Distinct lanes include every sign, exponent, mantissa and NaN payload.
            let (a, b) = (u32::from(code) << 16, u32::from(code ^ 0x9235) << 16);
            regs[8] = a;
            regs[27] = b;
            execute_pack(&inst, &mut regs, &[true; 8]);
            same_bf16((regs[19] >> 16) as u16, reference(a, rn, relu), a);
            same_bf16(regs[19] as u16, reference(b, rn, relu), b);
        }
    }
}

#[test]
fn all_finite_bf16_midpoints_and_neighbors_select_correct_rounding() {
    // 32,640 intervals x both signs x three neighboring f32 values x two modes.
    // Lift only twice; includes underflow, normal/subnormal transition, overflow.
    for (modifier, rn) in [("", true), (".RZ", false)] {
        let result = lift_float(&format!("F2FP.BF16.F32.PACK_AB{modifier} R9, R1, R2 ;"));
        let inst = instruction(&result.ptx, "cvt.");
        let mut regs = [0; 256];
        for lower in 0u32..=0x7f7f {
            for tail in [0x7fff, 0x8000, 0x8001] {
                let bits = (lower << 16) | tail;
                regs[1] = bits;
                regs[2] = bits | 0x8000_0000;
                execute_pack(&inst, &mut regs, &[true; 8]);
                same_bf16(
                    (regs[9] >> 16) as u16,
                    reference(regs[1], rn, false),
                    regs[1],
                );
                same_bf16(regs[9] as u16, reference(regs[2], rn, false), regs[2]);
            }
        }
    }
}

#[test]
fn packed_conversion_renaming_aliasing_predicate_and_mode_mutations_are_observable() {
    for (dst, a, b) in [(3, 4, 5), (4, 4, 5), (5, 4, 5), (199, 17, 253)] {
        for negated in [false, true] {
            for (modifier, rn, relu) in [
                ("", true, false),
                (".RZ", false, false),
                (".RELU", true, true),
                (".RELU.RZ", false, true),
            ] {
                let guard = if negated { "@!P3" } else { "@P3" };
                let result = lift_float(&format!(
                    "{guard} F2FP.BF16.F32.PACK_AB{modifier} R{dst}, R{a}, R{b} ;"
                ));
                let inst = instruction(&result.ptx, "cvt.");
                assert_eq!(
                    inst.guard.as_deref(),
                    Some(if negated { "@!%p3" } else { "@%p3" })
                );
                for enabled in [false, true] {
                    let mut regs = [0xdead_beef; 256];
                    regs[a] = 0x3f81_8000;
                    regs[b] = 0xbf80_c000;
                    let before = regs;
                    let mut predicates = [false; 8];
                    predicates[3] = enabled != negated;
                    execute_pack(&inst, &mut regs, &predicates);
                    let mut expected = before;
                    if enabled {
                        expected[dst] = (u32::from(reference(before[a], rn, relu)) << 16)
                            | u32::from(reference(before[b], rn, relu));
                    }
                    assert_eq!(regs, expected);
                    if enabled {
                        let mut swapped = inst.clone();
                        swapped.operands.swap(1, 2);
                        let mut mutated = before;
                        execute_pack(&swapped, &mut mutated, &predicates);
                        assert_ne!(mutated[dst], expected[dst], "lane mutation must be killed");
                        let mut wrong_mode = inst.clone();
                        wrong_mode.opcode = if rn {
                            inst.opcode.replace(".rn.", ".rz.")
                        } else {
                            inst.opcode.replace(".rz.", ".rn.")
                        };
                        mutated = before;
                        execute_pack(&wrong_mode, &mut mutated, &predicates);
                        assert_ne!(
                            mutated[dst], expected[dst],
                            "rounding mutation must be killed"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn bf16_real_parser_acceptance_and_relu_consumer_boundary_are_explicit() {
    for modifier in ["", ".RZ", ".RELU", ".RELU.RZ"] {
        let result = lift_float(&format!("F2FP.BF16.F32.PACK_AB{modifier} R9, R1, R2 ;"));
        let parsed = ptx_parser::parse_module_checked(&result.ptx);
        if modifier.contains("RELU") {
            let errors = parsed
                .err()
                .expect("RELU consumer support must not be claimed");
            assert_eq!(errors.len(), 1, "{errors:?}");
            assert!(matches!(errors[0], ptx_parser::PtxError::Todo));
        } else {
            assert!(parsed.is_ok(), "{:?}", parsed.err());
        }
    }
}

#[test]
fn geu_cartesian_float_classes_match_integer_order_oracle() {
    let values = [
        0u32,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fa1_2345,
        0xffa1_2345,
        0x7fff_ffff,
        0xffff_ffff,
    ];
    for (a_reg, b_reg, pred) in [(0, 5, 0), (251, 17, 6)] {
        let result = lift_float(&format!(
            "FSETP.GEU.AND P{pred}, PT, R{a_reg}, R{b_reg}, PT ;"
        ));
        ptx_parser::parse_module_checked(&result.ptx).unwrap();
        let inst = instruction(&result.ptx, "setp.");
        assert_eq!(
            inst.operands,
            [
                format!("%p{pred}"),
                format!("%r{a_reg}"),
                format!("%r{b_reg}")
            ]
        );
        for a in values {
            for b in values {
                // Bitwise total order except signed zero and unordered NaNs.
                let key = |x: u32| {
                    if x & 0x8000_0000 != 0 {
                        !x
                    } else {
                        x ^ 0x8000_0000
                    }
                };
                let expected = nan32(a)
                    || nan32(b)
                    || (a & 0x7fff_ffff == 0 && b & 0x7fff_ffff == 0)
                    || key(a) >= key(b);
                let (af, bf) = (f32::from_bits(a), f32::from_bits(b));
                let actual = match inst.opcode.as_str() {
                    "setp.geu.f32" => af.is_nan() || bf.is_nan() || af >= bf,
                    "setp.ge.f32" => af >= bf,
                    other => panic!("unexpected {other}"),
                };
                assert_eq!(actual, expected, "a={a:08x}, b={b:08x}");
            }
        }
    }
}

#[test]
fn bf16_fp32_special_classes_and_tiny_nan_payloads_remain_distinct() {
    let values = [
        0u32,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fc0_0001,
        0xffc0_0001,
        0x7fff_ffff,
        0xffff_ffff,
    ];
    for (modifier, rn, relu) in [
        ("", true, false),
        (".RZ", false, false),
        (".RELU", true, true),
        (".RELU.RZ", false, true),
    ] {
        let result = lift_float(&format!("F2FP.BF16.F32.PACK_AB{modifier} R9, R1, R2 ;"));
        let inst = instruction(&result.ptx, "cvt.");
        let mut regs = [0; 256];
        for a in values {
            for b in values {
                regs[1] = a;
                regs[2] = b;
                execute_pack(&inst, &mut regs, &[true; 8]);
                same_bf16((regs[9] >> 16) as u16, reference(a, rn, relu), a);
                same_bf16(regs[9] as u16, reference(b, rn, relu), b);
            }
        }
    }
}

#[test]
fn geu_predicate_renaming_and_inversion_preserve_disabled_destination() {
    for negated in [false, true] {
        for (dst, guard) in [(0, 2), (6, 5), (3, 3)] {
            let sass_guard = if negated {
                format!("@!P{guard}")
            } else {
                format!("@P{guard}")
            };
            let result = lift_float(&format!(
                "{sass_guard} FSETP.GEU.AND P{dst}, PT, R31, R63, PT ;"
            ));
            let inst = instruction(&result.ptx, "setp.");
            assert_eq!(inst.opcode, "setp.geu.f32");
            assert_eq!(
                inst.operands,
                [format!("%p{dst}"), "%r31".into(), "%r63".into()]
            );
            assert_eq!(
                inst.guard,
                Some(if negated {
                    format!("@!%p{guard}")
                } else {
                    format!("@%p{guard}")
                })
            );
            for guard_value in [false, true] {
                for (a, b) in [(0x7f80_0001, 0u32), (0xbf80_0000, 0x3f80_0000)] {
                    let mut predicates = [false; 8];
                    predicates[dst] = true;
                    predicates[guard] = guard_value;
                    let before = predicates;
                    let emitted_guard = inst.guard.as_ref().unwrap();
                    let inverted = emitted_guard.starts_with("@!");
                    let index: usize = emitted_guard
                        .trim_start_matches('@')
                        .trim_start_matches('!')
                        .trim_start_matches("%p")
                        .parse()
                        .unwrap();
                    if predicates[index] != inverted {
                        let destination: usize =
                            inst.operands[0].trim_start_matches("%p").parse().unwrap();
                        predicates[destination] =
                            f32::from_bits(a).is_nan() || f32::from_bits(a) >= f32::from_bits(b);
                    }
                    let mut expected = before;
                    if guard_value != negated {
                        expected[dst] = nan32(a);
                    }
                    assert_eq!(predicates, expected);
                }
            }
        }
    }
}
