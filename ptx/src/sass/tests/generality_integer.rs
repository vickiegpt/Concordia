//! CPU semantic regression checks: execute generated PTX, never helper output.
//! SM120 only: bounded supported-form tests, not a full SASS ISA emulator.
use super::*;

fn lift(line: &str) -> String {
    let result = lift_sass_text_to_ptx(
        &format!("Function : generality\n/*0000*/ {line};\n/*0010*/ EXIT;"),
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
    ptx_parser::parse_module_checked(&result.ptx).expect("generated PTX parses");
    result.ptx
}

fn read(regs: &BTreeMap<String, u64>, name: &str) -> u64 {
    if name.starts_with('%') {
        *regs
            .get(name)
            .unwrap_or_else(|| panic!("uninitialized PTX operand {name}"))
    } else if let Some(hex) = name.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).unwrap()
    } else {
        name.parse().unwrap()
    }
}

// Every source is read before writing an instruction's destination. Unsupported
// opcodes and uninitialized temporaries fail loudly; scratch names are immaterial.
fn execute(ptx: &str, regs: &mut BTreeMap<String, u64>, predicates: &[bool; 7]) {
    let predicate =
        |name: &str| predicates[name.strip_prefix("%p").unwrap().parse::<usize>().unwrap()];
    for raw in ptx.lines().map(str::trim) {
        let mut line = raw;
        if line.starts_with('@') {
            let (guard, body) = line.split_once(' ').unwrap();
            let guard = &guard[1..];
            let (invert, name) = guard
                .strip_prefix('!')
                .map_or((false, guard), |p| (true, p));
            if predicate(name) == invert {
                continue;
            }
            line = body.trim();
        }
        if line.is_empty()
            || line.starts_with('.')
            || line.starts_with("//")
            || line.ends_with(':')
            || matches!(line, "{" | "}" | "ret;")
        {
            continue;
        }
        let w: Vec<_> = line
            .split(|c: char| c.is_whitespace() || ",;{}".contains(c))
            .filter(|s| !s.is_empty())
            .collect();
        if w[0] == "mov.b64" && line.split_once(' ').unwrap().1.trim().starts_with('{') {
            let value = read(regs, w[3]);
            regs.insert(w[1].into(), value as u32 as u64);
            regs.insert(w[2].into(), value >> 32);
            continue;
        }
        let value = match w[0] {
            "mov.b64" => read(regs, w[2]) as u32 as u64 | ((read(regs, w[3]) as u32 as u64) << 32),
            "mov.u32" | "mov.b32" | "cvt.u32.u64" => read(regs, w[2]) as u32 as u64,
            "mov.u64" => read(regs, w[2]),
            "mul.wide.u32" => (read(regs, w[2]) as u32 as u64) * (read(regs, w[3]) as u32 as u64),
            "add.u64" => read(regs, w[2]).wrapping_add(read(regs, w[3])),
            "add.u32" | "add.s32" => read(regs, w[2]).wrapping_add(read(regs, w[3])) as u32 as u64,
            "shr.u64" => read(regs, w[2]) >> read(regs, w[3]),
            "selp.b32" => read(regs, if predicate(w[4]) { w[2] } else { w[3] }),
            other => panic!("unsupported generated opcode {other}: {raw}"),
        };
        regs.insert(w[1].into(), value);
    }
}

fn initial(seed: u32) -> BTreeMap<String, u64> {
    let mut state = seed;
    let mut regs = BTreeMap::new();
    for (prefix, count) in [("r", 255), ("ur", 63)] {
        for i in 0..count {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            regs.insert(format!("%{prefix}{i}"), state as u64);
        }
    }
    regs
}
fn preserved(before: &BTreeMap<String, u64>, after: &BTreeMap<String, u64>, outputs: &[String]) {
    for (name, value) in before {
        if !outputs.contains(name) {
            assert_eq!(after[name], *value, "clobbered {name}");
        }
    }
}

#[test]
fn cs2r_every_supported_pair_and_predicate_direction() {
    for dst in (0..254).step_by(2) {
        for guard in ["", "@P6 ", "@!P6 "] {
            let ptx = lift(&format!("{guard}CS2R R{dst}, SRZ"));
            for enabled in [false, true] {
                let mut predicates = [false; 7];
                predicates[6] = enabled;
                let before = initial(dst + 1);
                let mut after = before.clone();
                execute(&ptx, &mut after, &predicates);
                let active = guard.is_empty() || enabled != guard.contains('!');
                let outputs = [format!("%r{dst}"), format!("%r{}", dst + 1)];
                for name in &outputs {
                    assert_eq!(after[name], if active { 0 } else { before[name] });
                }
                preserved(&before, &after, &outputs);
            }
        }
    }
}

#[test]
fn fsel_preserves_payload_bits_renaming_aliases_and_both_predicates() {
    let payloads = [
        0xffc00000u32,
        0xffc00001,
        0xffd23456,
        0xffe00000,
        0xffffffff,
    ];
    for (dst, src) in [(0, 1), (17, 17), (254, 253)] {
        for bits in payloads {
            for invert in [false, true] {
                let ptx = lift(&format!(
                    "FSEL R{dst}, R{src}, -QNAN, {}P5 /* 0x{bits:08x}{src:02x}{dst:02x}7808 */",
                    if invert { "!" } else { "" }
                ));
                for selected in [false, true] {
                    let before = initial(bits);
                    let mut after = before.clone();
                    let mut predicates = [false; 7];
                    predicates[5] = selected;
                    execute(&ptx, &mut after, &predicates);
                    assert_eq!(
                        after[&format!("%r{dst}")],
                        if selected != invert {
                            before[&format!("%r{src}")]
                        } else {
                            bits as u64
                        }
                    );
                    preserved(&before, &after, &[format!("%r{dst}")]);
                }
            }
        }
    }
    // Register forms have independent execution and selection predicates.
    for (dst, a, b) in [(0, 1, 2), (63, 63, 9), (254, 7, 254)] {
        for guard in ["@P6 ", "@!P6 "] {
            for invert in [false, true] {
                let ptx = lift(&format!(
                    "{guard}FSEL R{dst}, R{a}, R{b}, {}P5",
                    if invert { "!" } else { "" }
                ));
                for flags in 0..4 {
                    let before = initial(flags + 1);
                    let mut after = before.clone();
                    let mut predicates = [false; 7];
                    predicates[5] = flags & 1 != 0;
                    predicates[6] = flags & 2 != 0;
                    execute(&ptx, &mut after, &predicates);
                    let source = if predicates[5] != invert { a } else { b };
                    let expected = if predicates[6] != guard.contains('!') {
                        before[&format!("%r{source}")]
                    } else {
                        before[&format!("%r{dst}")]
                    };
                    assert_eq!(after[&format!("%r{dst}")], expected);
                    preserved(&before, &after, &[format!("%r{dst}")]);
                }
            }
        }
    }
}

#[test]
fn multiply_pair_paths_match_u128_across_aliases_and_bit_boundaries() {
    let boundaries = [0u32, 1, 0x7fffffff, 0x80000000, 0xfffffffe, u32::MAX];
    // Destination aliasing each multiplicand or either addend half, and top GPRs.
    for (dst, a, b, base) in [
        (2, 2, 3, 2),
        (8, 9, 8, 12),
        (20, 4, 5, 20),
        (252, 254, 253, 250),
    ] {
        for mode in ["HI", "WIDE"] {
            for guard in ["", "@P4 ", "@!P4 "] {
                let ptx = lift(&format!(
                    "{guard}IMAD.{mode}.U32 R{dst}, R{a}, R{b}, R{base}"
                ));
                for n in 0..42 {
                    let mut before = initial(0x31415926u32.wrapping_add(n));
                    if n < 6 {
                        before.insert(format!("%r{a}"), boundaries[n as usize] as u64);
                        before.insert(format!("%r{b}"), boundaries[5 - n as usize] as u64);
                        before.insert(format!("%r{base}"), u32::MAX as u64);
                        before.insert(format!("%r{}", base + 1), 0x89abcdef);
                    }
                    let mut after = before.clone();
                    let mut predicates = [false; 7];
                    predicates[4] = n % 2 != 0;
                    let active = guard.is_empty() || predicates[4] != guard.contains('!');
                    let value = (before[&format!("%r{a}")] as u128)
                        * (before[&format!("%r{b}")] as u128)
                        + before[&format!("%r{base}")] as u128
                        + ((before[&format!("%r{}", base + 1)] as u128) << 32);
                    execute(&ptx, &mut after, &predicates);
                    let mut outputs = vec![format!("%r{dst}")];
                    if mode == "WIDE" {
                        outputs.push(format!("%r{}", dst + 1));
                    }
                    for (i, name) in outputs.iter().enumerate() {
                        let shift = if mode == "HI" { 32 } else { 32 * i };
                        assert_eq!(
                            after[name],
                            if active {
                                (value >> shift) as u32 as u64
                            } else {
                                before[name]
                            },
                            "{mode} dst={dst}, vector={n}, guard={guard}"
                        );
                    }
                    preserved(&before, &after, &outputs);
                }
            }
        }
    }
}

#[test]
fn iadd3_all_source_alias_positions_preserve_third_input() {
    for base in [0, 10, 250] {
        for dst in [base, base + 1, base + 2, 254] {
            for guard in ["", "@P3 ", "@!P3 "] {
                let ptx = lift(&format!(
                    "{guard}IADD3 R{dst}, R{base}, R{}, R{}",
                    base + 1,
                    base + 2
                ));
                for n in 0..40 {
                    let mut before = initial(0xabcdef01u32.wrapping_add(n));
                    // Only named SASS operands are live here. The lifter may
                    // allocate a fresh %r above that set as PTX scratch.
                    before.retain(|name, _| {
                        (0..3).any(|offset| *name == format!("%r{}", base + offset))
                            || *name == format!("%r{dst}")
                    });
                    if n == 0 {
                        for offset in 0..3 {
                            before.insert(format!("%r{}", base + offset), u32::MAX as u64);
                        }
                    }
                    let mut after = before.clone();
                    let mut predicates = [false; 7];
                    predicates[3] = n % 2 != 0;
                    let sum: u128 = (0..3)
                        .map(|offset| before[&format!("%r{}", base + offset)] as u128)
                        .sum();
                    execute(&ptx, &mut after, &predicates);
                    let active = guard.is_empty() || predicates[3] != guard.contains('!');
                    assert_eq!(
                        after[&format!("%r{dst}")],
                        if active {
                            sum as u32 as u64
                        } else {
                            before[&format!("%r{dst}")]
                        }
                    );
                    preserved(&before, &after, &[format!("%r{dst}")]);
                }
            }
        }
    }
}

#[test]
fn semantic_oracles_detect_original_failure_mutations() {
    let predicates = [false; 7];
    let mut before = initial(1);
    before.insert("%r6".into(), 0);
    before.insert("%r7".into(), 0);
    before.insert("%r18".into(), 0);
    before.insert("%r19".into(), 7);
    let hi = lift("IMAD.HI.U32 R5, R6, R7, R18");
    let lost_high = hi.replace("{%r18, %r19}", "{%r18, 0}");
    assert_ne!(hi, lost_high, "mutation must change emitted dataflow");
    let mut good = before.clone();
    let mut bad = before.clone();
    execute(&hi, &mut good, &predicates);
    execute(&lost_high, &mut bad, &predicates);
    let oracle = ((0u128 * 0 + (7u128 << 32)) >> 32) as u64;
    assert_eq!(good["%r5"], oracle);
    assert_ne!(bad["%r5"], oracle);

    let pair = lift("@P6 CS2R R20, SRZ");
    let unguarded = pair.replace("@%p6 ", "");
    assert_ne!(pair, unguarded);
    good = before.clone();
    bad = before.clone();
    execute(&pair, &mut good, &predicates);
    execute(&unguarded, &mut bad, &predicates);
    assert_eq!(good["%r21"], before["%r21"]);
    assert_ne!(bad["%r21"], before["%r21"]);
    let missing_high = pair
        .lines()
        .filter(|line| !line.trim().starts_with("@%p6 mov.u32 %r21,"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_ne!(pair, missing_high);
    let mut active = predicates;
    active[6] = true;
    bad = before.clone();
    execute(&missing_high, &mut bad, &active);
    assert_ne!(bad["%r21"], 0);

    let select = lift("FSEL R5, R6, R19, P2");
    let swapped = select.replace("%r5, %r6, %r19,", "%r5, %r19, %r6,");
    assert_ne!(select, swapped);
    good = before.clone();
    bad = before;
    execute(&select, &mut good, &predicates);
    execute(&swapped, &mut bad, &predicates);
    assert_eq!(good["%r5"], 7);
    assert_ne!(bad["%r5"], 7);
}

#[test]
fn multiply_zero_immediate_and_uniform_forms_match_u128() {
    for (mode, a, b, base) in [
        ("HI", "R254", "0xffffffff", "R252"),
        ("HI", "RZ", "R2", "R252"),
        ("HI", "R2", "0x80000000", "RZ"),
        ("WIDE", "R254", "UR62", "R252"),
        ("WIDE", "R2", "URZ", "R252"),
        ("WIDE", "R2", "0xffffffff", "RZ"),
    ] {
        for guard in ["@P1 ", "@!P1 "] {
            let ptx = lift(&format!("{guard}IMAD.{mode}.U32 R252, {a}, {b}, {base}"));
            for n in 0..32 {
                let before = initial(0x12345678u32.wrapping_add(n));
                let scalar = |name: &str| -> u128 {
                    if matches!(name, "RZ" | "URZ") {
                        0
                    } else if name.starts_with("0x") {
                        u128::from_str_radix(&name[2..], 16).unwrap()
                    } else {
                        before[&format!("%{}", name.to_lowercase())] as u128
                    }
                };
                let base_value = if base == "RZ" {
                    0
                } else {
                    scalar("R252") | (scalar("R253") << 32)
                };
                let expected = scalar(a) * scalar(b) + base_value;
                let mut predicates = [false; 7];
                predicates[1] = n % 2 != 0;
                let active = predicates[1] != guard.contains('!');
                let mut after = before.clone();
                execute(&ptx, &mut after, &predicates);
                let outputs = if mode == "HI" {
                    vec!["%r252".to_owned()]
                } else {
                    vec!["%r252".to_owned(), "%r253".to_owned()]
                };
                for (i, name) in outputs.iter().enumerate() {
                    let shift = if mode == "HI" { 32 } else { i * 32 };
                    assert_eq!(
                        after[name],
                        if active {
                            (expected >> shift) as u32 as u64
                        } else {
                            before[name]
                        }
                    );
                }
                preserved(&before, &after, &outputs);
            }
        }
    }
}
