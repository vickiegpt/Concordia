use super::*;

fn lift(line: &str) -> SassLiftResult {
    lift_sass_text_to_ptx(
        &format!("Function : imad_test\n/*0000*/ {line}\n/*0010*/ EXIT ;\n"),
        SassLiftOptions::default(),
    )
    .unwrap()
}

// Interpret only the emitted integer PTX subset, not the SASS implementation.
// The oracle below uses a separate u128 expression and exercises the generated
// operand joins, register aliases, arithmetic width and execution predicate.
fn execute(ptx: &str, registers: &mut BTreeMap<String, u64>, predicate: bool) {
    for line in ptx.lines().map(str::trim) {
        if line.is_empty()
            || line.starts_with('.')
            || line.starts_with("//")
            || line.starts_with("L_")
            || matches!(line, "{" | "}" | "ret;")
        {
            continue;
        }
        let mut words: Vec<&str> = line
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '{' | '}'))
            .filter(|s| !s.is_empty())
            .collect();
        if words[0].starts_with('@') {
            assert_eq!(words.remove(0), "@%p0");
            if !predicate {
                continue;
            }
        }
        let read = |v: &str| {
            registers
                .get(v)
                .copied()
                .unwrap_or_else(|| v.parse().unwrap())
        };
        let value = match words[0] {
            "mul.wide.u32" => (read(words[2]) as u32 as u64) * (read(words[3]) as u32 as u64),
            "mov.b64" => (read(words[2]) as u32 as u64) | ((read(words[3]) as u32 as u64) << 32),
            "mov.u64" => read(words[2]),
            "add.u64" => read(words[2]).wrapping_add(read(words[3])),
            "shr.u64" => read(words[2]) >> read(words[3]),
            "cvt.u32.u64" => read(words[2]) as u32 as u64,
            opcode => panic!("unexpected generated instruction {opcode}: {line}"),
        };
        registers.insert(words[1].to_owned(), value);
    }
}

#[test]
fn imad_hi_pair_arithmetic_matches_independent_wide_oracle() {
    let mut vectors = vec![
        (0, 0, 0, 7),        // Previous mad.hi mapping returned 0 instead of 7.
        (u32::MAX, 1, 1, 0), // Carry from low product plus low addend.
        (u32::MAX, u32::MAX, u32::MAX, u32::MAX), // Wrap past 64 bits.
        (0x80000000, 2, 0, 0),
        (1, 1, 0, 0),
        (0x87654321, 0xfedcba98, 0x11223344, 0xaabbccdd),
    ];
    let mut state = 0x12345678u32;
    for _ in 0..128 {
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        vectors.push((next(), next(), next(), next()));
    }
    for dst in [5, 6, 18, 19] {
        let result = lift(&format!("IMAD.HI.U32 R{dst}, R6, R7, R18 ;"));
        assert!(result.diagnostics.is_empty());
        assert!(result.ptx.contains(".reg .b32 %r<20>;"));
        ptx_parser::parse_module_checked(&result.ptx).expect("IMAD.HI emitted PTX parses");
        for &(a, b, lo, hi) in &vectors {
            let mut registers = BTreeMap::from([
                ("%r6".into(), a as u64),
                ("%r7".into(), b as u64),
                ("%r18".into(), lo as u64),
                ("%r19".into(), hi as u64),
            ]);
            execute(&result.ptx, &mut registers, true);
            let expected =
                (((a as u128) * (b as u128) + ((hi as u128) << 32) + lo as u128) >> 32) as u32;
            assert_eq!(
                registers[&format!("%r{dst}")] as u32,
                expected,
                "dst={dst}, a={a:#x}, b={b:#x}, pair={hi:#x}:{lo:#x}"
            );
        }
    }
}

#[test]
fn imad_hi_zero_immediate_and_predicated_off_preserve_values() {
    let zero = lift("IMAD.HI.U32 R5, R6, 0xffffffff, RZ ;");
    assert!(zero.diagnostics.is_empty());
    let mut registers = BTreeMap::from([("%r6".into(), 0xffffffff)]);
    execute(&zero.ptx, &mut registers, true);
    assert_eq!(registers["%r5"], 0xfffffffe);
    let pred = lift("@P0 IMAD.HI.U32 R19, R6, R7, R18 ;");
    assert!(pred.diagnostics.is_empty());
    let initial = BTreeMap::from([
        ("%r6".into(), 0),
        ("%r7".into(), 0),
        ("%r18".into(), 0),
        ("%r19".into(), 7),
    ]);
    let mut registers = initial.clone();
    execute(&pred.ptx, &mut registers, false);
    assert_eq!(registers, initial);
    execute(&pred.ptx, &mut registers, true);
    assert_eq!(registers["%r19"], 7);
}

#[test]
fn imad_hi_rejects_unsupported_layouts_without_dropping_operands() {
    for line in [
        "IMAD.HI R5, R6, R7, R18 ;",
        "IMAD.HI.U32.X R5, R6, R7, R18, P0 ;",
        "IMAD.HI.U32.SAT R5, R6, R7, R18 ;",
        "IMAD.HI.U32 R5, R6, c[0x0][0x4], R18 ;",
        "IMAD.HI.U32 R5, -R6, R7, R18 ;",
        "IMAD.HI.U32 R5, |R6|, R7, R18 ;",
        "IMAD.HI.U32 R5, R6.H0, R7, R18 ;",
        "IMAD.HI.U32 R5, R6, R7, R19 ;",
        "IMAD.HI.U32 R5, R6, R7, R254 ;",
        "IMAD.HI.U32 R5, R6, R7, R256 ;",
        "IMAD.HI.U32 R5, R6, R7, UR4 ;",
        "IMAD.HI.U32 R5, R6, R7, 1 ;",
        "IMAD.HI.U32 R5, R6, R7 ;",
        "IMAD.HI.U32 RZ, R6, R7, R18 ;",
    ] {
        let result = lift(line);
        assert_eq!(result.diagnostics.len(), 1, "{line}");
        assert!(!result.ptx.contains("mul.wide.u32"), "{line}");
        assert!(!result.ptx.contains("mad.hi.u32"), "{line}");
    }
}

#[test]
fn imad_hi_keeps_ordinary_and_wide_neighbors() {
    assert!(lift("IMAD.U32 R5, R6, R7, R18 ;")
        .ptx
        .contains("mad.lo.u32"));
    let wide = lift("IMAD.WIDE.U32 R4, R6, R7, R18 ;");
    assert!(wide.ptx.contains("mul.wide.u32 %rd15, %r6, %r7;"));
    assert!(wide.ptx.contains("add.u64 %rd4, %rd18, %rd15;"));
    assert!(!wide.ptx.contains("%imad_product"));
}

#[test]
#[ignore = "requires NVIDIA ptxas with sm_120 support; set HETGPU_TEST_PTXAS"]
fn imad_hi_generated_ptx_assembles_offline() {
    let assembler = std::env::var_os("HETGPU_TEST_PTXAS").expect("set HETGPU_TEST_PTXAS");
    let text = "Function : imad_assembly\n\
        /*0000*/ IMAD.HI.U32 R5, R6, R7, R18 ;\n\
        /*0010*/ IMAD.HI.U32 R18, R6, R7, R18 ;\n\
        /*0020*/ @P0 IMAD.HI.U32 R19, R6, R7, R18 ;\n\
        /*0030*/ IMAD.HI.U32 R6, R6, 0xffffffff, RZ ;\n\
        /*0040*/ EXIT ;\n";
    let result = lift_sass_text_to_ptx(text, SassLiftOptions::default()).unwrap();
    assert!(result.diagnostics.is_empty());
    ptx_parser::parse_module_checked(&result.ptx).expect("scoped temporaries parse");
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("imad.ptx");
    std::fs::write(&input, result.ptx).unwrap();
    let output = Command::new(assembler)
        .arg("-arch=sm_120")
        .arg(&input)
        .arg("-o")
        .arg(temp.path().join("imad.cubin"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
