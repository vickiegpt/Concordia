use super::*;

fn lift(line: &str) -> SassLiftResult {
    lift_sass_text_to_ptx(
        &format!("Function : float_test\n/*0000*/ {line}\n/*0010*/ EXIT ;\n"),
        SassLiftOptions::default(),
    )
    .unwrap()
}

// Independent finite-input interpreter of the selected PTX conversion. Fixed
// expected bit patterns below test mode selection and packing, not only strings.
fn execute_pack(ptx: &str, a: u32, b: u32) -> u32 {
    let line = ptx
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("cvt."))
        .unwrap();
    let words: Vec<_> = line
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|s| !s.is_empty())
        .collect();
    assert_eq!(words[1..], ["%r5", "%r4", "%r5"]);
    assert!(words[0].ends_with(".bf16x2.f32"));
    let rn = match words[0].split('.').nth(1).unwrap() {
        "rn" => true,
        "rz" => false,
        mode => panic!("unsupported mode {mode}"),
    };
    let relu = words[0].split('.').any(|s| s == "relu");
    let convert = |bits: u32| -> u32 {
        assert!(f32::from_bits(bits).is_finite());
        if relu && f32::from_bits(bits) < 0.0 {
            return 0;
        }
        let upper = bits >> 16;
        let lower = bits & 0xffff;
        upper + u32::from(rn && (lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0)))
    };
    (convert(a) << 16) | convert(b)
}

#[test]
fn bf16_rounding_relu_and_lane_order_have_numeric_counterexamples() {
    // a is in the upper half, b in the lower half, including dst==b alias.
    for (modifier, a, b, expected) in [
        ("", 0x3f80c000, 0xbf800000, 0x3f81bf80),
        (".RZ", 0x3f80c000, 0xbf800000, 0x3f80bf80),
        (".RELU", 0x3f80c000, 0xbf800000, 0x3f810000),
        (".RELU.RZ", 0x3f80c000, 0xbf800000, 0x3f800000),
        ("", 0x3f808000, 0x3f818000, 0x3f803f82), // ties to even in both lanes
        (".RZ", 0xbf80c000, 0x40000000, 0xbf804000),
        ("", 0x00000000, 0x80000000, 0x00008000), // signed zero retained
    ] {
        let result = lift(&format!("F2FP.BF16.F32.PACK_AB{modifier} R5, R4, R5 ;"));
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(execute_pack(&result.ptx, a, b), expected, "{modifier}");
    }
}

#[test]
fn bf16_and_geu_preserve_execution_predicate() {
    let result = lift("@P2 F2FP.RELU.BF16.F32.PACK_AB.RZ R5, R4, R5 ;");
    assert!(result.diagnostics.is_empty());
    assert!(result
        .ptx
        .contains("@%p2 cvt.rz.relu.bf16x2.f32 %r5, %r4, %r5;"));
    let result = lift("@P2 FSETP.GEU.AND P0, PT, R5, R0, PT ;");
    assert!(result.diagnostics.is_empty());
    assert!(result.ptx.contains("@%p2 setp.geu.f32 %p0, %r5, %r0;"));
}

#[test]
fn geu_nan_infinity_zero_and_ordering_follow_unordered_semantics() {
    let result = lift("FSETP.GEU.AND P0, PT, R5, R0, PT ;");
    assert!(result.diagnostics.is_empty());
    let compare = result
        .ptx
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("setp."))
        .unwrap();
    let opcode = compare.split_whitespace().next().unwrap();
    for (a, b, expected) in [
        (f32::NAN, 1.0, true),
        (1.0, f32::NAN, true),
        (f32::NEG_INFINITY, f32::INFINITY, false),
        (f32::INFINITY, f32::INFINITY, true),
        (-0.0, 0.0, true),
        (-1.0, 0.0, false),
        (2.0, 1.0, true),
    ] {
        let actual = match opcode {
            "setp.geu.f32" => a.is_nan() || b.is_nan() || a >= b,
            "setp.ge.f32" => a >= b,
            "setp.eq.f32" => a == b,
            other => panic!("unexpected opcode {other}"),
        };
        assert_eq!(actual, expected, "a={a}, b={b}");
    }
}

#[test]
fn bf16_and_geu_reject_unimplemented_semantics() {
    for line in [
        "F2FP.BF16.F32.PACK_AB.RM R5, R4, R5 ;",
        "F2FP.BF16.F32.PACK_AB.RP R5, R4, R5 ;",
        "F2FP.BF16.F32.PACK_AB.SAT R5, R4, R5 ;",
        "F2FP.BF16.F32.PACK_AB.FTZ R5, R4, R5 ;",
        "F2FP.BF16.F32.PACK_AB R5, -R4, R5 ;",
        "F2FP.BF16.F32.PACK_AB R5, |R4|, R5 ;",
        "F2FP.BF16.F32.PACK_AB R5, R4.H0, R5 ;",
        "F2FP.BF16.F32.PACK_AB R5, R4, c[0x0][0x4] ;",
        "F2FP.BF16.F32.PACK_AB R5, R4 ;",
        "F2FP.BF16.F32.PACK_AB RZ, R4, R5 ;",
        "F2FP.BF16.F32 R5, R4, R5 ;",
        "FSETP.GEU.FTZ.AND P0, PT, R5, R0, PT ;",
        "FSETP.GEU.AND P0, PT, R5, R0, !PT ;",
        "FSETP.GEU.AND P0, PT, R5, R0, P1 ;",
        "FSETP.GEU.OR P0, PT, R5, R0, PT ;",
        "FSETP.GEU.XOR P0, PT, R5, R0, PT ;",
        "FSETP.GEU.AND P0, P1, R5, R0, PT ;",
        "FSETP.GEU.AND P0, PT, -R5, R0, PT ;",
        "FSETP.GEU.AND P0, PT, |R5|, R0, PT ;",
        "FSETP.GEU.AND P0, PT, R5, R0 ;",
    ] {
        let result = lift(line);
        assert_eq!(result.diagnostics.len(), 1, "{line}");
        assert!(!result.ptx.contains("cvt."), "{line}");
        assert!(!result.ptx.contains("setp.geu"), "{line}");
    }
}

#[test]
fn existing_fp16_and_ordered_comparisons_are_unchanged() {
    let result = lift("F2FP.F16.F32.PACK_AB R5, R4, R5 ;");
    assert!(result.diagnostics.is_empty());
    assert!(result.ptx.contains("cvt.rn.f16x2.f32 %r5, %r4, %r5;"));
    for (comparison, ptx_comparison) in [("GE", "ge"), ("EQ", "eq")] {
        let result = lift(&format!("FSETP.{comparison}.AND P0, PT, R5, R0, PT ;"));
        assert!(result.diagnostics.is_empty());
        assert!(result.ptx.contains(&format!("setp.{ptx_comparison}.f32")));
    }
}

#[test]
#[ignore = "requires NVIDIA ptxas with sm_120 support; set HETGPU_TEST_PTXAS"]
fn bf16_generated_ptx_all_modes_assemble_offline() {
    // The project's PTX parser returns Todo for RELU-qualified BF16 conversion;
    // plain RN and RZ are accepted and tested separately.
    // Use the actual NVIDIA assembler for these generated forms, not a stub
    // parser or a handwritten PTX replacement. This does not execute a GPU.
    let assembler = std::env::var_os("HETGPU_TEST_PTXAS").expect("set HETGPU_TEST_PTXAS");
    let temp = tempfile::tempdir().unwrap();
    for (index, modifier) in ["", ".RZ", ".RELU", ".RELU.RZ"].iter().enumerate() {
        let result = lift(&format!("@P2 F2FP.BF16.F32.PACK_AB{modifier} R5, R4, R5 ;"));
        assert!(result.diagnostics.is_empty());
        let input = temp.path().join(format!("bf16_{index}.ptx"));
        std::fs::write(&input, result.ptx).unwrap();
        let output = Command::new(&assembler)
            .arg("-arch=sm_120")
            .arg(&input)
            .arg("-o")
            .arg(temp.path().join(format!("bf16_{index}.cubin")))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{modifier}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn bf16_plain_rn_and_rz_are_accepted_by_the_real_parser() {
    for modifier in ["", ".RZ"] {
        let result = lift(&format!("F2FP.BF16.F32.PACK_AB{modifier} R5, R4, R5 ;"));
        assert!(result.diagnostics.is_empty());
        ptx_parser::parse_module_checked(&result.ptx)
            .expect("ordinary BF16 RN/RZ conversion parses");
    }
}
