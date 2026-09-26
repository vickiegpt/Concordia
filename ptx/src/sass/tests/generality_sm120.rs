//! Optional SM120 compiler/disassembler integration, entirely offline.
//! Independent PTX arithmetic/conversion -> NVIDIA SASS -> selected lifter output
//! -> NVIDIA assembly. This is not a whole-kernel or GPU execution equivalence test.
use super::*;

fn assemble(assembler: &std::ffi::OsStr, input: &Path, output: &Path) {
    let result = Command::new(assembler)
        .arg("-arch=sm_120")
        .arg(input)
        .arg("-o")
        .arg(output)
        .output()
        .expect("execute configured ptxas");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
#[ignore = "offline SM120 toolchain; set HETGPU_TEST_PTXAS and HETGPU_TEST_CUOBJDUMP"]
fn independent_sm120_ptx_to_sass_to_lifted_ptx_assembles() {
    let assembler = std::env::var_os("HETGPU_TEST_PTXAS").expect("set HETGPU_TEST_PTXAS");
    let dumper = std::env::var_os("HETGPU_TEST_CUOBJDUMP").expect("set HETGPU_TEST_CUOBJDUMP");
    let temp = tempfile::tempdir().unwrap();
    // This is a separate algebraic specification, not PTX copied from the lifter.
    // Keeping parameters runtime-valued prevents constant folding of the target.
    let hi = r#".version 8.8
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 output, .param .u32 a, .param .u32 b, .param .u64 c) {
.reg .u64 %out, %add, %product, %sum, %high;
.reg .u32 %a, %b, %d;
ld.param.u64 %out, [output];
ld.param.u32 %a, [a];
ld.param.u32 %b, [b];
ld.param.u64 %add, [c];
mul.wide.u32 %product, %a, %b;
add.u64 %sum, %product, %add;
shr.u64 %high, %sum, 32;
cvt.u32.u64 %d, %high;
st.global.u32 [%out], %d;
ret;
}"#;
    // Standalone equivalent of the original SM120 pack_rz_relu reference probe.
    // RELU's local PTX parser Todo remains a separate consumer limitation.
    let bf16 = r#".version 8.8
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 output, .param .u32 a, .param .u32 b) {
.reg .u64 %out;
.reg .b32 %a, %b, %d;
ld.param.u64 %out, [output];
ld.param.b32 %a, [a];
ld.param.b32 %b, [b];
cvt.rz.relu.bf16x2.f32 %d, %a, %b;
st.global.b32 [%out], %d;
ret;
}"#;
    for (name, reference) in [("hi", hi), ("bf16", bf16)] {
        let input = temp.path().join(format!("{name}.ptx"));
        let cubin = temp.path().join(format!("{name}.cubin"));
        std::fs::write(&input, reference).unwrap();
        assemble(&assembler, &input, &cubin);
        let output = Command::new(&dumper)
            .arg("--dump-sass")
            .arg(&cubin)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let sass = String::from_utf8(output.stdout).unwrap();
        // Extract actual parsed target instructions; never reconstruct their
        // operands from expected scratch names or a previously captured SASS file.
        let selected: Vec<_> = sass
            .lines()
            .filter_map(|line| {
                let inst = TextDisassemblyParser::parse_instruction_line(line)?;
                let matches = if name == "hi" {
                    inst.opcode == "IMAD" && inst.modifiers.iter().any(|m| m == "HI")
                } else {
                    inst.opcode == "F2FP" && inst.modifiers.iter().any(|m| m == "BF16")
                };
                matches.then_some((line, inst))
            })
            .collect();
        assert_eq!(
            selected.len(),
            1,
            "compiler target changed; inspect actual SASS, do not infer a pass:\n{sass}"
        );
        let (line, inst) = &selected[0];
        let result = lift_sass_text_to_ptx(
            &format!("Function : recovered\n{line}\n/*fff0*/ EXIT;"),
            SassLiftOptions {
                sm_version: 120,
                ..SassLiftOptions::default()
            },
        )
        .unwrap();
        println!("CASE {name}\nREFERENCE:\n{reference}\nSELECTED SASS:\n{line}\nLIFTED PTX:\n{}\nDIAGNOSTICS: {:?}", result.ptx, result.diagnostics);
        assert!(
            result.diagnostics.is_empty(),
            "compiler emitted unsupported target form: {:?}",
            result.diagnostics
        );
        if name == "hi" {
            let SassOperand::Register(base) = &inst.src_operands[2] else {
                panic!("expected independent reference's register addend")
            };
            assert_eq!(base.prefix, "R");
            assert_eq!(base.number % 2, 0);
            assert!(
                result
                    .ptx
                    .contains(&format!("{{%r{}, %r{}}}", base.number, base.number + 1)),
                "full actual addend pair must be consumed"
            );
            assert!(result.ptx.contains("mul.wide.u32"));
            assert!(result.ptx.contains("shr.u64"));
            assert!(!result.ptx.contains("mad.hi.u32"));
            ptx_parser::parse_module_checked(&result.ptx).expect("HI PTX parser");
        } else {
            assert!(inst.modifiers.iter().any(|m| m == "RZ"));
            assert!(inst.modifiers.iter().any(|m| m == "RELU"));
            assert!(result.ptx.contains("cvt.rz.relu.bf16x2.f32"));
        }
        let lifted = temp.path().join(format!("{name}-lifted.ptx"));
        std::fs::write(&lifted, result.ptx).unwrap();
        assemble(
            &assembler,
            &lifted,
            &temp.path().join(format!("{name}-lifted.cubin")),
        );
    }
}
