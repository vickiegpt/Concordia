use ptx::sass::fuzz::{run_sass_lifter_fuzzer, SassLifterFuzzConfig};
use ptx::{
    lift_cubin_to_ptx_with_cuobjdump, lift_sass_text_to_ptx, SassLiftOptions, SassOperand,
    SassRegister, TextDisassemblyParser,
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[test]
fn sass_lifter_fuzzer_is_deterministic_and_generates_parseable_ptx() {
    let config = SassLifterFuzzConfig {
        seed: 0x5a55_1200,
        cases: 24,
        max_instructions: 12,
        sm_version: 120,
        parse_lifted_ptx: true,
    };

    let first = run_sass_lifter_fuzzer(config.clone()).expect("fuzz run should pass");
    let second = run_sass_lifter_fuzzer(config).expect("fuzz run should be repeatable");

    assert_eq!(first, second);
    assert_eq!(first.cases, 24);
    assert!(first.instructions >= 24);
    assert_eq!(first.lift_diagnostics, 0);
    assert_eq!(first.parse_failures, 0);
}

#[test]
fn sass_lifter_fuzzer_rejects_empty_runs() {
    let config = SassLifterFuzzConfig {
        seed: 1,
        cases: 0,
        max_instructions: 8,
        sm_version: 120,
        parse_lifted_ptx: true,
    };

    let error = run_sass_lifter_fuzzer(config).expect_err("zero cases should be rejected");
    assert!(error
        .to_string()
        .contains("cases must be greater than zero"));
}

#[test]
fn text_parser_strips_real_sm120_cuobjdump_annotations() {
    let line = "/*00b0*/                   IMAD.WIDE.U32 R2, R7, 0x4, R2              &req={0}         ?WAIT6_END_GROUP;  /* 0x0000000407027825 */";

    let inst = TextDisassemblyParser::parse_instruction_line(line)
        .expect("real cuobjdump instruction line should parse");

    assert_eq!(inst.address, 0x00b0);
    assert_eq!(inst.opcode, "IMAD");
    assert!(inst.modifiers.contains(&"WIDE".to_string()));
    assert!(inst.modifiers.contains(&"U32".to_string()));
    assert_eq!(inst.dest_operands.len(), 1);
    assert_eq!(inst.src_operands.len(), 3);
    assert_eq!(inst.src_operands[1], SassOperand::Immediate(0x4));
    assert_eq!(
        inst.src_operands[2],
        SassOperand::Register(SassRegister::new("R", 2))
    );
}

#[cfg(unix)]
#[test]
fn cuobjdump_backed_lifter_uses_external_text_disassembly() {
    let temp_dir = tempfile::tempdir().expect("tempdir should be created");
    let tool_path = temp_dir.path().join("fake-cuobjdump");
    std::fs::write(
        &tool_path,
        r#"#!/bin/sh
cat <<'EOF'
Fatbin elf code:
================
arch = sm_120
code version = [1,7]
host = linux
compile_size = 64bit

	code for sm_120
		Function : vector_add
.headerflags @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   EXIT ;
EOF
"#,
    )
    .expect("fake cuobjdump should be written");
    let mut permissions = std::fs::metadata(&tool_path)
        .expect("fake cuobjdump metadata should be readable")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&tool_path, permissions).expect("fake cuobjdump should be executable");

    let result = lift_cubin_to_ptx_with_cuobjdump(
        b"not actually an elf",
        SassLiftOptions {
            sm_version: 120,
            kernel_name: "vector_add".to_string(),
            include_sass_comments: true,
            emit_unsupported_comments: true,
        },
        &tool_path,
    )
    .expect("cuobjdump-backed lifting should use the external text stream");

    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.ptx.contains(".target sm_120"));
    assert!(result.ptx.contains(".visible .entry vector_add()"));
    assert!(result.ptx.contains("mov.u32 %r0, %tid.x;"));
}

#[test]
fn text_lifter_handles_real_sm120_roundtrip_integer_pattern() {
    let text = r#"Function : int_add
        /*0000*/                   LDC R1, c[0x0][0x37c]                      &wr=0x0          ?trans1;           /* 0x0000df00ff017b82 */
        /*0010*/                   S2R R7, SR_TID.X                           &wr=0x1          ?trans7;           /* 0x0000000000077919 */
        /*0020*/                   S2UR UR4, SR_CTAID.X                       &wr=0x1          ?trans1;           /* 0x00000000000479c3 */
        /*0030*/                   LDCU UR5, c[0x0][0x390]                    &wr=0x2          ?trans7;           /* 0x00007200ff0577ac */
        /*0040*/                   LDC R0, c[0x0][0x360]                      &wr=0x1          ?trans2;           /* 0x0000d800ff007b82 */
        /*0050*/                   IMAD R7, R0, UR4, R7                       &req={1}         ?WAIT5_END_GROUP;  /* 0x0000000400077c24 */
        /*0060*/                   ISETP.GE.U32.AND P0, PT, R7, UR5, PT       &req={2}         ?WAIT13_END_GROUP; /* 0x0000000507007c0c */
        /*0070*/               @P0 EXIT                                       &req={0}         ?trans5;           /* 0x000000000000094d */
        /*0080*/                   LDC.64 R2, c[0x0][0x388]                   &wr=0x0          ?trans1;           /* 0x0000e200ff027b82 */
        /*0090*/                   LDCU.64 UR4, c[0x0][0x358]                 &wr=0x1          ?trans7;           /* 0x00006b00ff0477ac */
        /*00a0*/                   LDC.64 R4, c[0x0][0x380]                   &wr=0x2          ?trans1;           /* 0x0000e000ff047b82 */
        /*00b0*/                   IMAD.WIDE.U32 R2, R7, 0x4, R2              &req={0}         ?WAIT6_END_GROUP;  /* 0x0000000407027825 */
        /*00c0*/                   LDG.E R2, desc[UR4][R2.64]                 &req={1} &wr=0x3 ?trans1;           /* 0x0000000402027981 */
        /*00d0*/                   IMAD.WIDE.U32 R4, R7, 0x4, R4              &req={2}         ?trans1;           /* 0x0000000407047825 */
        /*00e0*/                   IADD3 R0, PT, PT, R2, 0x11, RZ             &req={3}         ?WAIT4_END_GROUP;  /* 0x0000001102007810 */
        /*00f0*/                   LOP3.LUT R7, R0, 0x5a5a5a5a, RZ, 0x3c, !PT                  ?WAIT5_END_GROUP;  /* 0x5a5a5a5a00077812 */
        /*0100*/                   STG.E desc[UR4][R4.64], R7                                  ?trans1;           /* 0x0000000704007986 */
        /*0110*/                   EXIT                                                        ?trans5;           /* 0x000000000000794d */
        /*0120*/                   BRA 0x120;                                                                     /* 0xfffffffc00fc7947 */
        /*0130*/                   NOP;                                                                           /* 0x0000000000007918 */
"#;

    let result = lift_sass_text_to_ptx(
        text,
        SassLiftOptions {
            sm_version: 120,
            kernel_name: "int_add".to_string(),
            include_sass_comments: false,
            emit_unsupported_comments: true,
        },
    )
    .expect("real SM120 SASS text should lift");

    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.ptx.contains(".version 8.7"));
    assert!(result.ptx.contains(".param .u64 out"));
    assert!(result.ptx.contains(".param .u64 in"));
    assert!(result.ptx.contains(".param .u32 n"));
    assert!(result.ptx.contains(".reg .b32 %ur<6>;"));
    assert!(result.ptx.contains(".reg .b64 %rd<16>;"));
    assert!(result.ptx.contains("ld.param.u32 %ur5, [n];"));
    assert!(result.ptx.contains("mov.u32 %r0, %ntid.x;"));
    assert!(result.ptx.contains("setp.ge.u32 %p0, %r7, %ur5;"));
    assert!(result.ptx.contains("ld.param.u64 %rd2, [in];"));
    assert!(result.ptx.contains("ld.param.u64 %rd4, [out];"));
    assert!(result.ptx.contains("mul.wide.u32 %rd15, %r7, 4;"));
    assert!(result.ptx.contains("add.u64 %rd2, %rd2, %rd15;"));
    assert!(result.ptx.contains("ld.global.u32 %r2, [%rd2];"));
    assert!(result.ptx.contains("add.u32 %r0, %r2, 17;"));
    assert!(result.ptx.contains("xor.b32 %r7, %r0, 1515870810;"));
    assert!(result.ptx.contains("st.global.u32 [%rd4], %r7;"));
}

#[test]
fn text_lifter_handles_real_sm120_predicate_shift_pattern() {
    let text = r#"Function : pred_select
        /*00d0*/                   LOP3.LUT P0, RZ, R7.reuse, 0x1, RZ, 0xc0, !PT                  ?trans1;           /* 0x0000000107ff7812 */
        /*00f0*/                   SHF.L.U32 R9, R2, 0x1, RZ                     &req={3}         ?WAIT11_END_GROUP; /* 0x0000000102097819 */
        /*0100*/               @P0 SHF.R.U32.HI R9, RZ, 0x1, R2                                   ?WAIT5_END_GROUP;  /* 0x00000001ff090819 */
        /*0110*/                   EXIT                                                           ?trans5;           /* 0x000000000000794d */
"#;

    let result = lift_sass_text_to_ptx(
        text,
        SassLiftOptions {
            sm_version: 120,
            kernel_name: "pred_select".to_string(),
            include_sass_comments: false,
            emit_unsupported_comments: true,
        },
    )
    .expect("real SM120 predicate/shift text should lift");

    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.ptx.contains("and.b32 %r10, %r7, 1;"));
    assert!(result.ptx.contains("setp.eq.u32 %p0, %r10, 0;"));
    assert!(result.ptx.contains("shl.b32 %r9, %r2, 1;"));
    assert!(result.ptx.contains("@%p0 shr.u32 %r9, %r2, 1;"));
}
