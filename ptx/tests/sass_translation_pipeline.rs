use ptx::sass::translation_pipeline::{
    run_default_sass_to_ptx_pipeline, SassTranslationDirection, SassTranslationPass,
    SassTranslationState,
};
use ptx::sass::{
    EnhancedSassInstruction, SassDataType, SassLiftOptions, SassOpcodeClass, SassOperand,
    SassRegister,
};

fn reg(n: u32) -> SassOperand {
    SassOperand::Register(SassRegister::new("R", n))
}

fn sample_instructions() -> Vec<EnhancedSassInstruction> {
    let mut s2r = EnhancedSassInstruction::new("S2R".to_string(), 0x0);
    s2r.opcode_class = SassOpcodeClass::SpecialRegRead;
    s2r.data_type = Some(SassDataType::U32);
    s2r.dest_operands.push(reg(0));
    s2r.src_operands
        .push(SassOperand::SpecialRegister("SR_TID.X".to_string()));
    s2r.instruction_text = "S2R R0, SR_TID.X;".to_string();

    let mut add = EnhancedSassInstruction::new("IADD3".to_string(), 0x10);
    add.opcode_class = SassOpcodeClass::IntegerArithmetic;
    add.data_type = Some(SassDataType::S32);
    add.dest_operands.push(reg(1));
    add.src_operands.push(reg(0));
    add.src_operands.push(reg(0));
    add.instruction_text = "IADD3 R1, R0, R0;".to_string();

    let mut exit = EnhancedSassInstruction::new("EXIT".to_string(), 0x20);
    exit.opcode_class = SassOpcodeClass::Exit;
    exit.instruction_text = "EXIT;".to_string();

    vec![s2r, add, exit]
}

#[test]
fn default_sass_to_ptx_pipeline_runs_named_passes_and_produces_parseable_ptx() {
    let result = run_default_sass_to_ptx_pipeline(
        sample_instructions(),
        SassLiftOptions {
            sm_version: 120,
            kernel_name: "pipeline_kernel".to_string(),
            include_sass_comments: false,
            emit_unsupported_comments: false,
        },
    )
    .expect("pipeline should lift supported SASS");

    assert_eq!(
        result.pass_names,
        ["sass-lift-to-ptx", "validate-lifted-ptx"]
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.ptx.contains(".target sm_120"));
    assert!(result.ptx.contains("mov.u32 %r0, %tid.x;"));
    assert!(result.ptx.contains("add.s32 %r1, %r0, %r0;"));
    ptx_parser::parse_module_checked(&result.ptx).expect("lifted PTX should parse");
}

struct NotePass;

impl SassTranslationPass for NotePass {
    fn name(&self) -> &'static str {
        "note-pass"
    }

    fn direction(&self) -> SassTranslationDirection {
        SassTranslationDirection::SassToPtx
    }

    fn run(&self, state: &mut SassTranslationState) -> Result<(), String> {
        state.notes.push("custom pass ran".to_string());
        Ok(())
    }
}

#[test]
fn sass_translation_pipeline_accepts_custom_plugin_passes() {
    let mut state = SassTranslationState::new(
        sample_instructions(),
        SassLiftOptions {
            sm_version: 120,
            kernel_name: "plugin_kernel".to_string(),
            include_sass_comments: false,
            emit_unsupported_comments: false,
        },
    );

    ptx::sass::translation_pipeline::SassTranslationPipeline::new()
        .add_pass(NotePass)
        .run(&mut state)
        .expect("custom pass should run");

    assert_eq!(state.pass_names, ["note-pass"]);
    assert_eq!(state.notes, ["custom pass ran"]);
}
