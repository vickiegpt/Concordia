use nvidia_sass::pipeline::{
    compile_ptx_to_cubin, PtxToSassContext, PtxToSassPass, PtxToSassPipeline,
};

const SIMPLE_PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

.visible .entry simple_kernel()
{
    .reg .b32 %r<4>;
    mov.u32 %r1, %tid.x;
    add.s32 %r2, %r1, %r1;
    ret;
}
"#;

#[test]
fn default_ptx_to_sass_pipeline_builds_cubin_for_supported_ptx_subset() {
    let result = compile_ptx_to_cubin(SIMPLE_PTX, "simple_kernel", 120)
        .expect("supported PTX should compile");

    assert_eq!(
        result.pass_names,
        [
            "parse-ptx-subset",
            "allocate-registers",
            "schedule-control-codes",
            "validate-encoding",
            "build-cubin"
        ]
    );
    assert_eq!(result.module.sm_version, 120);
    assert_eq!(result.module.kernels[0].name, "simple_kernel");
    assert_eq!(
        result.module.kernels[0].instructions[0].opcode.mnemonic,
        "S2R"
    );
    assert_eq!(
        result.module.kernels[0].instructions[1].opcode.mnemonic,
        "IADD3"
    );
    assert_eq!(
        result.module.kernels[0].instructions[2].opcode.mnemonic,
        "EXIT"
    );
    assert!(result.cubin.len() > 64);
    assert_eq!(&result.cubin[0..4], b"\x7fELF");
}

#[test]
fn ptx_to_sass_pipeline_reports_unsupported_instruction_with_pass_name() {
    let ptx = r#"
.version 8.7
.target sm_120
.address_size 64

.visible .entry unsupported_kernel()
{
    .reg .f32 %f<2>;
    sin.approx.f32 %f1, %f1;
    ret;
}
"#;

    let err = compile_ptx_to_cubin(ptx, "unsupported_kernel", 120)
        .expect_err("unsupported PTX should fail");

    assert!(err.contains("parse-ptx-subset"), "{err}");
    assert!(err.contains("unsupported instruction"), "{err}");
    assert!(err.contains("sin.approx.f32"), "{err}");
}

struct RecordPluginPass;

impl PtxToSassPass for RecordPluginPass {
    fn name(&self) -> &'static str {
        "record-plugin-pass"
    }

    fn run(&self, ctx: &mut PtxToSassContext) -> Result<(), String> {
        let kernel_name = ctx.requested_kernel_name().to_string();
        ctx.notes_mut().push(format!("kernel={kernel_name}"));
        assert!(ctx.ptx().contains(".entry simple_kernel"));
        Ok(())
    }
}

#[test]
fn ptx_to_sass_pipeline_accepts_custom_plugin_passes() {
    let mut ctx = PtxToSassContext::new(SIMPLE_PTX, "simple_kernel", 120);

    PtxToSassPipeline::new()
        .add_pass(RecordPluginPass)
        .run(&mut ctx)
        .expect("custom plugin pass should run");

    assert_eq!(ctx.pass_names(), &["record-plugin-pass"]);
    assert_eq!(ctx.notes(), &["kernel=simple_kernel".to_string()]);
}
