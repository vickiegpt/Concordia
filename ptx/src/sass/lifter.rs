use std::{collections::HashSet, io::Write, path::Path, process::Command};

use super::{
    CubinKernel, CubinParser, EnhancedSassInstruction, ParsedCubin, SassDataType, SassDisassembler,
    SassMemorySpace, SassOpcodeClass, SassOperand, SassRegister, TextDisassemblyParser,
};

#[derive(Debug, Clone)]
pub struct SassLiftOptions {
    pub sm_version: u32,
    pub kernel_name: String,
    pub include_sass_comments: bool,
    pub emit_unsupported_comments: bool,
}

impl Default for SassLiftOptions {
    fn default() -> Self {
        Self {
            sm_version: 120,
            kernel_name: String::new(),
            include_sass_comments: true,
            emit_unsupported_comments: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLiftDiagnostic {
    pub address: Option<u64>,
    pub opcode: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLiftResult {
    pub ptx: String,
    pub diagnostics: Vec<SassLiftDiagnostic>,
}

pub fn lift_instructions_to_ptx(
    instructions: &[EnhancedSassInstruction],
    options: &SassLiftOptions,
) -> SassLiftResult {
    let mut ctx = LiftContext::new(options);
    ctx.emit_module(instructions);
    SassLiftResult {
        ptx: ctx.output,
        diagnostics: ctx.diagnostics,
    }
}

pub fn lift_sass_text_to_ptx(
    text: &str,
    mut options: SassLiftOptions,
) -> Result<SassLiftResult, String> {
    let instructions = TextDisassemblyParser::parse_cuobjdump_output(text);
    if instructions.is_empty() {
        return Err("No SASS instructions parsed from text input".to_string());
    }

    let (kernel_name, instructions) =
        select_sass_text_instructions(&instructions, &options.kernel_name)?;
    options.kernel_name = kernel_name;

    Ok(lift_instructions_to_ptx(&instructions, &options))
}

pub fn lift_cubin_to_ptx(
    cubin_data: &[u8],
    mut options: SassLiftOptions,
) -> Result<SassLiftResult, String> {
    if let Some(cuobjdump_path) = std::env::var_os("HETGPU_SASS_LIFTER_CUOBJDUMP") {
        if !cuobjdump_path.is_empty() {
            return lift_cubin_to_ptx_with_cuobjdump(
                cubin_data,
                options,
                Path::new(&cuobjdump_path),
            );
        }
    }

    let parsed = CubinParser::new(cubin_data.to_vec())
        .parse()
        .map_err(|e| format!("Failed to parse CUBIN: {}", e))?;
    let kernel = select_cubin_kernel(&parsed, &options.kernel_name)?;

    options.kernel_name = kernel.name.clone();
    if options.sm_version == 0 {
        options.sm_version = kernel.sm_version;
    }

    let disassembler = SassDisassembler::new(kernel.sm_version)
        .map_err(|e| format!("Failed to create SASS disassembler: {}", e))?;
    let mut instructions = disassembler.disassemble(&kernel.code, kernel.address);
    for inst in &mut instructions {
        inst.function_name = Some(kernel.name.clone());
        if let Some(debug_info) = parsed.debug_lines.get(&inst.address) {
            inst.ptx_file = Some(debug_info.file.clone());
            inst.ptx_line = Some(debug_info.line);
            inst.ptx_column = Some(debug_info.column);
        }
    }

    Ok(lift_instructions_to_ptx(&instructions, &options))
}

pub fn lift_cubin_to_ptx_with_cuobjdump(
    cubin_data: &[u8],
    mut options: SassLiftOptions,
    cuobjdump_path: impl AsRef<Path>,
) -> Result<SassLiftResult, String> {
    let cuobjdump_path = cuobjdump_path.as_ref();
    let mut cubin_file = tempfile::Builder::new()
        .prefix("hetgpu-sass-lifter-")
        .suffix(".cubin")
        .tempfile()
        .map_err(|err| format!("Failed to create temporary CUBIN file: {err}"))?;
    cubin_file
        .write_all(cubin_data)
        .map_err(|err| format!("Failed to write temporary CUBIN file: {err}"))?;
    cubin_file
        .flush()
        .map_err(|err| format!("Failed to flush temporary CUBIN file: {err}"))?;

    let output = Command::new(cuobjdump_path)
        .arg("--dump-sass")
        .arg(cubin_file.path())
        .output()
        .map_err(|err| {
            format!(
                "Failed to execute cuobjdump '{}': {err}",
                cuobjdump_path.display()
            )
        })?;

    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| format!("cuobjdump stdout was not UTF-8: {err}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!(
            "cuobjdump '{}' failed with status {}: {}",
            cuobjdump_path.display(),
            output.status,
            stderr.trim()
        ));
    }

    if options.sm_version == 0 {
        options.sm_version = infer_sm_version_from_cuobjdump_text(&stdout)
            .ok_or_else(|| "cuobjdump output did not include an sm_ target".to_string())?;
    }

    lift_sass_text_to_ptx(&stdout, options)
}

fn infer_sm_version_from_cuobjdump_text(text: &str) -> Option<u32> {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .find_map(|token| token.strip_prefix("sm_")?.parse::<u32>().ok())
}

fn select_sass_text_instructions(
    instructions: &[EnhancedSassInstruction],
    requested_kernel_name: &str,
) -> Result<(String, Vec<EnhancedSassInstruction>), String> {
    if !is_unspecified_kernel_name(requested_kernel_name) {
        let selected: Vec<_> = instructions
            .iter()
            .filter(|inst| inst.function_name.as_deref() == Some(requested_kernel_name))
            .cloned()
            .collect();
        if selected.is_empty() {
            return Err(format!(
                "No SASS instructions parsed for kernel '{}'",
                requested_kernel_name
            ));
        }
        return Ok((requested_kernel_name.to_string(), selected));
    }

    let function_names: HashSet<&str> = instructions
        .iter()
        .filter_map(|inst| inst.function_name.as_deref())
        .collect();

    match function_names.len() {
        0 => Ok(("kernel".to_string(), instructions.to_vec())),
        1 => {
            let function_name = function_names.iter().next().copied().unwrap();
            let selected = instructions
                .iter()
                .filter(|inst| inst.function_name.as_deref() == Some(function_name))
                .cloned()
                .collect();
            Ok((function_name.to_string(), selected))
        }
        _ => Err("Multiple SASS functions parsed; set kernel_name to select one".to_string()),
    }
}

/// CUBIN parsing already provides kernel boundaries, so selection is a simple
/// metadata lookup before disassembling the chosen kernel's code bytes.
fn select_cubin_kernel<'a>(
    parsed: &'a ParsedCubin,
    requested_kernel_name: &str,
) -> Result<&'a CubinKernel, String> {
    if is_unspecified_kernel_name(requested_kernel_name) {
        return parsed
            .kernels
            .first()
            .ok_or_else(|| "No kernels found in CUBIN".to_string());
    }

    parsed
        .kernels
        .iter()
        .find(|kernel| kernel.name == requested_kernel_name)
        .ok_or_else(|| format!("No kernel named '{}' found in CUBIN", requested_kernel_name))
}

fn is_unspecified_kernel_name(kernel_name: &str) -> bool {
    kernel_name.is_empty()
}

struct LiftContext<'a> {
    options: &'a SassLiftOptions,
    output: String,
    diagnostics: Vec<SassLiftDiagnostic>,
    branch_targets: HashSet<u64>,
    scratch_gpr: Option<String>,
    uses_cuda_param_abi: bool,
    uses_shared_memory: bool,
}

impl<'a> LiftContext<'a> {
    fn new(options: &'a SassLiftOptions) -> Self {
        Self {
            options,
            output: String::new(),
            diagnostics: Vec::new(),
            branch_targets: HashSet::new(),
            scratch_gpr: None,
            uses_cuda_param_abi: false,
            uses_shared_memory: false,
        }
    }

    fn emit_module(&mut self, instructions: &[EnhancedSassInstruction]) {
        self.collect_branch_targets(instructions);
        self.uses_cuda_param_abi = uses_cuda_param_abi(instructions);
        self.uses_shared_memory = uses_shared_memory(instructions);
        let mut regs = RegisterDecls::from_instructions(instructions);
        if needs_gpr_scratch(instructions) {
            self.scratch_gpr = Some(format!("%r{}", regs.max_gpr));
            regs.max_gpr += 1;
        }
        if self.uses_cuda_param_abi || self.uses_shared_memory {
            regs.max_b64 = regs.max_b64.max(16);
        }

        self.output
            .push_str(ptx_version_for_sm(self.options.sm_version));
        self.output.push('\n');
        self.output
            .push_str(&format!(".target sm_{}\n", self.options.sm_version));
        self.output.push_str(".address_size 64\n\n");
        if self.uses_cuda_param_abi {
            self.output.push_str(&format!(
                ".visible .entry {}(\n    .param .u64 out,\n    .param .u64 in,\n    .param .u32 n\n)\n{{\n",
                sanitize_ident(&self.options.kernel_name)
            ));
        } else {
            self.output.push_str(&format!(
                ".visible .entry {}()\n{{\n",
                sanitize_ident(&self.options.kernel_name)
            ));
        }

        if regs.max_gpr > 0 {
            self.output
                .push_str(&format!("    .reg .b32 %r<{}>;\n", regs.max_gpr));
        }
        if regs.max_uniform_gpr > 0 {
            self.output
                .push_str(&format!("    .reg .b32 %ur<{}>;\n", regs.max_uniform_gpr));
        }
        if regs.max_pred > 0 {
            self.output
                .push_str(&format!("    .reg .pred %p<{}>;\n", regs.max_pred));
        }
        if regs.max_uniform_pred > 0 {
            self.output
                .push_str(&format!("    .reg .pred %up<{}>;\n", regs.max_uniform_pred));
        }
        if regs.max_b64 > 0 {
            self.output
                .push_str(&format!("    .reg .b64 %rd<{}>;\n", regs.max_b64));
        }
        if self.uses_shared_memory {
            self.output
                .push_str("    .shared .align 4 .b8 scratch[512];\n");
        }
        if regs.has_decls() || self.uses_shared_memory {
            self.output.push('\n');
        }

        for inst in instructions {
            self.output
                .push_str(&format!("{}:\n", label_for_address(inst.address)));
            if self.options.include_sass_comments {
                self.output.push_str(&format!(
                    "    // 0x{:04x}: {}\n",
                    inst.address, inst.instruction_text
                ));
            }
            if let Some(line) = self.lift_instruction(inst) {
                self.output.push_str(&format!("    {}\n", line));
            }
        }

        self.output.push_str("}\n");
    }

    fn collect_branch_targets(&mut self, instructions: &[EnhancedSassInstruction]) {
        for inst in instructions {
            if matches!(
                inst.opcode_class,
                SassOpcodeClass::Branch | SassOpcodeClass::ConditionalBranch
            ) {
                if let Some(target) = branch_target(inst) {
                    self.branch_targets.insert(target);
                }
            }
        }
    }

    fn lift_instruction(&mut self, inst: &EnhancedSassInstruction) -> Option<String> {
        let pred = predicate_prefix(inst);
        match inst.opcode.as_str() {
            "S2R" | "S2UR" | "CS2R" => Some(format!(
                "{}mov.u32 {}, {};",
                pred,
                dest_operand(inst).unwrap_or_else(|| "%r0".to_string()),
                inst.src_operands
                    .first()
                    .map(format_operand)
                    .unwrap_or_else(|| "%tid.x".to_string())
            )),
            "MOV" | "MOV32I" => Some(unary_op(
                inst,
                &pred,
                "mov",
                &data_type_suffix(inst, SassDataType::U32),
            )),
            "UMOV" => Some(unary_op(inst, &pred, "mov", "u32")),
            "ULEA" => Some(ulea_op(inst, &pred)),
            "LEA" => Some(lea_op(inst, &pred)),
            "IADD" => Some(binary_op(
                inst,
                &pred,
                "add",
                &data_type_suffix(inst, SassDataType::S32),
            )),
            "IADD3" if is_extended_iadd3(inst) => Some(iadd3_extended_op(
                inst,
                &pred,
                &data_type_suffix(inst, SassDataType::U32),
            )),
            "IADD3" if inst.src_operands.len() == 3 => Some(iadd3_op(
                inst,
                &pred,
                &data_type_suffix(inst, SassDataType::S32),
                self.scratch_gpr.as_deref(),
            )),
            "IADD3" if inst.src_operands.len() < 3 => Some(binary_op(
                inst,
                &pred,
                "add",
                &data_type_suffix(inst, SassDataType::S32),
            )),
            "IADD3" => self.unsupported(inst, "IADD3 extended operand lifting is not implemented"),
            "IMUL" => Some(binary_op(
                inst,
                &pred,
                "mul.lo",
                &data_type_suffix(inst, SassDataType::S32),
            )),
            "IMAD" if has_modifier(inst, "WIDE") => Some(imad_wide_op(inst, &pred)),
            "IMAD" => Some(ternary_op(
                inst,
                &pred,
                "mad.lo",
                &data_type_suffix(inst, SassDataType::S32),
            )),
            "SHL" => Some(binary_op(inst, &pred, "shl", "b32")),
            "SHR" => Some(binary_op(
                inst,
                &pred,
                "shr",
                &data_type_suffix(inst, SassDataType::U32),
            )),
            "SHF" => Some(shf_op(inst, &pred, self.scratch_gpr.as_deref())),
            "LOP" if inst.src_operands.len() == 2 => Some(binary_op(inst, &pred, "and", "b32")),
            "LOP" => self.unsupported(inst, "logical operation lifting is not implemented"),
            "LOP3" if is_lop3_odd_predicate(inst) => Some(lop3_odd_predicate_op(
                inst,
                &pred,
                self.scratch_gpr.as_deref(),
            )),
            "LOP3" if is_lop3_and(inst) => Some(lop3_binary_op(inst, &pred, "and")),
            "LOP3" if is_lop3_or(inst) => Some(lop3_binary_op(inst, &pred, "or")),
            "LOP3" if is_lop3_xor(inst) => Some(lop3_xor_op(inst, &pred)),
            "LOP3" => self.unsupported(inst, "LOP3 truth-table lifting is not implemented"),
            "POPC" => Some(unary_op(inst, &pred, "popc", "b32")),
            "FADD" => Some(float_binary_op(
                inst,
                &pred,
                "add",
                &data_type_suffix(inst, SassDataType::F32),
            )),
            "FMUL" => Some(float_binary_op(
                inst,
                &pred,
                "mul",
                &data_type_suffix(inst, SassDataType::F32),
            )),
            "FFMA" => Some(float_ternary_op(
                inst,
                &pred,
                "fma.rn",
                &data_type_suffix(inst, SassDataType::F32),
            )),
            "HFMA2" => Some(hfma2_constant_op(inst, &pred)),
            "I2FP" => Some(i2fp_op(inst, &pred)),
            "F2I" => Some(f2i_op(inst, &pred)),
            "FMNMX" => Some(fmnmx_op(inst, &pred)),
            "FABS" => Some(unary_op(
                inst,
                &pred,
                "abs",
                &data_type_suffix(inst, SassDataType::F32),
            )),
            "FNEG" => Some(unary_op(
                inst,
                &pred,
                "neg",
                &data_type_suffix(inst, SassDataType::F32),
            )),
            "LDG" | "LDS" | "LDL" => Some(load_op(inst, &pred)),
            "LDC" | "LDCU" => Some(ldc_op(inst, &pred)),
            "STG" | "STS" | "STL" => Some(store_op(inst, &pred)),
            "ISETP" | "FSETP" => Some(setp_op(inst, &pred)),
            "SEL" => Some(sel_op(inst, &pred)),
            "PSETP" => self.unsupported(inst, "predicate set lifting is not implemented"),
            "BRA" | "BRX" | "JMP" => Some(branch_op(inst, &pred)),
            "BAR" => Some(format!("{}bar.sync 0;", pred)),
            "DEPBAR" => Some("// depbar preserved from SASS;".to_string()),
            "MEMBAR" => Some(format!("{}membar.gl;", pred)),
            "NOP" => Some("// nop;".to_string()),
            "EXIT" | "RET" => Some(format!("{}ret;", pred)),
            "HMMA" | "IMMA" | "BMMA" | "DMMA" => {
                self.unsupported(inst, "tensor instruction lifting is not implemented")
            }
            "MUFU" if has_modifier(inst, "RSQ") => Some(mufu_rsq_op(inst, &pred)),
            "MUFU" => self.unsupported(inst, "MUFU sub-operation lifting is not implemented"),
            _ => self.unsupported(inst, "instruction lifting is not implemented"),
        }
    }

    fn unsupported(&mut self, inst: &EnhancedSassInstruction, message: &str) -> Option<String> {
        self.diagnostics.push(SassLiftDiagnostic {
            address: Some(inst.address),
            opcode: inst.opcode.clone(),
            message: message.to_string(),
        });
        if self.options.emit_unsupported_comments {
            Some(format!(
                "// unsupported SASS {} at 0x{:04x}: {}",
                inst.opcode, inst.address, message
            ))
        } else {
            None
        }
    }
}

fn needs_gpr_scratch(instructions: &[EnhancedSassInstruction]) -> bool {
    instructions.iter().any(|inst| {
        inst.opcode == "IADD3" && inst.src_operands.len() == 3
            || is_lop3_odd_predicate(inst)
            || is_shf_left_rotate(inst)
    })
}

#[derive(Debug, Default)]
struct RegisterDecls {
    max_gpr: u32,
    max_pred: u32,
    max_uniform_gpr: u32,
    max_uniform_pred: u32,
    max_b64: u32,
}

impl RegisterDecls {
    fn from_instructions(instructions: &[EnhancedSassInstruction]) -> Self {
        let mut decls = Self::default();
        for inst in instructions {
            for operand in inst.dest_operands.iter().chain(inst.src_operands.iter()) {
                collect_register_decl(operand, &mut decls);
            }
            if let Some(predicate) = &inst.predicate {
                collect_register_decl(predicate, &mut decls);
            }
        }
        decls
    }

    fn has_decls(&self) -> bool {
        self.max_gpr > 0
            || self.max_pred > 0
            || self.max_uniform_gpr > 0
            || self.max_uniform_pred > 0
            || self.max_b64 > 0
    }
}

fn collect_register_decl(operand: &SassOperand, decls: &mut RegisterDecls) {
    match operand {
        SassOperand::Register(reg) => collect_register(reg, decls),
        SassOperand::Predicate { register, .. } => collect_register(register, decls),
        SassOperand::Memory { base, index, .. } => {
            if let Some(reg) = base {
                collect_register(reg, decls);
            }
            if let Some(reg) = index {
                collect_register(reg, decls);
            }
        }
        _ => {}
    }
}

fn collect_register(reg: &SassRegister, decls: &mut RegisterDecls) {
    if reg.is_zero {
        return;
    }
    match reg.prefix.as_str() {
        "R" => decls.max_gpr = decls.max_gpr.max(reg.number + 1),
        "P" => decls.max_pred = decls.max_pred.max(reg.number + 1),
        "UR" => decls.max_uniform_gpr = decls.max_uniform_gpr.max(reg.number + 1),
        "UP" => decls.max_uniform_pred = decls.max_uniform_pred.max(reg.number + 1),
        _ => {}
    }
}

fn ptx_version_for_sm(sm_version: u32) -> &'static str {
    if sm_version >= 120 {
        ".version 8.7"
    } else {
        ".version 8.5"
    }
}

fn unary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!("{}{}.{} {}, {};", pred, op, ty, dst, src)
}

fn binary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!("{}{}.{} {}, {}, {};", pred, op, ty, dst, src0, src1)
}

fn float_binary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst
        .src_operands
        .first()
        .map(format_float_operand)
        .unwrap_or_else(|| "0.0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_float_operand)
        .unwrap_or_else(|| "0.0".to_string());
    format!("{}{}.{} {}, {}, {};", pred, op, ty, dst, src0, src1)
}

fn iadd3_op(
    inst: &EnhancedSassInstruction,
    pred: &str,
    ty: &str,
    scratch_gpr: Option<&str>,
) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let src2 = inst
        .src_operands
        .get(2)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let scratch = scratch_gpr.unwrap_or("%r0");
    format!(
        "{}add.{} {}, {}, {};\n    {}add.{} {}, {}, {};",
        pred, ty, scratch, src0, src1, pred, ty, dst, scratch, src2
    )
}

fn ternary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let src2 = inst
        .src_operands
        .get(2)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!(
        "{}{}.{} {}, {}, {}, {};",
        pred, op, ty, dst, src0, src1, src2
    )
}

fn float_ternary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst
        .src_operands
        .first()
        .map(format_float_operand)
        .unwrap_or_else(|| "0.0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_float_operand)
        .unwrap_or_else(|| "0.0".to_string());
    let src2 = inst
        .src_operands
        .get(2)
        .map(format_float_operand)
        .unwrap_or_else(|| "0.0".to_string());
    format!(
        "{}{}.{} {}, {}, {}, {};",
        pred, op, ty, dst, src0, src1, src2
    )
}

fn hfma2_constant_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let bits = extract_sass_encoding(inst)
        .map(|encoding| (encoding >> 32) as u32)
        .unwrap_or(0);
    format!("{}mov.b32 {}, 0x{:08x};", pred, dst, bits)
}

fn i2fp_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!("{}cvt.rn.f32.u32 {}, {};", pred, dst, src)
}

fn f2i_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!("{}cvt.rzi.u32.f32 {}, {};", pred, dst, src)
}

fn mufu_rsq_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!("{}rsqrt.approx.ftz.f32 {}, {};", pred, dst, src)
}

fn fmnmx_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst
        .src_operands
        .first()
        .map(format_float_operand)
        .unwrap_or_else(|| "0.0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_float_operand)
        .unwrap_or_else(|| "0.0".to_string());
    format!("{}min.f32 {}, {}, {};", pred, dst, src0, src1)
}

fn imad_wide_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_rd_operand(inst).unwrap_or_else(|| "%rd0".to_string());
    let src0 = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let base = inst
        .src_operands
        .get(2)
        .map(format_rd_operand)
        .unwrap_or_else(|| "0".to_string());

    format!(
        "{}mul.wide.u32 %rd15, {}, {};\n    {}add.u64 {}, {}, %rd15;",
        pred, src0, src1, pred, dst, base
    )
}

fn iadd3_extended_op(inst: &EnhancedSassInstruction, pred: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let terms: Vec<(String, bool)> = inst
        .src_operands
        .iter()
        .filter(|op| !is_zero_register_operand(op))
        .map(format_signed_operand)
        .collect();

    match terms.as_slice() {
        [] => format!("{}mov.u32 {}, 0;", pred, dst),
        [(src, false)] => format!("{}mov.u32 {}, {};", pred, dst, src),
        [(src, true)] => format!("{}sub.{} {}, 0, {};", pred, ty, dst, src),
        [(src0, false), (src1, false)] => {
            format!("{}add.{} {}, {}, {};", pred, ty, dst, src0, src1)
        }
        [(src0, false), (src1, true)] => {
            format!("{}sub.{} {}, {}, {};", pred, ty, dst, src0, src1)
        }
        [(src0, true), (src1, false)] => {
            format!("{}sub.{} {}, {}, {};", pred, ty, dst, src1, src0)
        }
        [(src0, true), (src1, true)] => format!(
            "{}sub.{} {}, 0, {};\n    {}sub.{} {}, {}, {};",
            pred, ty, dst, src0, pred, ty, dst, dst, src1
        ),
        [(src0, neg0), rest @ ..] => {
            let mut lines = if *neg0 {
                format!("{}sub.{} {}, 0, {};", pred, ty, dst, src0)
            } else {
                format!("{}mov.u32 {}, {};", pred, dst, src0)
            };
            for (src, negated) in rest {
                let op = if *negated { "sub" } else { "add" };
                lines.push_str(&format!(
                    "\n    {}{}.{} {}, {}, {};",
                    pred, op, ty, dst, dst, src
                ));
            }
            lines
        }
    }
}

fn lop3_xor_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    lop3_binary_op(inst, pred, "xor")
}

fn lop3_binary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!("{}{}.b32 {}, {}, {};", pred, op, dst, src0, src1)
}

fn lop3_odd_predicate_op(
    inst: &EnhancedSassInstruction,
    pred: &str,
    scratch_gpr: Option<&str>,
) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%p0".to_string());
    let value = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let scratch = scratch_gpr.unwrap_or("%r0");
    format!(
        "{}and.b32 {}, {}, 1;\n    {}setp.ne.u32 {}, {}, 0;",
        pred, scratch, value, pred, dst, scratch
    )
}

fn shf_op(inst: &EnhancedSassInstruction, pred: &str, scratch_gpr: Option<&str>) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let amount = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    if has_modifier(inst, "R") {
        let value = inst
            .src_operands
            .get(2)
            .map(format_operand)
            .unwrap_or_else(|| "0".to_string());
        format!(
            "{}shr.{} {}, {}, {};",
            pred,
            data_type_suffix(inst, SassDataType::U32),
            dst,
            value,
            amount
        )
    } else if is_shf_left_rotate(inst) {
        let value = inst
            .src_operands
            .first()
            .map(format_operand)
            .unwrap_or_else(|| "0".to_string());
        let scratch = scratch_gpr.unwrap_or("%r0");
        let right_amount = shift_amount_immediate(inst)
            .map(|amount| 32 - amount)
            .unwrap_or(0);
        format!(
            "{}shl.b32 {}, {}, {};\n    {}shr.u32 {}, {}, {};\n    {}or.b32 {}, {}, {};",
            pred, dst, value, amount, pred, scratch, value, right_amount, pred, dst, dst, scratch
        )
    } else {
        let value = inst
            .src_operands
            .first()
            .map(format_operand)
            .unwrap_or_else(|| "0".to_string());
        format!("{}shl.b32 {}, {}, {};", pred, dst, value, amount)
    }
}

fn sel_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let mut src0 = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let mut src1 = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let (predicate, negated) = inst
        .src_operands
        .get(2)
        .map(format_sel_predicate_operand)
        .unwrap_or_else(|| ("%p0".to_string(), false));
    if negated {
        std::mem::swap(&mut src0, &mut src1);
    }
    format!(
        "{}selp.{} {}, {}, {}, {};",
        pred,
        data_type_suffix(inst, SassDataType::U32),
        dst,
        src0,
        src1,
        predicate
    )
}

fn ulea_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%ur0".to_string());
    format!("{}mov.u32 {}, 0;", pred, dst)
}

fn lea_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let shift = inst
        .src_operands
        .get(2)
        .and_then(|operand| match operand {
            SassOperand::Immediate(value) => Some(*value),
            _ => None,
        })
        .unwrap_or(0);
    if shift == 0 {
        format!("{}mov.u32 {}, {};", pred, dst, src)
    } else {
        format!("{}shl.b32 {}, {}, {};", pred, dst, src, shift)
    }
}

fn load_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    if is_shared_memory_inst(inst) {
        return shared_load_op(inst, pred);
    }
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let addr = inst
        .src_operands
        .first()
        .map(|operand| format_memory_address_operand(inst, operand))
        .unwrap_or_else(|| "[0]".to_string());
    format!(
        "{}ld.{}.{} {}, {};",
        pred,
        memory_space_suffix(inst),
        data_type_suffix(inst, SassDataType::U32),
        dst,
        addr
    )
}

fn ldc_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let const_bank = inst.src_operands.first().and_then(constant_bank_operand);
    match const_bank {
        Some((0, 0x360)) => format!("{}mov.u32 {}, %ntid.x;", pred, dst),
        Some((0, 0x380)) if is_64bit_modifier(inst) => format!(
            "{}ld.param.u64 {}, [out];",
            pred,
            dest_rd_operand(inst).unwrap_or_else(|| "%rd0".to_string())
        ),
        Some((0, 0x388)) if is_64bit_modifier(inst) => format!(
            "{}ld.param.u64 {}, [in];",
            pred,
            dest_rd_operand(inst).unwrap_or_else(|| "%rd0".to_string())
        ),
        Some((0, 0x390)) => format!("{}ld.param.u32 {}, [n];", pred, dst),
        Some((0, 0x358)) if is_64bit_modifier(inst) => format!("{}mov.u32 {}, 0;", pred, dst),
        Some(_) => format!("{}mov.u32 {}, 0;", pred, dst),
        None => load_op(inst, pred),
    }
}

fn store_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    if is_shared_memory_inst(inst) {
        return shared_store_op(inst, pred);
    }
    let addr = inst
        .dest_operands
        .first()
        .map(|operand| format_memory_address_operand(inst, operand))
        .unwrap_or_else(|| "[0]".to_string());
    let src = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!(
        "{}st.{}.{} {}, {};",
        pred,
        memory_space_suffix(inst),
        data_type_suffix(inst, SassDataType::U32),
        addr,
        src
    )
}

fn shared_load_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let addr_setup = inst
        .src_operands
        .first()
        .map(shared_address_setup)
        .unwrap_or_else(|| "mov.u64 %rd14, scratch;".to_string());
    format!(
        "{}\n    {}ld.shared.{} {}, [%rd14];",
        addr_setup,
        pred,
        data_type_suffix(inst, SassDataType::U32),
        dst
    )
}

fn shared_store_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let addr_setup = inst
        .dest_operands
        .first()
        .map(shared_address_setup)
        .unwrap_or_else(|| "mov.u64 %rd14, scratch;".to_string());
    let src = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!(
        "{}\n    {}st.shared.{} [%rd14], {};",
        addr_setup,
        pred,
        data_type_suffix(inst, SassDataType::U32),
        src
    )
}

fn shared_address_setup(operand: &SassOperand) -> String {
    match operand {
        SassOperand::Memory { base, offset, .. } => {
            let offset_reg = base
                .as_ref()
                .map(format_register)
                .unwrap_or_else(|| "0".to_string());
            let mut setup = format!(
                "cvt.u64.u32 %rd14, {};\n    mov.u64 %rd13, scratch;\n    add.u64 %rd14, %rd13, %rd14;",
                offset_reg
            );
            if *offset != 0 {
                setup.push_str(&format!("\n    add.u64 %rd14, %rd14, {};", offset));
            }
            setup
        }
        _ => "mov.u64 %rd14, scratch;".to_string(),
    }
}

fn setp_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%p0".to_string());
    let (src0, src1) = setp_compare_operands(inst);
    let default_ty = match inst.opcode.as_str() {
        "FSETP" => SassDataType::F32,
        "PSETP" => SassDataType::Pred,
        _ => SassDataType::S32,
    };
    format!(
        "{}setp.{}.{} {}, {}, {};",
        pred,
        comparison_suffix(inst),
        data_type_suffix(inst, default_ty),
        dst,
        src0,
        src1
    )
}

fn setp_compare_operands(inst: &EnhancedSassInstruction) -> (String, String) {
    if inst.src_operands.len() >= 4 && is_pt_register_operand(&inst.src_operands[0]) {
        return (
            format_operand(&inst.src_operands[1]),
            format_operand(&inst.src_operands[2]),
        );
    }

    let src0 = inst
        .src_operands
        .first()
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    let src1 = inst
        .src_operands
        .get(1)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    (src0, src1)
}

fn branch_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let target = branch_target(inst)
        .map(label_for_address)
        .or_else(|| {
            inst.dest_operands
                .iter()
                .chain(inst.src_operands.iter())
                .next()
                .map(format_operand)
        })
        .unwrap_or_else(|| "L_0000".to_string());
    format!("{}bra {};", pred, target)
}

fn data_type_suffix(inst: &EnhancedSassInstruction, default: SassDataType) -> String {
    match inst.data_type.unwrap_or(default) {
        SassDataType::U8 => "u8",
        SassDataType::U16 => "u16",
        SassDataType::U32 => "u32",
        SassDataType::U64 => "u64",
        SassDataType::U128 => "u128",
        SassDataType::S8 => "s8",
        SassDataType::S16 => "s16",
        SassDataType::S32 => "s32",
        SassDataType::S64 => "s64",
        SassDataType::F16 => "f16",
        SassDataType::F32 => "f32",
        SassDataType::F64 => "f64",
        SassDataType::BF16 => "bf16",
        SassDataType::TF32 => "tf32",
        SassDataType::FP8E4M3 => "e4m3",
        SassDataType::FP8E5M2 => "e5m2",
        SassDataType::B8 => "b8",
        SassDataType::B16 => "b16",
        SassDataType::B32 => "b32",
        SassDataType::B64 => "b64",
        SassDataType::B128 => "b128",
        SassDataType::Pred => "pred",
        SassDataType::Unknown => "b32",
    }
    .to_string()
}

fn memory_space_suffix(inst: &EnhancedSassInstruction) -> String {
    let space = inst.memory_space.or(match inst.opcode.as_str() {
        "LDG" | "STG" => Some(SassMemorySpace::Global),
        "LDS" | "STS" => Some(SassMemorySpace::Shared),
        "LDL" | "STL" => Some(SassMemorySpace::Local),
        "LDC" => Some(SassMemorySpace::Constant),
        _ => None,
    });
    match space.unwrap_or(SassMemorySpace::Global) {
        SassMemorySpace::Global => "global",
        SassMemorySpace::Shared => "shared",
        SassMemorySpace::Local => "local",
        SassMemorySpace::Constant => "const",
        SassMemorySpace::Texture => "tex",
        SassMemorySpace::Surface => "surf",
        SassMemorySpace::Generic => "generic",
    }
    .to_string()
}

fn is_shared_memory_inst(inst: &EnhancedSassInstruction) -> bool {
    matches!(inst.memory_space, Some(SassMemorySpace::Shared))
        || matches!(inst.opcode.as_str(), "LDS" | "STS")
}

fn format_memory_address_operand(inst: &EnhancedSassInstruction, operand: &SassOperand) -> String {
    if is_shared_memory_inst(inst) {
        return format_shared_address_operand(operand);
    }
    format_address_operand(operand)
}

fn format_shared_address_operand(operand: &SassOperand) -> String {
    match operand {
        SassOperand::Memory { base, offset, .. } => {
            let mut expr = "scratch".to_string();
            if let Some(base) = base {
                expr.push('+');
                expr.push_str(&format_register(base));
            }
            if *offset > 0 {
                expr.push_str(&format!("+{}", offset));
            } else if *offset < 0 {
                expr.push_str(&offset.to_string());
            }
            format!("[{}]", expr)
        }
        _ => format_address_operand(operand),
    }
}

fn comparison_suffix(inst: &EnhancedSassInstruction) -> String {
    inst.modifiers
        .iter()
        .find_map(|modifier| {
            let normalized = modifier.trim_start_matches('.').to_ascii_uppercase();
            match normalized.as_str() {
                "EQ" => Some("eq"),
                "NE" | "NEU" => Some("ne"),
                "LT" | "LO" => Some("lt"),
                "LE" | "LS" => Some("le"),
                "GT" | "HI" => Some("gt"),
                "GE" | "HS" => Some("ge"),
                _ => None,
            }
        })
        .unwrap_or("eq")
        .to_string()
}

fn format_address_operand(operand: &SassOperand) -> String {
    match operand {
        SassOperand::ConstantBank { bank, offset } => {
            format!("[c[0x{:x}][0x{:x}]]", bank, offset)
        }
        SassOperand::Memory {
            base,
            offset,
            index,
            scale,
        } => {
            let mut expr = base
                .as_ref()
                .map(format_register)
                .unwrap_or_else(|| "0".to_string());
            if let Some(index) = index {
                if expr != "0" {
                    expr.push('+');
                } else {
                    expr.clear();
                }
                expr.push_str(&format_register(index));
                if *scale != 1 {
                    expr.push_str(&format!("*{}", scale));
                }
            }
            if *offset > 0 {
                if expr.is_empty() {
                    expr.push_str(&offset.to_string());
                } else {
                    expr.push_str(&format!("+{}", offset));
                }
            } else if *offset < 0 {
                expr.push_str(&offset.to_string());
            }
            if expr.is_empty() {
                expr.push('0');
            }
            format!("[{}]", expr)
        }
        SassOperand::Address(address) => format!("[{}]", label_for_address(*address)),
        SassOperand::Label(label) => {
            desc_address_to_ptx(label).unwrap_or_else(|| format!("[{}]", label))
        }
        _ => format!("[{}]", format_operand(operand)),
    }
}

fn dest_operand(inst: &EnhancedSassInstruction) -> Option<String> {
    inst.dest_operands.first().map(format_operand)
}

fn dest_rd_operand(inst: &EnhancedSassInstruction) -> Option<String> {
    inst.dest_operands.first().map(format_rd_operand)
}

fn predicate_prefix(inst: &EnhancedSassInstruction) -> String {
    match &inst.predicate {
        Some(SassOperand::Predicate { register, negated }) => {
            if *negated {
                format!("@!{} ", format_register(register))
            } else {
                format!("@{} ", format_register(register))
            }
        }
        _ => String::new(),
    }
}

fn format_rd_operand(operand: &SassOperand) -> String {
    match operand {
        SassOperand::Register(reg) if reg.prefix == "R" && !reg.is_zero => {
            format!("%rd{}", reg.number)
        }
        SassOperand::Label(label) => desc_address_to_ptx(label).unwrap_or_else(|| label.clone()),
        _ => format_operand(operand),
    }
}

fn constant_bank_operand(operand: &SassOperand) -> Option<(u32, u32)> {
    match operand {
        SassOperand::ConstantBank { bank, offset } => Some((*bank, *offset)),
        _ => None,
    }
}

fn has_modifier(inst: &EnhancedSassInstruction, modifier: &str) -> bool {
    inst.modifiers
        .iter()
        .any(|m| m.eq_ignore_ascii_case(modifier))
}

fn is_64bit_modifier(inst: &EnhancedSassInstruction) -> bool {
    has_modifier(inst, "64")
        || matches!(
            inst.data_type,
            Some(SassDataType::U64 | SassDataType::S64 | SassDataType::B64)
        )
}

fn is_extended_iadd3(inst: &EnhancedSassInstruction) -> bool {
    inst.opcode == "IADD3" && inst.src_operands.len() > 3
}

fn is_lop3_xor(inst: &EnhancedSassInstruction) -> bool {
    is_lop3_binary_truth_table(inst, 0x3c)
}

fn is_lop3_and(inst: &EnhancedSassInstruction) -> bool {
    is_lop3_binary_truth_table(inst, 0xc0)
}

fn is_lop3_or(inst: &EnhancedSassInstruction) -> bool {
    is_lop3_binary_truth_table(inst, 0xfc)
}

fn is_lop3_binary_truth_table(inst: &EnhancedSassInstruction, lut: i64) -> bool {
    inst.opcode == "LOP3"
        && inst.src_operands.len() >= 4
        && matches!(inst.src_operands.get(3), Some(SassOperand::Immediate(value)) if *value == lut)
        && inst
            .src_operands
            .get(2)
            .map(is_zero_register_operand)
            .unwrap_or(false)
}

fn is_lop3_odd_predicate(inst: &EnhancedSassInstruction) -> bool {
    if inst.opcode != "LOP3" || inst.src_operands.len() < 5 {
        return false;
    }
    let dest_is_pred = matches!(
        inst.dest_operands.first(),
        Some(SassOperand::Register(reg)) if reg.prefix == "P"
    );
    dest_is_pred
        && inst
            .src_operands
            .first()
            .map(is_zero_register_operand)
            .unwrap_or(false)
        && matches!(inst.src_operands.get(2), Some(SassOperand::Immediate(1)))
        && matches!(inst.src_operands.get(4), Some(SassOperand::Immediate(0xc0)))
}

fn is_shf_left_rotate(inst: &EnhancedSassInstruction) -> bool {
    inst.opcode == "SHF"
        && has_modifier(inst, "HI")
        && !has_modifier(inst, "R")
        && shift_amount_immediate(inst)
            .map(|amount| (1..32).contains(&amount))
            .unwrap_or(false)
        && operands_same_register(inst.src_operands.first(), inst.src_operands.get(2))
}

fn shift_amount_immediate(inst: &EnhancedSassInstruction) -> Option<i64> {
    match inst.src_operands.get(1) {
        Some(SassOperand::Immediate(amount)) => Some(*amount),
        _ => None,
    }
}

fn operands_same_register(lhs: Option<&SassOperand>, rhs: Option<&SassOperand>) -> bool {
    matches!((lhs, rhs), (Some(SassOperand::Register(a)), Some(SassOperand::Register(b))) if a == b)
}

fn is_zero_register_operand(operand: &SassOperand) -> bool {
    matches!(operand, SassOperand::Register(reg) if reg.is_zero)
}

fn is_pt_register_operand(operand: &SassOperand) -> bool {
    matches!(operand, SassOperand::Register(reg) if reg.prefix == "PT")
}

fn desc_address_to_ptx(label: &str) -> Option<String> {
    let marker = "][R";
    let reg_start = label.find(marker)? + marker.len();
    let rest = &label[reg_start..];
    let digits: String = rest.chars().take_while(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let reg_num = digits.parse::<u32>().ok()?;
    Some(format!("[%rd{}]", reg_num))
}

fn uses_cuda_param_abi(instructions: &[EnhancedSassInstruction]) -> bool {
    instructions.iter().any(|inst| {
        inst.src_operands
            .iter()
            .chain(inst.dest_operands.iter())
            .filter_map(constant_bank_operand)
            .any(|(bank, offset)| {
                bank == 0 && matches!(offset, 0x358 | 0x360 | 0x380 | 0x388 | 0x390)
            })
    })
}

fn uses_shared_memory(instructions: &[EnhancedSassInstruction]) -> bool {
    instructions.iter().any(|inst| {
        matches!(inst.memory_space, Some(SassMemorySpace::Shared))
            || matches!(inst.opcode.as_str(), "LDS" | "STS")
    })
}

fn format_operand(operand: &SassOperand) -> String {
    match operand {
        SassOperand::Register(reg) => format_register(reg),
        SassOperand::Predicate { register, negated } => {
            if *negated {
                format!("!{}", format_register(register))
            } else {
                format_register(register)
            }
        }
        SassOperand::Immediate(value) => value.to_string(),
        SassOperand::FloatImmediate(value) => value.to_string(),
        SassOperand::ConstantBank { bank, offset } => format!("c[0x{:x}][0x{:x}]", bank, offset),
        SassOperand::Memory { base, offset, .. } => {
            let base = base
                .as_ref()
                .map(format_register)
                .unwrap_or_else(|| "0".to_string());
            if *offset == 0 {
                format!("[{}]", base)
            } else if *offset > 0 {
                format!("[{}+{}]", base, offset)
            } else {
                format!("[{}{}]", base, offset)
            }
        }
        SassOperand::Barrier(id) => id.to_string(),
        SassOperand::SpecialRegister(name) => map_special_register(name),
        SassOperand::Label(label) => label.clone(),
        SassOperand::Address(address) => label_for_address(*address),
    }
}

fn format_signed_operand(operand: &SassOperand) -> (String, bool) {
    match operand {
        SassOperand::Register(reg) if reg.prefix.starts_with('-') => {
            (format_register_without_negation(reg), true)
        }
        SassOperand::Immediate(value) if *value < 0 => (value.abs().to_string(), true),
        _ => (format_operand(operand), false),
    }
}

fn format_float_operand(operand: &SassOperand) -> String {
    match operand {
        SassOperand::Immediate(value) => format!("{}.0", value),
        SassOperand::FloatImmediate(value) => value.to_string(),
        _ => format_operand(operand),
    }
}

fn extract_sass_encoding(inst: &EnhancedSassInstruction) -> Option<u64> {
    let marker = "/* 0x";
    let start = inst.instruction_text.rfind(marker)? + marker.len();
    let end = inst.instruction_text[start..].find(" */")? + start;
    u64::from_str_radix(&inst.instruction_text[start..end], 16).ok()
}

fn format_register_without_negation(reg: &SassRegister) -> String {
    let prefix = reg.prefix.trim_start_matches('-');
    if reg.is_zero {
        return "%rz".to_string();
    }
    match prefix {
        "R" => format!("%r{}", reg.number),
        "UR" => format!("%ur{}", reg.number),
        "P" => format!("%p{}", reg.number),
        "UP" => format!("%up{}", reg.number),
        _ => format_register(reg),
    }
}

fn format_sel_predicate_operand(operand: &SassOperand) -> (String, bool) {
    match operand {
        SassOperand::Predicate { register, negated } => (format_register(register), *negated),
        SassOperand::Register(reg) if reg.prefix == "P" => (format_register(reg), false),
        SassOperand::Register(reg) if reg.prefix == "!P" => (format!("%p{}", reg.number), true),
        _ => (format_operand(operand), false),
    }
}

fn format_register(reg: &SassRegister) -> String {
    if reg.is_zero {
        return "0".to_string();
    }
    match reg.prefix.as_str() {
        "R" => format!("%r{}", reg.number),
        "P" => format!("%p{}", reg.number),
        "UR" => format!("%ur{}", reg.number),
        "UP" => format!("%up{}", reg.number),
        _ => format!("%r{}", reg.number),
    }
}

fn map_special_register(name: &str) -> String {
    match name {
        "SR_TID.X" | "SR_TIDX" => "%tid.x".to_string(),
        "SR_TID.Y" | "SR_TIDY" => "%tid.y".to_string(),
        "SR_TID.Z" | "SR_TIDZ" => "%tid.z".to_string(),
        "SR_CTAID.X" | "SR_CTAIDX" => "%ctaid.x".to_string(),
        "SR_CTAID.Y" | "SR_CTAIDY" => "%ctaid.y".to_string(),
        "SR_CTAID.Z" | "SR_CTAIDZ" => "%ctaid.z".to_string(),
        "SR_NTID.X" | "SR_NTIDX" => "%ntid.x".to_string(),
        "SR_NTID.Y" | "SR_NTIDY" => "%ntid.y".to_string(),
        "SR_NTID.Z" | "SR_NTIDZ" => "%ntid.z".to_string(),
        "SR_NCTAID.X" | "SR_NCTAIDX" => "%nctaid.x".to_string(),
        "SR_NCTAID.Y" | "SR_NCTAIDY" => "%nctaid.y".to_string(),
        "SR_NCTAID.Z" | "SR_NCTAIDZ" => "%nctaid.z".to_string(),
        "SR_CgaCtaId" | "SR_CGACTAID" => "0".to_string(),
        "SR_LANEID" => "%laneid".to_string(),
        "SR_WARPID" => "%warpid".to_string(),
        "SR_CLOCK" | "SR_CLOCK_LO" => "%clock".to_string(),
        "SR_CLOCK_HI" => "%clock_hi".to_string(),
        _ => format!("%{}", name.trim_start_matches("SR_").to_ascii_lowercase()),
    }
}

fn branch_target(inst: &EnhancedSassInstruction) -> Option<u64> {
    inst.dest_operands
        .iter()
        .chain(inst.src_operands.iter())
        .find_map(|operand| match operand {
            SassOperand::Immediate(value) if *value >= 0 => Some(*value as u64),
            SassOperand::Address(address) => Some(*address),
            _ => None,
        })
}

fn sanitize_ident(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch == '_' || ch.is_ascii_alphanumeric() && (i > 0 || !ch.is_ascii_digit()) {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "kernel".to_string()
    } else {
        out
    }
}

fn label_for_address(address: u64) -> String {
    format!("L_{:04x}", address)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sass::{
        CubinKernel, EnhancedSassInstruction, ParsedCubin, SassDataType, SassMemorySpace,
        SassOpcodeClass, SassOperand, SassRegister,
    };
    use std::collections::HashMap;

    fn reg(n: u32) -> SassOperand {
        SassOperand::Register(SassRegister::new("R", n))
    }

    fn pred(n: u32) -> SassOperand {
        SassOperand::Predicate {
            register: SassRegister::new("P", n),
            negated: false,
        }
    }

    fn mem(base: u32, offset: i64) -> SassOperand {
        SassOperand::Memory {
            base: Some(SassRegister::new("R", base)),
            offset,
            index: None,
            scale: 1,
        }
    }

    #[test]
    fn sass_lifter_text_frontend_uses_function_name_and_sm120() {
        let text = r#"Function : vector_add
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   LDG.E.U32 R1, [R2] ;
        /*0020*/                   STG.E.U32 [R3], R1 ;
        /*0030*/                   EXIT ;
"#;

        let result = lift_sass_text_to_ptx(
            text,
            SassLiftOptions {
                sm_version: 120,
                kernel_name: String::new(),
                include_sass_comments: true,
                emit_unsupported_comments: true,
            },
        )
        .expect("text frontend should lift cuobjdump text");

        assert!(result.ptx.contains(".visible .entry vector_add()"));
        assert!(result.ptx.contains(".target sm_120"));
        assert!(result.ptx.contains("ld.global.u32 %r1, [%r2];"));
        assert!(result.ptx.contains("st.global.u32 [%r3], %r1;"));
    }

    #[test]
    fn sass_lifter_text_frontend_default_options_infer_single_function_name() {
        let text = r#"Function : vector_add
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   EXIT ;
"#;

        let result = lift_sass_text_to_ptx(text, SassLiftOptions::default())
            .expect("default text frontend should infer an unambiguous function name");

        assert!(result.ptx.contains(".visible .entry vector_add()"));
    }

    #[test]
    fn sass_lifter_text_frontend_rejects_empty_input() {
        let err = lift_sass_text_to_ptx("", SassLiftOptions::default())
            .expect_err("empty text should not lift");

        assert!(err.contains("No SASS instructions parsed"), "{err}");
    }

    #[test]
    fn sass_lifter_text_frontend_rejects_ambiguous_multi_function_input() {
        let text = r#"Function : first
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   EXIT ;
Function : second
        /*0020*/                   IADD R4, R5, 7 ;
        /*0030*/                   EXIT ;
"#;

        let err = lift_sass_text_to_ptx(text, SassLiftOptions::default())
            .expect_err("default multi-function text should require a selector");
        assert!(err.contains("Multiple SASS functions parsed"), "{err}");

        let err = lift_sass_text_to_ptx(
            text,
            SassLiftOptions {
                kernel_name: String::new(),
                ..SassLiftOptions::default()
            },
        )
        .expect_err("empty kernel_name should require a selector for multi-function text");

        assert!(err.contains("Multiple SASS functions parsed"), "{err}");
    }

    #[test]
    fn sass_lifter_text_frontend_selects_requested_function_only() {
        let text = r#"Function : first
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   EXIT ;
Function : second
        /*0020*/                   IADD R4, R5, 7 ;
        /*0030*/                   EXIT ;
"#;

        let result = lift_sass_text_to_ptx(
            text,
            SassLiftOptions {
                kernel_name: "second".to_string(),
                include_sass_comments: false,
                ..SassLiftOptions::default()
            },
        )
        .expect("explicit selector should lift only the requested function");

        assert!(result.ptx.contains(".visible .entry second()"));
        assert!(result.ptx.contains("add.s32 %r4, %r5, 7;"));
        assert!(result.ptx.contains("L_0020:"));
        assert!(!result.ptx.contains("mov.u32 %r0, %tid.x;"));
        assert!(!result.ptx.contains("L_0000:"));
    }

    #[test]
    fn sass_lifter_text_frontend_selects_literal_kernel_function_name() {
        let text = r#"Function : first
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   EXIT ;
Function : kernel
        /*0020*/                   IADD R4, R5, 7 ;
        /*0030*/                   EXIT ;
"#;

        let result = lift_sass_text_to_ptx(
            text,
            SassLiftOptions {
                kernel_name: "kernel".to_string(),
                include_sass_comments: false,
                ..SassLiftOptions::default()
            },
        )
        .expect("literal kernel selector should select Function : kernel");

        assert!(result.ptx.contains(".visible .entry kernel()"));
        assert!(result.ptx.contains("add.s32 %r4, %r5, 7;"));
        assert!(result.ptx.contains("L_0020:"));
        assert!(!result.ptx.contains("mov.u32 %r0, %tid.x;"));
        assert!(!result.ptx.contains("L_0000:"));
    }

    #[test]
    fn sass_lifter_cubin_selection_uses_requested_kernel_name() {
        let parsed = ParsedCubin {
            sm_version: 120,
            ptx_version: None,
            kernels: vec![
                test_cubin_kernel("first", 0x100),
                test_cubin_kernel("kernel", 0x180),
                test_cubin_kernel("second", 0x200),
            ],
            constants: Vec::new(),
            debug_lines: HashMap::new(),
            symbols: Vec::new(),
            sections: HashMap::new(),
        };

        let selected =
            select_cubin_kernel(&parsed, "second").expect("requested kernel should be selected");
        assert_eq!(selected.name, "second");
        assert_eq!(selected.address, 0x200);

        let selected =
            select_cubin_kernel(&parsed, "").expect("empty selector should use the first kernel");
        assert_eq!(selected.name, "first");
        assert_eq!(selected.address, 0x100);

        let selected = select_cubin_kernel(&parsed, "kernel")
            .expect("literal kernel selector should select the kernel named kernel");
        assert_eq!(selected.name, "kernel");
        assert_eq!(selected.address, 0x180);

        let err = select_cubin_kernel(&parsed, "missing")
            .expect_err("missing requested kernel should fail");
        assert_eq!(err, "No kernel named 'missing' found in CUBIN");
    }

    #[test]
    fn sass_lifter_cubin_frontend_rejects_malformed_input() {
        let err = lift_cubin_to_ptx(b"not an elf", SassLiftOptions::default())
            .expect_err("malformed CUBIN should fail");

        assert!(err.contains("Failed to parse CUBIN"), "{err}");
    }

    fn test_cubin_kernel(name: &str, address: u64) -> CubinKernel {
        CubinKernel {
            name: name.to_string(),
            address,
            size: 0,
            code: Vec::new(),
            num_registers: 0,
            shared_mem_size: 0,
            const_mem_size: 0,
            local_mem_size: 0,
            max_threads_per_block: 0,
            sm_version: 120,
        }
    }

    #[test]
    fn sass_lifter_emits_complete_module_for_sm120_text() {
        let mut s2r = EnhancedSassInstruction::new("S2R".to_string(), 0x0);
        s2r.opcode_class = SassOpcodeClass::SpecialRegRead;
        s2r.data_type = Some(SassDataType::U32);
        s2r.dest_operands.push(reg(0));
        s2r.src_operands
            .push(SassOperand::SpecialRegister("SR_TID.X".to_string()));

        let mut fadd = EnhancedSassInstruction::new("FADD".to_string(), 0x10);
        fadd.opcode_class = SassOpcodeClass::FloatArithmetic;
        fadd.data_type = Some(SassDataType::F32);
        fadd.dest_operands.push(reg(1));
        fadd.src_operands.push(reg(2));
        fadd.src_operands.push(reg(3));

        let mut exit = EnhancedSassInstruction::new("EXIT".to_string(), 0x20);
        exit.opcode_class = SassOpcodeClass::Exit;

        let result = lift_instructions_to_ptx(
            &[s2r, fadd, exit],
            &SassLiftOptions {
                sm_version: 120,
                kernel_name: "sm120_kernel".to_string(),
                include_sass_comments: true,
                emit_unsupported_comments: true,
            },
        );

        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(result.ptx.contains(".version 8.5"));
        assert!(result.ptx.contains(".target sm_120"));
        assert!(result.ptx.contains(".visible .entry sm120_kernel()"));
        assert!(result.ptx.contains(".reg .b32 %r<4>;"));
        assert!(result.ptx.contains("L_0000:"));
        assert!(result.ptx.contains("mov.u32 %r0, %tid.x;"));
        assert!(result.ptx.contains("add.f32 %r1, %r2, %r3;"));
        assert!(result.ptx.contains("ret;"));
    }

    #[test]
    fn sass_lifter_normalizes_load_store_and_constant_memory() {
        let mut ldg = EnhancedSassInstruction::new("LDG".to_string(), 0x0);
        ldg.opcode_class = SassOpcodeClass::GlobalLoad;
        ldg.memory_space = Some(SassMemorySpace::Global);
        ldg.data_type = Some(SassDataType::U32);
        ldg.dest_operands.push(reg(0));
        ldg.src_operands.push(mem(2, 16));

        let mut stg = EnhancedSassInstruction::new("STG".to_string(), 0x10);
        stg.opcode_class = SassOpcodeClass::GlobalStore;
        stg.memory_space = Some(SassMemorySpace::Global);
        stg.data_type = Some(SassDataType::F32);
        stg.dest_operands.push(mem(4, 0));
        stg.src_operands.push(reg(1));

        let mut ldc = EnhancedSassInstruction::new("LDC".to_string(), 0x20);
        ldc.opcode_class = SassOpcodeClass::ConstantLoad;
        ldc.memory_space = Some(SassMemorySpace::Constant);
        ldc.data_type = Some(SassDataType::U64);
        ldc.dest_operands.push(reg(6));
        ldc.src_operands.push(SassOperand::ConstantBank {
            bank: 0,
            offset: 0x160,
        });

        let result = lift_instructions_to_ptx(&[ldg, stg, ldc], &SassLiftOptions::default());

        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(result.ptx.contains("ld.global.u32 %r0, [%r2+16];"));
        assert!(result.ptx.contains("st.global.f32 [%r4], %r1;"));
        assert!(result.ptx.contains("ld.const.u64 %r6, [c[0x0][0x160]];"));
    }

    #[test]
    fn sass_lifter_preserves_predicates_and_branch_targets() {
        let mut isetp = EnhancedSassInstruction::new("ISETP".to_string(), 0x0);
        isetp.opcode_class = SassOpcodeClass::IntegerComparison;
        isetp.data_type = Some(SassDataType::S32);
        isetp.modifiers.push("LT".to_string());
        isetp.dest_operands.push(pred(0));
        isetp.src_operands.push(reg(1));
        isetp.src_operands.push(reg(2));

        let mut bra = EnhancedSassInstruction::new("BRA".to_string(), 0x10);
        bra.opcode_class = SassOpcodeClass::Branch;
        bra.predicate = Some(pred(0));
        bra.src_operands.push(SassOperand::Immediate(0x40));

        let mut iadd = EnhancedSassInstruction::new("IADD".to_string(), 0x20);
        iadd.opcode_class = SassOpcodeClass::IntegerArithmetic;
        iadd.data_type = Some(SassDataType::S32);
        iadd.dest_operands.push(reg(3));
        iadd.src_operands.push(reg(3));
        iadd.src_operands.push(SassOperand::Immediate(1));

        let mut ret = EnhancedSassInstruction::new("RET".to_string(), 0x40);
        ret.opcode_class = SassOpcodeClass::Exit;

        let result =
            lift_instructions_to_ptx(&[isetp, bra, iadd, ret], &SassLiftOptions::default());

        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(result.ptx.contains("setp.lt.s32 %p0, %r1, %r2;"));
        assert!(result.ptx.contains("@%p0 bra L_0040;"));
        assert!(result.ptx.contains("add.s32 %r3, %r3, 1;"));
        assert!(result.ptx.contains("L_0040:"));
    }

    #[test]
    fn sass_lifter_expands_three_source_iadd3_with_scratch_register() {
        let mut iadd3 = EnhancedSassInstruction::new("IADD3".to_string(), 0x0);
        iadd3.opcode_class = SassOpcodeClass::IntegerArithmetic;
        iadd3.data_type = Some(SassDataType::S32);
        iadd3.dest_operands.push(reg(2));
        iadd3.src_operands.push(reg(9));
        iadd3.src_operands.push(reg(1));
        iadd3.src_operands.push(reg(4));

        let result = lift_instructions_to_ptx(&[iadd3], &SassLiftOptions::default());

        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(result.ptx.contains(".reg .b32 %r<11>;"));
        assert!(result.ptx.contains("add.s32 %r10, %r9, %r1;"));
        assert!(result.ptx.contains("add.s32 %r2, %r10, %r4;"));
    }

    #[test]
    fn sass_lifter_reports_lop3_instead_of_lowering_to_and() {
        let mut lop3 = EnhancedSassInstruction::new("LOP3".to_string(), 0x0);
        lop3.data_type = Some(SassDataType::B32);
        lop3.dest_operands.push(reg(0));
        lop3.src_operands.push(reg(1));
        lop3.src_operands.push(reg(2));
        lop3.src_operands.push(reg(3));
        lop3.src_operands.push(SassOperand::Immediate(0xca));

        let result = lift_instructions_to_ptx(&[lop3], &SassLiftOptions::default());

        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0].opcode, "LOP3");
        assert_eq!(
            result.diagnostics[0].message,
            "LOP3 truth-table lifting is not implemented"
        );
        assert!(result.ptx.contains(
            "unsupported SASS LOP3 at 0x0000: LOP3 truth-table lifting is not implemented"
        ));
        assert!(!result.ptx.contains("and.b32"));
    }

    #[test]
    fn sass_lifter_emits_rounded_ffma() {
        let mut ffma = EnhancedSassInstruction::new("FFMA".to_string(), 0x0);
        ffma.opcode_class = SassOpcodeClass::FloatArithmetic;
        ffma.data_type = Some(SassDataType::F32);
        ffma.dest_operands.push(reg(0));
        ffma.src_operands.push(reg(1));
        ffma.src_operands.push(reg(2));
        ffma.src_operands.push(reg(3));

        let result = lift_instructions_to_ptx(&[ffma], &SassLiftOptions::default());

        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(result.ptx.contains("fma.rn.f32 %r0, %r1, %r2, %r3;"));
    }

    #[test]
    fn sass_lifter_reports_psetp_instead_of_emitting_pred_setp() {
        let mut psetp = EnhancedSassInstruction::new("PSETP".to_string(), 0x0);
        psetp.data_type = Some(SassDataType::Pred);
        psetp.dest_operands.push(pred(0));
        psetp.src_operands.push(pred(1));
        psetp.src_operands.push(pred(2));

        let result = lift_instructions_to_ptx(&[psetp], &SassLiftOptions::default());

        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0].opcode, "PSETP");
        assert_eq!(
            result.diagnostics[0].message,
            "predicate set lifting is not implemented"
        );
        assert!(result.ptx.contains(
            "unsupported SASS PSETP at 0x0000: predicate set lifting is not implemented"
        ));
        assert!(!result.ptx.contains("setp.eq.pred"));
    }

    #[test]
    fn sass_lifter_reports_unsupported_tensor_instruction() {
        let hmma = EnhancedSassInstruction::new("HMMA".to_string(), 0x120);

        let result = lift_instructions_to_ptx(&[hmma], &SassLiftOptions::default());

        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0].opcode, "HMMA");
        assert_eq!(
            result.diagnostics[0].message,
            "tensor instruction lifting is not implemented"
        );
        assert!(result.ptx.contains("unsupported SASS HMMA at 0x0120"));
    }
}
