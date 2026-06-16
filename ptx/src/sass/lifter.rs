use std::collections::HashSet;

use super::{
    EnhancedSassInstruction, SassDataType, SassMemorySpace, SassOpcodeClass, SassOperand,
    SassRegister,
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
            kernel_name: "kernel".to_string(),
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

struct LiftContext<'a> {
    options: &'a SassLiftOptions,
    output: String,
    diagnostics: Vec<SassLiftDiagnostic>,
    branch_targets: HashSet<u64>,
    scratch_gpr: Option<String>,
}

impl<'a> LiftContext<'a> {
    fn new(options: &'a SassLiftOptions) -> Self {
        Self {
            options,
            output: String::new(),
            diagnostics: Vec::new(),
            branch_targets: HashSet::new(),
            scratch_gpr: None,
        }
    }

    fn emit_module(&mut self, instructions: &[EnhancedSassInstruction]) {
        self.collect_branch_targets(instructions);
        let mut regs = RegisterDecls::from_instructions(instructions);
        if needs_iadd3_scratch(instructions) {
            self.scratch_gpr = Some(format!("%r{}", regs.max_gpr));
            regs.max_gpr += 1;
        }

        self.output.push_str(".version 8.5\n");
        self.output
            .push_str(&format!(".target sm_{}\n", self.options.sm_version));
        self.output.push_str(".address_size 64\n\n");
        self.output.push_str(&format!(
            ".visible .entry {}()\n{{\n",
            sanitize_ident(&self.options.kernel_name)
        ));

        if regs.max_gpr > 0 {
            self.output
                .push_str(&format!("    .reg .b32 %r<{}>;\n", regs.max_gpr));
        }
        if regs.max_pred > 0 {
            self.output
                .push_str(&format!("    .reg .pred %p<{}>;\n", regs.max_pred));
        }
        if regs.max_gpr > 0 || regs.max_pred > 0 {
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
            "S2R" | "CS2R" => Some(format!(
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
            "IADD" => Some(binary_op(
                inst,
                &pred,
                "add",
                &data_type_suffix(inst, SassDataType::S32),
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
            "LOP" if inst.src_operands.len() == 2 => Some(binary_op(inst, &pred, "and", "b32")),
            "LOP" => self.unsupported(inst, "logical operation lifting is not implemented"),
            "LOP3" => self.unsupported(inst, "LOP3 truth-table lifting is not implemented"),
            "POPC" => Some(unary_op(inst, &pred, "popc", "b32")),
            "FADD" => Some(binary_op(
                inst,
                &pred,
                "add",
                &data_type_suffix(inst, SassDataType::F32),
            )),
            "FMUL" => Some(binary_op(
                inst,
                &pred,
                "mul",
                &data_type_suffix(inst, SassDataType::F32),
            )),
            "FFMA" => Some(ternary_op(
                inst,
                &pred,
                "fma.rn",
                &data_type_suffix(inst, SassDataType::F32),
            )),
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
            "LDG" | "LDS" | "LDL" | "LDC" => Some(load_op(inst, &pred)),
            "STG" | "STS" | "STL" => Some(store_op(inst, &pred)),
            "ISETP" | "FSETP" => Some(setp_op(inst, &pred)),
            "PSETP" => self.unsupported(inst, "predicate set lifting is not implemented"),
            "BRA" | "BRX" | "JMP" => Some(branch_op(inst, &pred)),
            "BAR" => Some(format!("{}bar.sync 0;", pred)),
            "DEPBAR" => Some("// depbar preserved from SASS;".to_string()),
            "MEMBAR" => Some(format!("{}membar.gl;", pred)),
            "EXIT" | "RET" => Some(format!("{}ret;", pred)),
            "HMMA" | "IMMA" | "BMMA" | "DMMA" => {
                self.unsupported(inst, "tensor instruction lifting is not implemented")
            }
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

fn needs_iadd3_scratch(instructions: &[EnhancedSassInstruction]) -> bool {
    instructions
        .iter()
        .any(|inst| inst.opcode == "IADD3" && inst.src_operands.len() >= 3)
}

#[derive(Debug, Default)]
struct RegisterDecls {
    max_gpr: u32,
    max_pred: u32,
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
        _ => {}
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

fn load_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let addr = inst
        .src_operands
        .first()
        .map(format_address_operand)
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

fn store_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let addr = inst
        .dest_operands
        .first()
        .map(format_address_operand)
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

fn setp_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%p0".to_string());
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
        SassOperand::Label(label) => format!("[{}]", label),
        _ => format!("[{}]", format_operand(operand)),
    }
}

fn dest_operand(inst: &EnhancedSassInstruction) -> Option<String> {
    inst.dest_operands.first().map(format_operand)
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
        EnhancedSassInstruction, SassDataType, SassMemorySpace, SassOpcodeClass, SassOperand,
        SassRegister,
    };

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
