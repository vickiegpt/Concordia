use std::collections::HashSet;

use super::{
    EnhancedSassInstruction, SassOpcodeClass, SassOperand, SassRegister,
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
}

impl<'a> LiftContext<'a> {
    fn new(options: &'a SassLiftOptions) -> Self {
        Self {
            options,
            output: String::new(),
            diagnostics: Vec::new(),
            branch_targets: HashSet::new(),
        }
    }

    fn emit_module(&mut self, instructions: &[EnhancedSassInstruction]) {
        self.collect_branch_targets(instructions);
        let regs = RegisterDecls::from_instructions(instructions);

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
            "FADD" => Some(binary_op(inst, &pred, "add", "f32")),
            "EXIT" | "RET" => Some(format!("{}ret;", pred)),
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
        EnhancedSassInstruction, SassDataType, SassOpcodeClass, SassOperand, SassRegister,
    };

    fn reg(n: u32) -> SassOperand {
        SassOperand::Register(SassRegister::new("R", n))
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
}
