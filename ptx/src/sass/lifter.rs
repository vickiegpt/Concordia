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
