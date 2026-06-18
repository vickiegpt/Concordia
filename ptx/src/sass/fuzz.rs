use std::error::Error;
use std::fmt;

use super::{
    lift_instructions_to_ptx, EnhancedSassInstruction, SassDataType, SassLiftOptions,
    SassOpcodeClass, SassOperand, SassRegister,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLifterFuzzConfig {
    pub seed: u64,
    pub cases: usize,
    pub max_instructions: usize,
    pub sm_version: u32,
    pub parse_lifted_ptx: bool,
}

impl Default for SassLifterFuzzConfig {
    fn default() -> Self {
        Self {
            seed: 0x5a55_1200,
            cases: 1024,
            max_instructions: 32,
            sm_version: 120,
            parse_lifted_ptx: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLifterFuzzSummary {
    pub seed: u64,
    pub cases: usize,
    pub instructions: usize,
    pub lift_diagnostics: usize,
    pub parse_failures: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassLifterFuzzError {
    pub case_index: Option<usize>,
    pub message: String,
}

impl SassLifterFuzzError {
    fn config(message: impl Into<String>) -> Self {
        Self {
            case_index: None,
            message: message.into(),
        }
    }

    fn case(case_index: usize, message: impl Into<String>) -> Self {
        Self {
            case_index: Some(case_index),
            message: message.into(),
        }
    }
}

impl fmt::Display for SassLifterFuzzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.case_index {
            Some(case_index) => write!(f, "case {}: {}", case_index, self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl Error for SassLifterFuzzError {}

pub fn run_sass_lifter_fuzzer(
    config: SassLifterFuzzConfig,
) -> Result<SassLifterFuzzSummary, SassLifterFuzzError> {
    if config.cases == 0 {
        return Err(SassLifterFuzzError::config(
            "cases must be greater than zero",
        ));
    }
    if config.max_instructions == 0 {
        return Err(SassLifterFuzzError::config(
            "max_instructions must be greater than zero",
        ));
    }

    let mut rng = Lcg::new(config.seed);
    let mut summary = SassLifterFuzzSummary {
        seed: config.seed,
        cases: 0,
        instructions: 0,
        lift_diagnostics: 0,
        parse_failures: 0,
    };

    for case_index in 0..config.cases {
        let instructions = generate_case(&mut rng, case_index, config.max_instructions);
        let options = SassLiftOptions {
            sm_version: config.sm_version,
            kernel_name: format!("fuzz_case_{}", case_index),
            include_sass_comments: false,
            emit_unsupported_comments: false,
        };
        let result = lift_instructions_to_ptx(&instructions, &options);
        if !result.diagnostics.is_empty() {
            summary.lift_diagnostics += result.diagnostics.len();
            let diagnostic = &result.diagnostics[0];
            return Err(SassLifterFuzzError::case(
                case_index,
                format!(
                    "unexpected lift diagnostic at {:?} for {}: {}",
                    diagnostic.address, diagnostic.opcode, diagnostic.message
                ),
            ));
        }

        if config.parse_lifted_ptx {
            if let Err(errors) = ptx_parser::parse_module_checked(&result.ptx) {
                summary.parse_failures += 1;
                return Err(SassLifterFuzzError::case(
                    case_index,
                    format!("lifted PTX did not parse: {:?}", errors),
                ));
            }
        }

        summary.cases += 1;
        summary.instructions += instructions.len();
    }

    Ok(summary)
}

fn generate_case(
    rng: &mut Lcg,
    case_index: usize,
    max_instructions: usize,
) -> Vec<EnhancedSassInstruction> {
    let body_len = if max_instructions == 1 {
        0
    } else {
        1 + rng.usize(max_instructions - 1)
    };
    let exit_address = (body_len as u64) * 0x10;
    let mut instructions = Vec::with_capacity(body_len + 1);

    for instruction_index in 0..body_len {
        let address = (instruction_index as u64) * 0x10;
        instructions.push(generate_instruction(rng, case_index, address, exit_address));
    }

    let mut exit = EnhancedSassInstruction::new("EXIT".to_string(), exit_address);
    exit.opcode_class = SassOpcodeClass::Exit;
    exit.instruction_text = "EXIT ;".to_string();
    instructions.push(exit);
    instructions
}

fn generate_instruction(
    rng: &mut Lcg,
    case_index: usize,
    address: u64,
    exit_address: u64,
) -> EnhancedSassInstruction {
    match rng.usize(13) {
        0 => {
            let mut inst = new_inst("S2R", address);
            inst.opcode_class = SassOpcodeClass::SpecialRegRead;
            inst.data_type = Some(SassDataType::U32);
            inst.dest_operands.push(reg(rng.gpr()));
            inst.src_operands
                .push(SassOperand::SpecialRegister("SR_TID.X".to_string()));
            inst
        }
        1 => {
            let mut inst = new_inst("MOV", address);
            inst.opcode_class = SassOpcodeClass::Move;
            inst.data_type = Some(SassDataType::U32);
            inst.dest_operands.push(reg(rng.gpr()));
            inst.src_operands.push(reg(rng.gpr()));
            maybe_predicate(rng, &mut inst);
            inst
        }
        2 => {
            let mut inst = new_inst("MOV32I", address);
            inst.opcode_class = SassOpcodeClass::Move;
            inst.data_type = Some(SassDataType::U32);
            inst.dest_operands.push(reg(rng.gpr()));
            inst.src_operands.push(SassOperand::Immediate(rng.imm20()));
            maybe_predicate(rng, &mut inst);
            inst
        }
        3 => binary_int_inst(rng, "IADD", address, SassOpcodeClass::IntegerArithmetic),
        4 => {
            let mut inst = new_inst("IADD3", address);
            inst.opcode_class = SassOpcodeClass::IntegerArithmetic;
            inst.data_type = Some(SassDataType::S32);
            inst.dest_operands.push(reg(rng.gpr()));
            inst.src_operands.push(reg(rng.gpr()));
            inst.src_operands.push(reg_or_imm(rng));
            inst.src_operands.push(reg_or_imm(rng));
            maybe_predicate(rng, &mut inst);
            inst
        }
        5 => binary_int_inst(rng, "IMUL", address, SassOpcodeClass::IntegerArithmetic),
        6 => {
            let mut inst = new_inst("IMAD", address);
            inst.opcode_class = SassOpcodeClass::IntegerArithmetic;
            inst.data_type = Some(SassDataType::S32);
            inst.dest_operands.push(reg(rng.gpr()));
            inst.src_operands.push(reg(rng.gpr()));
            inst.src_operands.push(reg_or_imm(rng));
            inst.src_operands.push(reg(rng.gpr()));
            maybe_predicate(rng, &mut inst);
            inst
        }
        7 => shift_inst(rng, "SHL", address),
        8 => shift_inst(rng, "SHR", address),
        9 => binary_int_inst(rng, "LOP", address, SassOpcodeClass::IntegerLogical),
        10 => {
            let mut inst = new_inst("POPC", address);
            inst.opcode_class = SassOpcodeClass::IntegerLogical;
            inst.data_type = Some(SassDataType::B32);
            inst.dest_operands.push(reg(rng.gpr()));
            inst.src_operands.push(reg(rng.gpr()));
            maybe_predicate(rng, &mut inst);
            inst
        }
        11 => {
            let mut inst = new_inst("ISETP", address);
            inst.opcode_class = SassOpcodeClass::IntegerComparison;
            inst.data_type = Some(SassDataType::S32);
            inst.modifiers.push(cmp_modifier(rng).to_string());
            inst.dest_operands.push(pred(rng.pred()));
            inst.src_operands.push(reg(rng.gpr()));
            inst.src_operands.push(reg_or_imm(rng));
            inst
        }
        _ => {
            let mut inst = new_inst("BRA", address);
            inst.opcode_class = SassOpcodeClass::Branch;
            inst.src_operands.push(SassOperand::Address(exit_address));
            if case_index % 2 == 0 {
                maybe_predicate(rng, &mut inst);
            }
            inst
        }
    }
}

fn binary_int_inst(
    rng: &mut Lcg,
    opcode: &str,
    address: u64,
    class: SassOpcodeClass,
) -> EnhancedSassInstruction {
    let mut inst = new_inst(opcode, address);
    inst.opcode_class = class;
    inst.data_type = Some(SassDataType::S32);
    inst.dest_operands.push(reg(rng.gpr()));
    inst.src_operands.push(reg(rng.gpr()));
    inst.src_operands.push(reg_or_imm(rng));
    maybe_predicate(rng, &mut inst);
    inst
}

fn shift_inst(rng: &mut Lcg, opcode: &str, address: u64) -> EnhancedSassInstruction {
    let mut inst = new_inst(opcode, address);
    inst.opcode_class = SassOpcodeClass::IntegerLogical;
    inst.data_type = Some(SassDataType::U32);
    inst.dest_operands.push(reg(rng.gpr()));
    inst.src_operands.push(reg(rng.gpr()));
    inst.src_operands
        .push(SassOperand::Immediate(rng.usize(32) as i64));
    maybe_predicate(rng, &mut inst);
    inst
}

fn new_inst(opcode: &str, address: u64) -> EnhancedSassInstruction {
    let mut inst = EnhancedSassInstruction::new(opcode.to_string(), address);
    inst.instruction_text = format!("{} ;", opcode);
    inst
}

fn maybe_predicate(rng: &mut Lcg, inst: &mut EnhancedSassInstruction) {
    if rng.usize(4) == 0 {
        inst.predicate = Some(SassOperand::Predicate {
            register: SassRegister::new("P", rng.pred()),
            negated: rng.usize(2) == 1,
        });
    }
}

fn reg(number: u32) -> SassOperand {
    SassOperand::Register(SassRegister::new("R", number))
}

fn pred(number: u32) -> SassOperand {
    SassOperand::Predicate {
        register: SassRegister::new("P", number),
        negated: false,
    }
}

fn reg_or_imm(rng: &mut Lcg) -> SassOperand {
    if rng.usize(3) == 0 {
        SassOperand::Immediate(rng.imm20())
    } else {
        reg(rng.gpr())
    }
}

fn cmp_modifier(rng: &mut Lcg) -> &'static str {
    match rng.usize(6) {
        0 => ".EQ",
        1 => ".NE",
        2 => ".LT",
        3 => ".LE",
        4 => ".GT",
        _ => ".GE",
    }
}

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn usize(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u32() as usize) % upper_exclusive
    }

    fn gpr(&mut self) -> u32 {
        self.usize(32) as u32
    }

    fn pred(&mut self) -> u32 {
        self.usize(7) as u32
    }

    fn imm20(&mut self) -> i64 {
        (self.usize(2048) as i64) - 1024
    }
}
