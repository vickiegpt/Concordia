# Full SASS Lifter SM120 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared SM120 SASS-to-PTX lifter that emits complete PTX modules from both disassembly text and raw CUBIN input.

**Architecture:** Add `ptx/src/sass/lifter.rs` as the single semantic lifting layer over `EnhancedSassInstruction`. Existing text parsing and CUBIN parsing stay as frontends, while `sass_inliner --recover-ptx` and `gpu_rr ptx` call the shared lifter instead of constructing PTX themselves.

**Tech Stack:** Rust workspace, `ptx` crate, existing `ptx::sass` parser/disassembler types, existing CLI binaries `sass_inliner` and `gpu_rr`, Cargo tests.

**Spec:** `docs/superpowers/specs/2026-06-16-full-sass-lifter-sm120-design.md`

---

## File Structure

### New File

`ptx/src/sass/lifter.rs`

Responsible for:
- `SassLiftOptions`, `SassLiftDiagnostic`, `SassLiftResult`
- `lift_instructions_to_ptx`
- `lift_sass_text_to_ptx`
- `lift_cubin_to_ptx`
- helper formatting for predicates, registers, memory operands, labels, data types, diagnostics, and instruction comments
- unit tests for the lifter core and frontend wrappers

### Modified Files

`ptx/src/sass/mod.rs`

Responsible for exporting the new lifter module:

```rust
pub mod lifter;
pub use lifter::*;
```

`ptx/src/lib.rs`

Responsible for re-exporting lifter APIs from the public `ptx` crate path.

`ptx/src/bin/sass_inliner.rs`

Responsible for replacing the bespoke `--recover-ptx` reconstruction path with calls into `ptx::sass::lift_instructions_to_ptx` after the current loader has produced instructions and a kernel name.

`ptx/src/bin/gpu_rr.rs`

Responsible for replacing the duplicate `run_ptx` reconstruction logic with `lift_cubin_to_ptx` or `lift_sass_text_to_ptx`.

`ptx/tests/sass_inliner_test.rs`

Responsible for integration coverage of `sass_inliner --stdin --recover-ptx --sm 120 -`.

---

## Task 1: Add Failing Lifter Core Test

**Files:**
- Create: `ptx/src/sass/lifter.rs`
- Modify: `ptx/src/sass/mod.rs`
- Test: `ptx/src/sass/lifter.rs`

- [ ] **Step 1: Create the failing test file**

Create `ptx/src/sass/lifter.rs` with this test-only content:

```rust
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
```

- [ ] **Step 2: Expose the module so the test compiles far enough to fail on missing API**

Append these lines to `ptx/src/sass/mod.rs`:

```rust
pub mod lifter;
pub use lifter::*;
```

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
cargo test -p ptx sass_lifter_emits_complete_module_for_sm120_text --lib
```

Expected: FAIL at compile time with unresolved items such as `cannot find function lift_instructions_to_ptx` and `cannot find struct SassLiftOptions`.

- [ ] **Step 4: Commit the failing test**

```bash
git add ptx/src/sass/lifter.rs ptx/src/sass/mod.rs
git commit -m "test: specify SM120 SASS lifter module output"
```

---

## Task 2: Implement Minimal Complete PTX Module Emission

**Files:**
- Modify: `ptx/src/sass/lifter.rs`
- Test: `ptx/src/sass/lifter.rs`

- [ ] **Step 1: Add the public types and top-level lifter function**

Replace the top of `ptx/src/sass/lifter.rs` before the `#[cfg(test)]` module with:

```rust
use std::collections::HashSet;

use super::{
    CubinParser, EnhancedSassInstruction, SassDataType, SassDisassembler, SassMemorySpace,
    SassOpcodeClass, SassOperand, SassRegister, TextDisassemblyParser,
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
```

- [ ] **Step 2: Add module context, header, declarations, and labels**

Add this implementation below `lift_instructions_to_ptx`:

```rust
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
            match self.lift_instruction(inst) {
                Some(line) => self.output.push_str(&format!("    {}\n", line)),
                None => {}
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
```

- [ ] **Step 3: Add minimal instruction lifting for the first test**

Add these helpers below the module context:

```rust
impl<'a> LiftContext<'a> {
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

fn binary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst
        .src_operands
        .get(0)
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
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
cargo test -p ptx sass_lifter_emits_complete_module_for_sm120_text --lib
```

Expected: PASS.

- [ ] **Step 5: Commit the minimal lifter**

```bash
git add ptx/src/sass/lifter.rs
git commit -m "feat: emit basic SM120 SASS lifter module"
```

---

## Task 3: Add Common Instruction Semantics, Predicates, Labels, and Diagnostics

**Files:**
- Modify: `ptx/src/sass/lifter.rs`
- Test: `ptx/src/sass/lifter.rs`

- [ ] **Step 1: Add failing tests for memory, branches, predicates, barriers, and unsupported tensor fallback**

Append these tests to the `#[cfg(test)] mod tests` block in `ptx/src/sass/lifter.rs`:

```rust
    fn mem(base: u32, offset: i64) -> SassOperand {
        SassOperand::Memory {
            base: Some(SassRegister::new("R", base)),
            offset,
            index: None,
            scale: 1,
        }
    }

    #[test]
    fn sass_lifter_normalizes_load_store_and_constant_memory() {
        let mut ldg = EnhancedSassInstruction::new("LDG".to_string(), 0x0);
        ldg.opcode_class = SassOpcodeClass::GlobalLoad;
        ldg.memory_space = Some(crate::sass::SassMemorySpace::Global);
        ldg.data_type = Some(SassDataType::U32);
        ldg.dest_operands.push(reg(0));
        ldg.src_operands.push(mem(2, 16));

        let mut stg = EnhancedSassInstruction::new("STG".to_string(), 0x10);
        stg.opcode_class = SassOpcodeClass::GlobalStore;
        stg.memory_space = Some(crate::sass::SassMemorySpace::Global);
        stg.data_type = Some(SassDataType::F32);
        stg.dest_operands.push(mem(4, 0));
        stg.src_operands.push(reg(1));

        let mut ldc = EnhancedSassInstruction::new("LDC".to_string(), 0x20);
        ldc.opcode_class = SassOpcodeClass::ConstantLoad;
        ldc.memory_space = Some(crate::sass::SassMemorySpace::Constant);
        ldc.data_type = Some(SassDataType::U64);
        ldc.dest_operands.push(reg(6));
        ldc.src_operands
            .push(SassOperand::ConstantBank { bank: 0, offset: 0x160 });

        let result = lift_instructions_to_ptx(
            &[ldg, stg, ldc],
            &SassLiftOptions {
                sm_version: 120,
                kernel_name: "mem_kernel".to_string(),
                include_sass_comments: false,
                emit_unsupported_comments: true,
            },
        );

        assert!(result.ptx.contains("ld.global.u32 %r0, [%r2+16];"));
        assert!(result.ptx.contains("st.global.f32 [%r4], %r1;"));
        assert!(result.ptx.contains("ld.const.u64 %r6, [c[0x0][0x160]];"));
    }

    #[test]
    fn sass_lifter_preserves_predicates_and_branch_targets() {
        let mut cmp = EnhancedSassInstruction::new("ISETP".to_string(), 0x0);
        cmp.opcode_class = SassOpcodeClass::IntegerComparison;
        cmp.data_type = Some(SassDataType::S32);
        cmp.modifiers.push("LT".to_string());
        cmp.dest_operands.push(SassOperand::Register(SassRegister::new("P", 0)));
        cmp.src_operands.push(reg(1));
        cmp.src_operands.push(reg(2));

        let mut bra = EnhancedSassInstruction::new("BRA".to_string(), 0x10);
        bra.opcode_class = SassOpcodeClass::Branch;
        bra.predicate = Some(SassOperand::Predicate {
            register: SassRegister::new("P", 0),
            negated: false,
        });
        bra.dest_operands.push(SassOperand::Immediate(0x40));

        let mut add = EnhancedSassInstruction::new("IADD".to_string(), 0x20);
        add.opcode_class = SassOpcodeClass::IntegerArithmetic;
        add.data_type = Some(SassDataType::S32);
        add.dest_operands.push(reg(3));
        add.src_operands.push(reg(3));
        add.src_operands.push(SassOperand::Immediate(1));

        let mut exit = EnhancedSassInstruction::new("EXIT".to_string(), 0x40);
        exit.opcode_class = SassOpcodeClass::Exit;

        let result = lift_instructions_to_ptx(
            &[cmp, bra, add, exit],
            &SassLiftOptions::default(),
        );

        assert!(result.ptx.contains("setp.lt.s32 %p0, %r1, %r2;"));
        assert!(result.ptx.contains("@%p0 bra L_0040;"));
        assert!(result.ptx.contains("add.s32 %r3, %r3, 1;"));
        assert!(result.ptx.contains("L_0040:"));
    }

    #[test]
    fn sass_lifter_reports_unsupported_tensor_instruction() {
        let mut hmma = EnhancedSassInstruction::new("HMMA".to_string(), 0x120);
        hmma.opcode_class = SassOpcodeClass::TensorCore;

        let result = lift_instructions_to_ptx(
            &[hmma],
            &SassLiftOptions {
                sm_version: 120,
                kernel_name: "tensor_kernel".to_string(),
                include_sass_comments: false,
                emit_unsupported_comments: true,
            },
        );

        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0].opcode, "HMMA");
        assert!(result.ptx.contains("unsupported SASS HMMA at 0x0120"));
    }
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
cargo test -p ptx sass_lifter_ --lib
```

Expected: FAIL because memory, comparison, branch, integer arithmetic, and tensor diagnostics are not yet implemented.

- [ ] **Step 3: Extend `lift_instruction` coverage**

Replace the `match inst.opcode.as_str()` body inside `LiftContext::lift_instruction` with:

```rust
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
            "MOV" | "MOV32I" => Some(unary_op(inst, &pred, "mov", data_type_suffix(inst))),
            "IADD" | "IADD3" => Some(binary_op(inst, &pred, "add", data_type_suffix(inst))),
            "IMUL" => Some(binary_op(inst, &pred, "mul.lo", data_type_suffix(inst))),
            "IMAD" => Some(ternary_op(inst, &pred, "mad.lo", data_type_suffix(inst))),
            "SHL" => Some(binary_op(inst, &pred, "shl", data_type_suffix(inst))),
            "SHR" => Some(binary_op(inst, &pred, "shr", data_type_suffix(inst))),
            "LOP" | "LOP3" => Some(binary_op(inst, &pred, "and", "b32")),
            "POPC" => Some(unary_op(inst, &pred, "popc", "b32")),
            "FADD" => Some(binary_op(inst, &pred, "add", "f32")),
            "FMUL" => Some(binary_op(inst, &pred, "mul", "f32")),
            "FFMA" => Some(ternary_op(inst, &pred, "fma.rn", "f32")),
            "FABS" => Some(unary_op(inst, &pred, "abs", "f32")),
            "FNEG" => Some(unary_op(inst, &pred, "neg", "f32")),
            "LDG" | "LDS" | "LDL" | "LDC" => Some(load_op(inst, &pred)),
            "STG" | "STS" | "STL" => Some(store_op(inst, &pred)),
            "ISETP" | "FSETP" | "PSETP" => Some(setp_op(inst, &pred)),
            "BRA" | "BRX" | "JMP" => Some(branch_op(inst, &pred)),
            "BAR" => Some(format!("{}bar.sync 0;", pred)),
            "DEPBAR" => Some(format!("{}// depbar preserved from SASS;", pred)),
            "MEMBAR" => Some(format!("{}membar.gl;", pred)),
            "EXIT" | "RET" => Some(format!("{}ret;", pred)),
            "HMMA" | "IMMA" | "BMMA" | "DMMA" => {
                self.unsupported(inst, "tensor instruction lifting is not implemented")
            }
            "MUFU" => self.unsupported(inst, "MUFU sub-operation lifting is not implemented"),
            _ => self.unsupported(inst, "instruction lifting is not implemented"),
        }
```

- [ ] **Step 4: Add the missing operation helpers**

Add these helpers near `binary_op`:

```rust
fn unary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src = inst
        .src_operands
        .get(0)
        .map(format_operand)
        .unwrap_or_else(|| "0".to_string());
    format!("{}{}.{} {}, {};", pred, op, ty, dst, src)
}

fn ternary_op(inst: &EnhancedSassInstruction, pred: &str, op: &str, ty: &str) -> String {
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let src0 = inst.src_operands.get(0).map(format_operand).unwrap_or_else(|| "0".to_string());
    let src1 = inst.src_operands.get(1).map(format_operand).unwrap_or_else(|| "0".to_string());
    let src2 = inst.src_operands.get(2).map(format_operand).unwrap_or_else(|| "0".to_string());
    format!("{}{}.{} {}, {}, {}, {};", pred, op, ty, dst, src0, src1, src2)
}

fn load_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let space = memory_space_suffix(inst);
    let ty = data_type_suffix(inst);
    let dst = dest_operand(inst).unwrap_or_else(|| "%r0".to_string());
    let addr = inst
        .src_operands
        .first()
        .map(format_address_operand)
        .unwrap_or_else(|| "[0]".to_string());
    format!("{}ld.{}.{} {}, {};", pred, space, ty, dst, addr)
}

fn store_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let space = memory_space_suffix(inst);
    let ty = data_type_suffix(inst);
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
    format!("{}st.{}.{} {}, {};", pred, space, ty, addr, src)
}

fn setp_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let cmp = comparison_suffix(inst);
    let ty = data_type_suffix(inst);
    let dst = dest_operand(inst).unwrap_or_else(|| "%p0".to_string());
    let src0 = inst.src_operands.get(0).map(format_operand).unwrap_or_else(|| "0".to_string());
    let src1 = inst.src_operands.get(1).map(format_operand).unwrap_or_else(|| "0".to_string());
    format!("{}setp.{}.{} {}, {}, {};", pred, cmp, ty, dst, src0, src1)
}

fn branch_op(inst: &EnhancedSassInstruction, pred: &str) -> String {
    let target = branch_target(inst)
        .map(label_for_address)
        .unwrap_or_else(|| "L_unknown".to_string());
    format!("{}bra {};", pred, target)
}

fn data_type_suffix(inst: &EnhancedSassInstruction) -> &'static str {
    match inst.data_type {
        Some(SassDataType::U8) => "u8",
        Some(SassDataType::U16) => "u16",
        Some(SassDataType::U32) => "u32",
        Some(SassDataType::U64) => "u64",
        Some(SassDataType::S8) => "s8",
        Some(SassDataType::S16) => "s16",
        Some(SassDataType::S32) => "s32",
        Some(SassDataType::S64) => "s64",
        Some(SassDataType::F16) => "f16",
        Some(SassDataType::F32) => "f32",
        Some(SassDataType::F64) => "f64",
        Some(SassDataType::B8) => "b8",
        Some(SassDataType::B16) => "b16",
        Some(SassDataType::B32) => "b32",
        Some(SassDataType::B64) => "b64",
        _ => match inst.opcode_class {
            SassOpcodeClass::FloatArithmetic | SassOpcodeClass::FloatComparison => "f32",
            SassOpcodeClass::IntegerArithmetic | SassOpcodeClass::IntegerComparison => "s32",
            _ => "b32",
        },
    }
}

fn memory_space_suffix(inst: &EnhancedSassInstruction) -> &'static str {
    match inst.memory_space {
        Some(SassMemorySpace::Global) => "global",
        Some(SassMemorySpace::Shared) => "shared",
        Some(SassMemorySpace::Local) => "local",
        Some(SassMemorySpace::Constant) => "const",
        _ => match inst.opcode.as_str() {
            "LDG" | "STG" => "global",
            "LDS" | "STS" => "shared",
            "LDL" | "STL" => "local",
            "LDC" => "const",
            _ => "global",
        },
    }
}

fn comparison_suffix(inst: &EnhancedSassInstruction) -> &'static str {
    if inst.modifiers.iter().any(|m| m.eq_ignore_ascii_case("LT")) {
        "lt"
    } else if inst.modifiers.iter().any(|m| m.eq_ignore_ascii_case("LE")) {
        "le"
    } else if inst.modifiers.iter().any(|m| m.eq_ignore_ascii_case("GT")) {
        "gt"
    } else if inst.modifiers.iter().any(|m| m.eq_ignore_ascii_case("GE")) {
        "ge"
    } else if inst.modifiers.iter().any(|m| m.eq_ignore_ascii_case("NE")) {
        "ne"
    } else {
        "eq"
    }
}

fn format_address_operand(operand: &SassOperand) -> String {
    match operand {
        SassOperand::Memory { .. } => format_operand(operand),
        SassOperand::ConstantBank { .. } => format!("[{}]", format_operand(operand)),
        _ => format!("[{}]", format_operand(operand)),
    }
}
```

- [ ] **Step 5: Run the lifter test group and verify GREEN**

Run:

```bash
cargo test -p ptx sass_lifter_ --lib
```

Expected: PASS.

- [ ] **Step 6: Commit expanded semantics**

```bash
git add ptx/src/sass/lifter.rs
git commit -m "feat: lift common SASS instructions to PTX"
```

---

## Task 4: Add Text and CUBIN Frontend Functions Plus Public Exports

**Files:**
- Modify: `ptx/src/sass/lifter.rs`
- Modify: `ptx/src/lib.rs`
- Test: `ptx/src/sass/lifter.rs`

- [ ] **Step 1: Add failing tests for text frontend and malformed CUBIN input**

Append these tests to `ptx/src/sass/lifter.rs`:

```rust
    #[test]
    fn sass_lifter_text_frontend_uses_function_name_and_sm120() {
        let text = r#"
Function : vector_add
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
        .unwrap();

        assert!(result.ptx.contains(".visible .entry vector_add()"));
        assert!(result.ptx.contains(".target sm_120"));
        assert!(result.ptx.contains("ld.global.u32 %r1, [%r2];"));
        assert!(result.ptx.contains("st.global.u32 [%r3], %r1;"));
    }

    #[test]
    fn sass_lifter_text_frontend_rejects_empty_input() {
        let err = lift_sass_text_to_ptx("", SassLiftOptions::default()).unwrap_err();
        assert!(err.contains("No SASS instructions parsed"));
    }

    #[test]
    fn sass_lifter_cubin_frontend_rejects_malformed_input() {
        let err = lift_cubin_to_ptx(b"not an elf", SassLiftOptions::default()).unwrap_err();
        assert!(err.contains("Failed to parse CUBIN"));
    }
```

- [ ] **Step 2: Run frontend tests and verify RED**

Run:

```bash
cargo test -p ptx sass_lifter_text_frontend --lib
cargo test -p ptx sass_lifter_cubin_frontend_rejects_malformed_input --lib
```

Expected: FAIL because `lift_sass_text_to_ptx` and `lift_cubin_to_ptx` are not implemented.

- [ ] **Step 3: Implement frontend functions and option normalization**

Add these functions to `ptx/src/sass/lifter.rs`:

```rust
pub fn lift_sass_text_to_ptx(
    text: &str,
    mut options: SassLiftOptions,
) -> Result<SassLiftResult, String> {
    let instructions = TextDisassemblyParser::parse_cuobjdump_output(text);
    if instructions.is_empty() {
        return Err("No SASS instructions parsed from text input".to_string());
    }
    if options.kernel_name.is_empty() {
        options.kernel_name = instructions
            .iter()
            .find_map(|inst| inst.function_name.clone())
            .unwrap_or_else(|| "kernel".to_string());
    }
    Ok(lift_instructions_to_ptx(&instructions, &options))
}

pub fn lift_cubin_to_ptx(
    cubin_data: &[u8],
    mut options: SassLiftOptions,
) -> Result<SassLiftResult, String> {
    let parsed = CubinParser::new(cubin_data.to_vec())
        .parse()
        .map_err(|e| format!("Failed to parse CUBIN: {}", e))?;
    let kernel = parsed
        .kernels
        .first()
        .ok_or_else(|| "No kernels found in CUBIN".to_string())?;
    if options.kernel_name.is_empty() || options.kernel_name == "kernel" {
        options.kernel_name = kernel.name.clone();
    }
    options.sm_version = if options.sm_version == 0 {
        kernel.sm_version
    } else {
        options.sm_version
    };

    let disasm = SassDisassembler::new(kernel.sm_version)
        .map_err(|e| format!("Failed to create SASS disassembler: {}", e))?;
    let mut instructions = disasm.disassemble(&kernel.code, kernel.address);
    for inst in &mut instructions {
        inst.function_name = Some(kernel.name.clone());
        if let Some(line) = parsed.debug_lines.get(&inst.address) {
            inst.ptx_file = Some(line.file.clone());
            inst.ptx_line = Some(line.line);
            inst.ptx_column = Some(line.column);
        }
    }

    Ok(lift_instructions_to_ptx(&instructions, &options))
}
```

- [ ] **Step 4: Re-export the lifter APIs from `ptx/src/lib.rs`**

Add these items to the existing `pub use sass::{ ... }` block in `ptx/src/lib.rs`:

```rust
    lift_cubin_to_ptx,
    lift_instructions_to_ptx,
    lift_sass_text_to_ptx,
    SassLiftDiagnostic,
    SassLiftOptions,
    SassLiftResult,
```

- [ ] **Step 5: Run frontend tests and verify GREEN**

Run:

```bash
cargo test -p ptx sass_lifter_text_frontend --lib
cargo test -p ptx sass_lifter_cubin_frontend_rejects_malformed_input --lib
```

Expected: PASS.

- [ ] **Step 6: Commit frontends and exports**

```bash
git add ptx/src/sass/lifter.rs ptx/src/lib.rs
git commit -m "feat: add SASS lifter frontends"
```

---

## Task 5: Route `sass_inliner --recover-ptx` Through Shared Lifter

**Files:**
- Modify: `ptx/src/bin/sass_inliner.rs`
- Modify: `ptx/tests/sass_inliner_test.rs`
- Test: `ptx/tests/sass_inliner_test.rs`

- [ ] **Step 1: Add failing integration test for recover-PTX stdin SM120**

Append this test to `ptx/tests/sass_inliner_test.rs`:

```rust
#[test]
fn test_sass_inliner_recover_ptx_sm120_stdin() {
    let sm120_kernel = r#"
Function : sm120_text_kernel
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   LDG.E.U32 R1, [R2] ;
        /*0020*/                   FADD R3, R4, R5 ;
        /*0030*/              @P0  BRA 0x50 ;
        /*0040*/                   STG.E.U32 [R6], R3 ;
        /*0050*/                   EXIT ;
"#;

    let mut child = Command::new(get_sass_inliner_path())
        .arg("--stdin")
        .arg("--recover-ptx")
        .arg("--sm")
        .arg("120")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn sass_inliner");

    {
        let stdin = child.stdin.as_mut().expect("Failed to get stdin");
        stdin
            .write_all(sm120_kernel.as_bytes())
            .expect("Failed to write");
    }

    let output = child.wait_with_output().expect("Failed to wait");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "sass_inliner failed: {}", stderr);
    assert!(stdout.contains(".target sm_120"), "stdout was: {}", stdout);
    assert!(
        stdout.contains(".visible .entry sm120_text_kernel()"),
        "stdout was: {}",
        stdout
    );
    assert!(stdout.contains("mov.u32 %r0, %tid.x;"), "stdout was: {}", stdout);
    assert!(
        stdout.contains("ld.global.u32 %r1, [%r2];"),
        "stdout was: {}",
        stdout
    );
    assert!(stdout.contains("add.f32 %r3, %r4, %r5;"), "stdout was: {}", stdout);
    assert!(stdout.contains("@%p0 bra L_0050;"), "stdout was: {}", stdout);
    assert!(
        stdout.contains("st.global.u32 [%r6], %r3;"),
        "stdout was: {}",
        stdout
    );
}
```

- [ ] **Step 2: Build the binary and verify RED**

Run:

```bash
cargo build -p ptx --bin sass_inliner
cargo test -p ptx --test sass_inliner_test test_sass_inliner_recover_ptx_sm120_stdin -- --nocapture
```

Expected: FAIL because current `--recover-ptx` emits `.version 7.0`, `.target sm_120` only by manual fallback, and uses `PtxReconstructor` output rather than the shared lifter. The assertion on labels or normalized store/load output should fail.

- [ ] **Step 3: Import lifter APIs in `sass_inliner.rs`**

Change the `use ptx::sass::{ ... }` import to include:

```rust
    lift_instructions_to_ptx, SassLiftOptions,
```

- [ ] **Step 4: Replace the `if args.recover_ptx` body with shared lifter emission**

Inside `run()` in `ptx/src/bin/sass_inliner.rs`, replace the current `if args.recover_ptx { ... return Ok(()); }` block with:

```rust
    if args.recover_ptx {
        if args.verbose {
            eprintln!("Recovering PTX from SASS with shared lifter...");
        }

        let result = lift_instructions_to_ptx(
            &instructions,
            &SassLiftOptions {
                sm_version: args.target_sm,
                kernel_name: kernel_name.clone(),
                include_sass_comments: true,
                emit_unsupported_comments: true,
            },
        );

        if args.verbose {
            for diagnostic in &result.diagnostics {
                eprintln!(
                    "lifter diagnostic at {} {}: {}",
                    diagnostic
                        .address
                        .map(|addr| format!("0x{:x}", addr))
                        .unwrap_or_else(|| "unknown".to_string()),
                    diagnostic.opcode,
                    diagnostic.message
                );
            }
        }

        if let Some(ref ptx_out) = args.ptx_output {
            fs::write(ptx_out, &result.ptx)
                .map_err(|e| format!("Failed to write PTX output: {}", e))?;
            if args.verbose {
                eprintln!("Recovered PTX written to {:?}", ptx_out);
            }
        } else {
            print!("{}", result.ptx);
        }

        return Ok(());
    }
```

- [ ] **Step 5: Run the integration test and verify GREEN**

Run:

```bash
cargo build -p ptx --bin sass_inliner
cargo test -p ptx --test sass_inliner_test test_sass_inliner_recover_ptx_sm120_stdin -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit `sass_inliner` integration**

```bash
git add ptx/src/bin/sass_inliner.rs ptx/tests/sass_inliner_test.rs
git commit -m "feat: route sass_inliner PTX recovery through lifter"
```

---

## Task 6: Route `gpu_rr ptx` Through Shared Lifter

**Files:**
- Modify: `ptx/src/bin/gpu_rr.rs`
- Test: `ptx/src/bin/gpu_rr.rs` via binary build

- [ ] **Step 1: Add lifter imports to `gpu_rr.rs`**

At the top import section, add these names to the existing `ptx::sass` import list:

```rust
    lift_cubin_to_ptx, lift_sass_text_to_ptx, SassLiftOptions,
```

- [ ] **Step 2: Replace the PTX reconstruction branch in `run_ptx`**

Replace the body of `run_ptx` from the `// Load CUBIN or text SASS` comment through the final `Ok(())` with:

```rust
    let content = fs::read(input).map_err(|e| format!("Failed to read input: {}", e))?;

    let options = SassLiftOptions {
        sm_version: 120,
        kernel_name: String::new(),
        include_sass_comments: true,
        emit_unsupported_comments: true,
    };

    let result = if content.starts_with(&[0x7f, b'E', b'L', b'F']) {
        lift_cubin_to_ptx(&content, options)?
    } else {
        let text = String::from_utf8_lossy(&content);
        lift_sass_text_to_ptx(&text, options)?
    };

    if let Some(addr) = address {
        let label = format!("L_{:04x}:", addr);
        println!("=== PTX for SASS address 0x{:x} ===", addr);
        let mut print_next = false;
        for line in result.ptx.lines() {
            if line.trim() == label {
                print_next = true;
                println!("{}", line);
                continue;
            }
            if print_next {
                println!("{}", line);
                if line.trim_end().ends_with(';') {
                    break;
                }
            }
        }
    } else if let Some(out_path) = output {
        fs::write(out_path, &result.ptx)
            .map_err(|e| format!("Failed to write output: {}", e))?;
        println!("Recovered PTX written to {:?}", out_path);
    } else {
        print!("{}", result.ptx);
    }

    if verbose {
        for diagnostic in &result.diagnostics {
            eprintln!(
                "lifter diagnostic at {} {}: {}",
                diagnostic
                    .address
                    .map(|addr| format!("0x{:x}", addr))
                    .unwrap_or_else(|| "unknown".to_string()),
                diagnostic.opcode,
                diagnostic.message
            );
        }
    }

    Ok(())
```

- [ ] **Step 3: Build `gpu_rr` and verify GREEN**

Run:

```bash
cargo build -p ptx --bin gpu_rr
```

Expected: PASS.

- [ ] **Step 4: Commit `gpu_rr` integration**

```bash
git add ptx/src/bin/gpu_rr.rs
git commit -m "feat: route gpu_rr PTX recovery through lifter"
```

---

## Task 7: Final Verification and Cleanup

**Files:**
- Modify only files already touched if formatting or warnings require it.
- Test: workspace commands below.

- [ ] **Step 1: Format the touched Rust files**

Run:

```bash
cargo fmt -- ptx/src/sass/lifter.rs ptx/src/sass/mod.rs ptx/src/lib.rs ptx/src/bin/sass_inliner.rs ptx/src/bin/gpu_rr.rs ptx/tests/sass_inliner_test.rs
```

Expected: PASS with no output.

- [ ] **Step 2: Run lifter library tests**

Run:

```bash
cargo test -p ptx sass_lifter_ --lib
```

Expected: PASS.

- [ ] **Step 3: Run the SM120 CLI integration test**

Run:

```bash
cargo build -p ptx --bin sass_inliner
cargo test -p ptx --test sass_inliner_test test_sass_inliner_recover_ptx_sm120_stdin -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Build the second CLI**

Run:

```bash
cargo build -p ptx --bin gpu_rr
```

Expected: PASS.

- [ ] **Step 5: Run a manual stdin smoke test**

Run:

```bash
printf 'Function : smoke\n        /*0000*/                   S2R R0, SR_TID.X ;\n        /*0010*/                   EXIT ;\n' | target/debug/sass_inliner --stdin --recover-ptx --sm 120 -
```

Expected output includes:

```text
.target sm_120
.visible .entry smoke()
mov.u32 %r0, %tid.x;
ret;
```

- [ ] **Step 6: Check git status**

Run:

```bash
git status --short
```

Expected: only intentional files changed, with the pre-existing untracked `zluda/src/impl/test_*` files still unrelated.

- [ ] **Step 7: Commit final formatting if needed**

If `cargo fmt` changed files after the previous commits, run:

```bash
git add ptx/src/sass/lifter.rs ptx/src/sass/mod.rs ptx/src/lib.rs ptx/src/bin/sass_inliner.rs ptx/src/bin/gpu_rr.rs ptx/tests/sass_inliner_test.rs
git commit -m "style: format SASS lifter integration"
```

Expected: commit created only when formatting produced changes.

---

## Self-Review

Spec coverage:
- Shared lifter core: Task 1 and Task 2.
- Complete PTX module emission: Task 2.
- Common scalar, memory, control, synchronization semantics: Task 3.
- Text frontend: Task 4.
- CUBIN frontend: Task 4.
- Public exports: Task 4.
- `sass_inliner --recover-ptx`: Task 5.
- `gpu_rr ptx`: Task 6.
- Verification without CUDA tools: Task 7.

Incomplete-content scan:
- No unresolved implementation markers are intentionally left in the plan.

Type consistency:
- `SassLiftOptions`, `SassLiftDiagnostic`, and `SassLiftResult` names match the spec and all task snippets.
- Function names match the public API in the spec: `lift_instructions_to_ptx`, `lift_sass_text_to_ptx`, and `lift_cubin_to_ptx`.
