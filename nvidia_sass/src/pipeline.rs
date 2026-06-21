use crate::cubin_builder;
use crate::isel;
use crate::regalloc;
use crate::roundtrip;
use crate::scheduler;
use crate::types::*;

#[derive(Debug)]
pub struct PtxToSassPipelineResult {
    pub module: SassModule,
    pub cubin: Vec<u8>,
    pub pass_names: Vec<&'static str>,
    pub notes: Vec<String>,
}

pub trait PtxToSassPass {
    fn name(&self) -> &'static str;
    fn run(&self, ctx: &mut PtxToSassContext) -> Result<(), String>;
}

pub struct PtxToSassPipeline {
    passes: Vec<Box<dyn PtxToSassPass>>,
}

impl PtxToSassPipeline {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(mut self, pass: impl PtxToSassPass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    pub fn run(&self, ctx: &mut PtxToSassContext) -> Result<(), String> {
        for pass in &self.passes {
            ctx.pass_names.push(pass.name());
            pass.run(ctx)
                .map_err(|err| format!("{}: {}", pass.name(), err))?;
        }
        Ok(())
    }
}

impl Default for PtxToSassPipeline {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PtxToSassContext {
    ptx: String,
    requested_kernel_name: String,
    kernel_name: String,
    sm_version: u32,
    virtual_instructions: Vec<SassInst>,
    physical_instructions: Vec<SassInst>,
    scheduled_instructions: Vec<SassInst>,
    num_registers: u32,
    module: Option<SassModule>,
    cubin: Option<Vec<u8>>,
    pass_names: Vec<&'static str>,
    notes: Vec<String>,
}

impl PtxToSassContext {
    pub fn new(ptx: impl Into<String>, kernel_name: impl Into<String>, sm_version: u32) -> Self {
        let kernel_name = kernel_name.into();
        Self {
            ptx: ptx.into(),
            requested_kernel_name: kernel_name.clone(),
            kernel_name,
            sm_version,
            virtual_instructions: Vec::new(),
            physical_instructions: Vec::new(),
            scheduled_instructions: Vec::new(),
            num_registers: 0,
            module: None,
            cubin: None,
            pass_names: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn ptx(&self) -> &str {
        &self.ptx
    }

    pub fn requested_kernel_name(&self) -> &str {
        &self.requested_kernel_name
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn set_kernel_name(&mut self, kernel_name: impl Into<String>) {
        self.kernel_name = kernel_name.into();
    }

    pub fn sm_version(&self) -> u32 {
        self.sm_version
    }

    pub fn set_sm_version(&mut self, sm_version: u32) {
        self.sm_version = sm_version;
    }

    pub fn virtual_instructions(&self) -> &[SassInst] {
        &self.virtual_instructions
    }

    pub fn virtual_instructions_mut(&mut self) -> &mut Vec<SassInst> {
        &mut self.virtual_instructions
    }

    pub fn set_virtual_instructions(&mut self, instructions: Vec<SassInst>) {
        self.virtual_instructions = instructions;
    }

    pub fn physical_instructions(&self) -> &[SassInst] {
        &self.physical_instructions
    }

    pub fn physical_instructions_mut(&mut self) -> &mut Vec<SassInst> {
        &mut self.physical_instructions
    }

    pub fn set_physical_instructions(&mut self, instructions: Vec<SassInst>) {
        self.physical_instructions = instructions;
    }

    pub fn scheduled_instructions(&self) -> &[SassInst] {
        &self.scheduled_instructions
    }

    pub fn scheduled_instructions_mut(&mut self) -> &mut Vec<SassInst> {
        &mut self.scheduled_instructions
    }

    pub fn set_scheduled_instructions(&mut self, instructions: Vec<SassInst>) {
        self.scheduled_instructions = instructions;
    }

    pub fn num_registers(&self) -> u32 {
        self.num_registers
    }

    pub fn set_num_registers(&mut self, num_registers: u32) {
        self.num_registers = num_registers;
    }

    pub fn module(&self) -> Option<&SassModule> {
        self.module.as_ref()
    }

    pub fn set_module(&mut self, module: SassModule) {
        self.module = Some(module);
    }

    pub fn cubin(&self) -> Option<&[u8]> {
        self.cubin.as_deref()
    }

    pub fn set_cubin(&mut self, cubin: Vec<u8>) {
        self.cubin = Some(cubin);
    }

    pub fn pass_names(&self) -> &[&'static str] {
        &self.pass_names
    }

    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    pub fn notes_mut(&mut self) -> &mut Vec<String> {
        &mut self.notes
    }
}

pub fn default_ptx_to_sass_pipeline() -> PtxToSassPipeline {
    PtxToSassPipeline::new()
        .add_pass(ParsePtxSubsetPass)
        .add_pass(AllocateRegistersPass)
        .add_pass(ScheduleControlCodesPass)
        .add_pass(ValidateEncodingPass)
        .add_pass(BuildCubinPass)
}

pub fn compile_ptx_to_cubin(
    ptx: &str,
    kernel_name: &str,
    sm_version: u32,
) -> Result<PtxToSassPipelineResult, String> {
    let mut ctx = PtxToSassContext::new(ptx, kernel_name, sm_version);
    default_ptx_to_sass_pipeline().run(&mut ctx)?;
    Ok(PtxToSassPipelineResult {
        module: ctx
            .module
            .ok_or_else(|| "PTX to SASS pipeline did not produce a module".to_string())?,
        cubin: ctx
            .cubin
            .ok_or_else(|| "PTX to SASS pipeline did not produce a CUBIN".to_string())?,
        pass_names: ctx.pass_names,
        notes: ctx.notes,
    })
}

struct ParsePtxSubsetPass;

impl PtxToSassPass for ParsePtxSubsetPass {
    fn name(&self) -> &'static str {
        "parse-ptx-subset"
    }

    fn run(&self, ctx: &mut PtxToSassContext) -> Result<(), String> {
        let parsed = parse_ptx_subset(&ctx.ptx, &ctx.requested_kernel_name, ctx.sm_version)?;
        ctx.kernel_name = parsed.kernel_name;
        ctx.sm_version = parsed.sm_version;
        ctx.virtual_instructions = parsed.instructions;
        Ok(())
    }
}

struct AllocateRegistersPass;

impl PtxToSassPass for AllocateRegistersPass {
    fn name(&self) -> &'static str {
        "allocate-registers"
    }

    fn run(&self, ctx: &mut PtxToSassContext) -> Result<(), String> {
        let (physical, num_registers) =
            regalloc::allocate(&ctx.virtual_instructions).map_err(|err| err.to_string())?;
        ctx.physical_instructions = physical;
        ctx.num_registers = num_registers;
        Ok(())
    }
}

struct ScheduleControlCodesPass;

impl PtxToSassPass for ScheduleControlCodesPass {
    fn name(&self) -> &'static str {
        "schedule-control-codes"
    }

    fn run(&self, ctx: &mut PtxToSassContext) -> Result<(), String> {
        ctx.scheduled_instructions = scheduler::schedule(&ctx.physical_instructions);
        Ok(())
    }
}

struct ValidateEncodingPass;

impl PtxToSassPass for ValidateEncodingPass {
    fn name(&self) -> &'static str {
        "validate-encoding"
    }

    fn run(&self, ctx: &mut PtxToSassContext) -> Result<(), String> {
        for inst in &ctx.scheduled_instructions {
            roundtrip::validate_roundtrip(inst, ctx.sm_version).map_err(|err| err.to_string())?;
        }
        Ok(())
    }
}

struct BuildCubinPass;

impl PtxToSassPass for BuildCubinPass {
    fn name(&self) -> &'static str {
        "build-cubin"
    }

    fn run(&self, ctx: &mut PtxToSassContext) -> Result<(), String> {
        let module = SassModule {
            kernels: vec![SassKernel {
                name: ctx.kernel_name.clone(),
                instructions: ctx.scheduled_instructions.clone(),
                num_registers: ctx.num_registers,
                shared_mem_bytes: 0,
                const_mem_bytes: 0,
                local_mem_bytes: 0,
                max_threads: 1024,
                params: Vec::new(),
            }],
            sm_version: ctx.sm_version,
            global_constants: Vec::new(),
        };
        let cubin =
            cubin_builder::build_cubin_from_module(&module).map_err(|err| err.to_string())?;
        ctx.module = Some(module);
        ctx.cubin = Some(cubin);
        Ok(())
    }
}

struct ParsedPtxSubset {
    kernel_name: String,
    sm_version: u32,
    instructions: Vec<SassInst>,
}

fn parse_ptx_subset(
    ptx: &str,
    requested_kernel_name: &str,
    requested_sm_version: u32,
) -> Result<ParsedPtxSubset, String> {
    let mut sm_version = requested_sm_version;
    let mut current_kernel: Option<String> = None;
    let mut selected_kernel: Option<String> = None;
    let mut in_selected_body = false;
    let mut instructions = Vec::new();

    for raw_line in ptx.lines() {
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(target) = line.strip_prefix(".target ") {
            if sm_version == 0 {
                sm_version = target
                    .trim()
                    .strip_prefix("sm_")
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(120);
            }
            continue;
        }

        if let Some(kernel_name) = parse_entry_name(line) {
            let selected = requested_kernel_name.is_empty() || requested_kernel_name == kernel_name;
            current_kernel = Some(kernel_name.to_string());
            in_selected_body = selected && line.contains('{');
            if selected {
                selected_kernel = Some(kernel_name.to_string());
            }
            continue;
        }

        if line == "{" {
            if let Some(name) = &current_kernel {
                in_selected_body =
                    requested_kernel_name.is_empty() || requested_kernel_name == name;
                if in_selected_body {
                    selected_kernel = Some(name.clone());
                }
            }
            continue;
        }

        if line == "}" {
            in_selected_body = false;
            current_kernel = None;
            continue;
        }

        if !in_selected_body || line.starts_with('.') || line.ends_with(':') {
            continue;
        }

        instructions.push(parse_ptx_instruction(line)?);
    }

    let kernel_name = selected_kernel.ok_or_else(|| {
        if requested_kernel_name.is_empty() {
            "no PTX entry function found".to_string()
        } else {
            format!("no PTX entry function named '{requested_kernel_name}' found")
        }
    })?;
    if instructions.is_empty() {
        return Err(format!(
            "no supported instructions found in PTX entry '{kernel_name}'"
        ));
    }
    Ok(ParsedPtxSubset {
        kernel_name,
        sm_version: if sm_version == 0 { 120 } else { sm_version },
        instructions,
    })
}

fn parse_ptx_instruction(line: &str) -> Result<SassInst, String> {
    let line = line.trim_end_matches(';').trim();
    let (mnemonic, operands) = line
        .split_once(char::is_whitespace)
        .map(|(mnemonic, operands)| (mnemonic.trim(), split_operands(operands)))
        .unwrap_or((line, Vec::new()));

    match mnemonic {
        "ret" | "exit" => Ok(isel::select_exit()),
        "mov.u32" => {
            let dst = parse_reg(operand(&operands, 0, line)?)?;
            let src = operand(&operands, 1, line)?;
            if let Some(sreg) = parse_special_reg(src) {
                Ok(isel::select_special_reg(dst, sreg))
            } else {
                Ok(isel::select_mov(dst, parse_reg(src)?))
            }
        }
        "add.s32" | "add.u32" => Ok(isel::select_add_i32(
            parse_reg(operand(&operands, 0, line)?)?,
            parse_reg(operand(&operands, 1, line)?)?,
            parse_reg(operand(&operands, 2, line)?)?,
        )),
        "add.f32" => Ok(isel::select_add_f32(
            parse_reg(operand(&operands, 0, line)?)?,
            parse_reg(operand(&operands, 1, line)?)?,
            parse_reg(operand(&operands, 2, line)?)?,
        )),
        "mul.f32" => Ok(isel::select_mul_f32(
            parse_reg(operand(&operands, 0, line)?)?,
            parse_reg(operand(&operands, 1, line)?)?,
            parse_reg(operand(&operands, 2, line)?)?,
        )),
        "fma.rn.f32" | "fma.f32" => Ok(isel::select_fma_f32(
            parse_reg(operand(&operands, 0, line)?)?,
            parse_reg(operand(&operands, 1, line)?)?,
            parse_reg(operand(&operands, 2, line)?)?,
            parse_reg(operand(&operands, 3, line)?)?,
        )),
        "ld.global.u32" => {
            let (base, offset) = parse_memory(operand(&operands, 1, line)?)?;
            Ok(isel::select_load_global(
                parse_reg(operand(&operands, 0, line)?)?,
                base,
                offset,
            ))
        }
        "st.global.u32" => {
            let (base, offset) = parse_memory(operand(&operands, 0, line)?)?;
            Ok(isel::select_store_global(
                base,
                offset,
                parse_reg(operand(&operands, 1, line)?)?,
            ))
        }
        "bar.sync" => Ok(isel::select_bar_sync(
            operand(&operands, 0, line)?.parse::<u32>().unwrap_or(0),
        )),
        _ => Err(format!(
            "unsupported instruction '{mnemonic}' in line '{line}'"
        )),
    }
}

fn strip_comment(line: &str) -> &str {
    line.split_once("//").map(|(line, _)| line).unwrap_or(line)
}

fn parse_entry_name(line: &str) -> Option<&str> {
    let marker = ".entry ";
    let start = line.find(marker)? + marker.len();
    let rest = line[start..].trim_start();
    let end = rest
        .find(|ch: char| ch == '(' || ch.is_whitespace())
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

fn split_operands(operands: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut bracket_depth = 0usize;

    for ch in operands.chars() {
        match ch {
            '[' => {
                bracket_depth += 1;
                current.push(ch);
            }
            ']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                current.push(ch);
            }
            ',' if bracket_depth == 0 => {
                result.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    if !current.trim().is_empty() {
        result.push(current.trim().to_string());
    }
    result
}

fn operand<'a>(operands: &'a [String], index: usize, line: &str) -> Result<&'a str, String> {
    operands
        .get(index)
        .map(|operand| operand.as_str())
        .ok_or_else(|| format!("missing operand {index} in line '{line}'"))
}

fn parse_reg(operand: &str) -> Result<u8, String> {
    operand
        .trim()
        .strip_prefix("%r")
        .ok_or_else(|| format!("expected PTX register, got '{operand}'"))?
        .parse::<u8>()
        .map_err(|err| format!("invalid PTX register '{operand}': {err}"))
}

fn parse_special_reg(operand: &str) -> Option<SpecialReg> {
    match operand.trim() {
        "%tid.x" => Some(SpecialReg::TidX),
        "%tid.y" => Some(SpecialReg::TidY),
        "%tid.z" => Some(SpecialReg::TidZ),
        "%ctaid.x" => Some(SpecialReg::CtaidX),
        "%ctaid.y" => Some(SpecialReg::CtaidY),
        "%ctaid.z" => Some(SpecialReg::CtaidZ),
        "%ntid.x" => Some(SpecialReg::NtidX),
        "%ntid.y" => Some(SpecialReg::NtidY),
        "%ntid.z" => Some(SpecialReg::NtidZ),
        "%laneid" => Some(SpecialReg::LaneId),
        "%warpid" => Some(SpecialReg::WarpId),
        _ => None,
    }
}

fn parse_memory(operand: &str) -> Result<(u8, i32), String> {
    let inner = operand
        .trim()
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| format!("expected memory operand, got '{operand}'"))?;

    if let Some((base, offset)) = inner.split_once('+') {
        return Ok((parse_reg(base.trim())?, parse_i32(offset.trim())?));
    }
    if let Some((base, offset)) = inner.split_once('-') {
        return Ok((parse_reg(base.trim())?, -parse_i32(offset.trim())?));
    }
    Ok((parse_reg(inner.trim())?, 0))
}

fn parse_i32(value: &str) -> Result<i32, String> {
    if let Some(hex) = value.strip_prefix("0x") {
        i32::from_str_radix(hex, 16).map_err(|err| format!("invalid integer '{value}': {err}"))
    } else {
        value
            .parse::<i32>()
            .map_err(|err| format!("invalid integer '{value}': {err}"))
    }
}
