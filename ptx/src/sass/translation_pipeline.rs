use super::{lift_instructions_to_ptx, EnhancedSassInstruction, SassLiftOptions, SassLiftResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassTranslationDirection {
    SassToPtx,
    PtxToSass,
}

pub trait SassTranslationPass {
    fn name(&self) -> &'static str;
    fn direction(&self) -> SassTranslationDirection;
    fn run(&self, state: &mut SassTranslationState) -> Result<(), String>;
}

pub struct SassTranslationPipeline {
    passes: Vec<Box<dyn SassTranslationPass>>,
}

impl SassTranslationPipeline {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(mut self, pass: impl SassTranslationPass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    pub fn run(&self, state: &mut SassTranslationState) -> Result<(), String> {
        for pass in &self.passes {
            if pass.direction() != SassTranslationDirection::SassToPtx {
                return Err(format!(
                    "pass '{}' has direction {:?}, expected SassToPtx",
                    pass.name(),
                    pass.direction()
                ));
            }
            state.pass_names.push(pass.name());
            pass.run(state)
                .map_err(|err| format!("{}: {}", pass.name(), err))?;
        }
        Ok(())
    }
}

impl Default for SassTranslationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SassTranslationState {
    pub instructions: Vec<EnhancedSassInstruction>,
    pub options: SassLiftOptions,
    pub result: Option<SassLiftResult>,
    pub pass_names: Vec<&'static str>,
    pub notes: Vec<String>,
}

impl SassTranslationState {
    pub fn new(instructions: Vec<EnhancedSassInstruction>, options: SassLiftOptions) -> Self {
        Self {
            instructions,
            options,
            result: None,
            pass_names: Vec::new(),
            notes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SassTranslationResult {
    pub ptx: String,
    pub diagnostics: Vec<super::SassLiftDiagnostic>,
    pub pass_names: Vec<&'static str>,
    pub notes: Vec<String>,
}

pub fn default_sass_to_ptx_pipeline() -> SassTranslationPipeline {
    SassTranslationPipeline::new()
        .add_pass(LiftToPtxPass)
        .add_pass(ValidateLiftedPtxPass)
}

pub fn run_default_sass_to_ptx_pipeline(
    instructions: Vec<EnhancedSassInstruction>,
    options: SassLiftOptions,
) -> Result<SassTranslationResult, String> {
    let mut state = SassTranslationState::new(instructions, options);
    default_sass_to_ptx_pipeline().run(&mut state)?;
    let result = state
        .result
        .ok_or_else(|| "SASS to PTX pipeline did not produce PTX".to_string())?;
    Ok(SassTranslationResult {
        ptx: result.ptx,
        diagnostics: result.diagnostics,
        pass_names: state.pass_names,
        notes: state.notes,
    })
}

pub struct LiftToPtxPass;

impl SassTranslationPass for LiftToPtxPass {
    fn name(&self) -> &'static str {
        "sass-lift-to-ptx"
    }

    fn direction(&self) -> SassTranslationDirection {
        SassTranslationDirection::SassToPtx
    }

    fn run(&self, state: &mut SassTranslationState) -> Result<(), String> {
        state.result = Some(lift_instructions_to_ptx(
            &state.instructions,
            &state.options,
        ));
        Ok(())
    }
}

pub struct ValidateLiftedPtxPass;

impl SassTranslationPass for ValidateLiftedPtxPass {
    fn name(&self) -> &'static str {
        "validate-lifted-ptx"
    }

    fn direction(&self) -> SassTranslationDirection {
        SassTranslationDirection::SassToPtx
    }

    fn run(&self, state: &mut SassTranslationState) -> Result<(), String> {
        let ptx = state
            .result
            .as_ref()
            .ok_or_else(|| "no PTX result from earlier pass".to_string())?;
        ptx_parser::parse_module_checked(&ptx.ptx)
            .map(|_| ())
            .map_err(|errors| format!("lifted PTX did not parse: {:?}", errors))
    }
}
