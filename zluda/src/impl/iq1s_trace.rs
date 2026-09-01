use ptx::pass::tmatmul_algorithm_tree::{
    AbstractOperation, AlgorithmTree, OperationInfo, TMatmulConfig,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub(crate) const QWEN_MODEL_CONTEXT_LIMIT: u32 = 262_144;
const TILE_DIM: usize = 1024;
const HANDWRITTEN_ASSEMBLY: &str = "ldv v0, PARAM_INPUT\ntmatmul_import v0\ntmatmul_go PARAM_MATRIX\ntmatmul_export v0\nsv v0, PARAM_OUTPUT\nstall\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TraceKind {
    Handwritten,
    AlgorithmTreeCompiler,
}

impl TraceKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Handwritten => "handwritten",
            Self::AlgorithmTreeCompiler => "compiler",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TmatmulTrace {
    pub(crate) kind: TraceKind,
    pub(crate) model_context_limit: u32,
    pub(crate) algorithm_tree_instruction_count: usize,
    pub(crate) assembly: String,
    pub(crate) instructions: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AssembledTrace {
    pub(crate) program: Vec<u8>,
    pub(crate) sha256: [u8; 32],
    pub(crate) num_vector_registers: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedTraceProgram {
    pub(crate) selected_kind: TraceKind,
    pub(crate) model_context_limit: u32,
    pub(crate) program: Vec<u8>,
    pub(crate) selected_sha256: [u8; 32],
    pub(crate) handwritten_sha256: [u8; 32],
    pub(crate) compiler_sha256: [u8; 32],
    pub(crate) semantic_sha256: [u8; 32],
    pub(crate) assembly_sha256: [u8; 32],
    pub(crate) assembly: String,
    pub(crate) instructions: Vec<Vec<String>>,
}

fn validate_model_context_limit(limit: u32) -> Result<(), String> {
    if limit == 0 {
        return Err("Qwen model context limit must be positive".to_string());
    }
    if limit > QWEN_MODEL_CONTEXT_LIMIT {
        return Err(format!(
            "Qwen model context limit {limit} exceeds supported limit {QWEN_MODEL_CONTEXT_LIMIT}"
        ));
    }
    Ok(())
}

fn executable_tokens(assembly: &str) -> Vec<Vec<String>> {
    assembly
        .lines()
        .filter_map(|raw| {
            let line = raw.split(';').next().unwrap_or("").trim();
            if line.is_empty() {
                return None;
            }
            Some(
                line.split(|byte: char| byte.is_whitespace() || byte == ',')
                    .filter(|token| !token.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn canonical_instructions() -> Vec<Vec<String>> {
    executable_tokens(HANDWRITTEN_ASSEMBLY)
}

pub(crate) fn semantic_sha256(instructions: &[Vec<String>]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"hetgpu-tmatmul-semantic-trace-v1\0");
    for instruction in instructions {
        for token in instruction {
            hash.update(token.as_bytes());
            hash.update([0]);
        }
        hash.update([b'\n']);
    }
    hash.finalize().into()
}

pub(crate) fn build_handwritten_trace(model_context_limit: u32) -> Result<TmatmulTrace, String> {
    validate_model_context_limit(model_context_limit)?;
    Ok(TmatmulTrace {
        kind: TraceKind::Handwritten,
        model_context_limit,
        algorithm_tree_instruction_count: 0,
        assembly: HANDWRITTEN_ASSEMBLY.to_string(),
        instructions: canonical_instructions(),
    })
}

pub(crate) fn build_compiler_trace(model_context_limit: u32) -> Result<TmatmulTrace, String> {
    validate_model_context_limit(model_context_limit)?;
    let mut tree = AlgorithmTree::new(TMatmulConfig {
        d: TILE_DIM,
        num_cores: 1,
        num_vector_registers: 8,
        fixed_point_precision: 16,
        fixed_point_exponent: -8,
        vector_parallelism: 16,
        tmatmul_parallelism: 64,
        lut_parallelism: 8,
        memory_bandwidth: 32,
        clock_period: 1e-9,
    });
    let input = tree.new_abstract_vector(TILE_DIM);
    let output = tree.new_abstract_vector(TILE_DIM);

    let mut input_info = HashMap::new();
    input_info.insert(
        "address_label".to_string(),
        OperationInfo::String("PARAM_INPUT".to_string()),
    );
    tree.new_abstract_operation(
        AbstractOperation::Ldv,
        vec![],
        vec![input],
        Some(input_info),
    );

    let mut matrix_info = HashMap::new();
    matrix_info.insert(
        "address_label".to_string(),
        OperationInfo::String("PARAM_MATRIX".to_string()),
    );
    matrix_info.insert("NumRows".to_string(), OperationInfo::Int(TILE_DIM as i64));
    matrix_info.insert(
        "NumColumns".to_string(),
        OperationInfo::Int(TILE_DIM as i64),
    );
    tree.new_abstract_operation(
        AbstractOperation::TMatmul,
        vec![input],
        vec![output],
        Some(matrix_info),
    );

    let mut output_info = HashMap::new();
    output_info.insert(
        "address_label".to_string(),
        OperationInfo::String("PARAM_OUTPUT".to_string()),
    );
    tree.new_abstract_operation(
        AbstractOperation::Sv,
        vec![output],
        vec![],
        Some(output_info),
    );

    let algorithm_tree_instruction_count = tree.instruction_operations.len();
    let mut assembly = tree.generate_assembly();
    assembly.push_str("    stall\n");
    let raw_instructions = executable_tokens(&assembly);
    let expected_tree_instructions = vec![
        vec!["ldv", "v0", "PARAM_INPUT_slice_0"],
        vec!["tmatmul_import", "v0"],
        vec!["tmatmul_go", "PARAM_MATRIX_block_0_0"],
        vec!["tmatmul_export", "v0"],
        vec!["sv", "v0", "PARAM_OUTPUT_slice_0"],
        vec!["stall"],
    ];
    if raw_instructions != expected_tree_instructions {
        return Err(format!(
            "AlgorithmTree emitted an unexpected AU250 trace: {raw_instructions:?}"
        ));
    }
    assembly = assembly
        .replace("PARAM_INPUT_slice_0", "PARAM_INPUT")
        .replace("PARAM_MATRIX_block_0_0", "PARAM_MATRIX")
        .replace("PARAM_OUTPUT_slice_0", "PARAM_OUTPUT");
    let instructions = executable_tokens(&assembly);
    if instructions != canonical_instructions() {
        return Err(format!(
            "AlgorithmTree emitted a noncanonical AU250 trace: {instructions:?}"
        ));
    }
    Ok(TmatmulTrace {
        kind: TraceKind::AlgorithmTreeCompiler,
        model_context_limit,
        algorithm_tree_instruction_count,
        assembly,
        instructions,
    })
}

pub(crate) fn assemble_trace(
    trace: &TmatmulTrace,
    labels: &HashMap<String, u64>,
    num_vector_registers: u8,
) -> Result<AssembledTrace, String> {
    validate_model_context_limit(trace.model_context_limit)?;
    if trace.instructions != canonical_instructions()
        || executable_tokens(&trace.assembly) != trace.instructions
    {
        return Err("tmatmul trace is not the canonical checked instruction sequence".to_string());
    }
    let program = super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
        &trace.assembly,
        labels,
        num_vector_registers,
    )
    .map_err(|error| error.to_string())?;
    let sha256 = Sha256::digest(&program).into();
    Ok(AssembledTrace {
        program,
        sha256,
        num_vector_registers,
    })
}

pub(crate) fn build_selected_trace(
    mode: &str,
    model_context_limit: u32,
    labels: &HashMap<String, u64>,
    num_vector_registers: u8,
) -> Result<SelectedTraceProgram, String> {
    let handwritten = build_handwritten_trace(model_context_limit)?;
    let compiler = build_compiler_trace(model_context_limit)?;
    let handwritten_binary = assemble_trace(&handwritten, labels, num_vector_registers)?;
    let compiler_binary = assemble_trace(&compiler, labels, num_vector_registers)?;
    if handwritten.instructions != compiler.instructions
        || handwritten_binary.program != compiler_binary.program
        || handwritten_binary.sha256 != compiler_binary.sha256
    {
        return Err(
            "handwritten and AlgorithmTree compiler traces are not byte-identical".to_string(),
        );
    }
    let selected_kind = match mode {
        "handwritten" => TraceKind::Handwritten,
        "compiler" => TraceKind::AlgorithmTreeCompiler,
        _ => {
            return Err(
                "HETGPU_IQ1S_TRACE_MODE must be exactly handwritten or compiler".to_string(),
            )
        }
    };
    let selected_trace = match selected_kind {
        TraceKind::Handwritten => &handwritten,
        TraceKind::AlgorithmTreeCompiler => &compiler,
    };
    Ok(SelectedTraceProgram {
        selected_kind,
        model_context_limit,
        program: handwritten_binary.program,
        selected_sha256: handwritten_binary.sha256,
        handwritten_sha256: handwritten_binary.sha256,
        compiler_sha256: compiler_binary.sha256,
        semantic_sha256: semantic_sha256(&selected_trace.instructions),
        assembly_sha256: Sha256::digest(selected_trace.assembly.as_bytes()).into(),
        assembly: selected_trace.assembly.clone(),
        instructions: selected_trace.instructions.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn labels() -> HashMap<String, u64> {
        HashMap::from([
            ("PARAM_MATRIX".to_string(), 0x1000_0000),
            ("PARAM_INPUT".to_string(), 0x2000_0000),
            ("PARAM_OUTPUT".to_string(), 0x3000_0000),
        ])
    }

    #[test]
    fn handwritten_and_algorithm_tree_compiler_emit_identical_real_instruction_trace() {
        let handwritten = build_handwritten_trace(262_144).unwrap();
        let compiler = build_compiler_trace(262_144).unwrap();
        let expected = vec![
            vec!["ldv", "v0", "PARAM_INPUT"],
            vec!["tmatmul_import", "v0"],
            vec!["tmatmul_go", "PARAM_MATRIX"],
            vec!["tmatmul_export", "v0"],
            vec!["sv", "v0", "PARAM_OUTPUT"],
            vec!["stall"],
        ];

        assert_eq!(handwritten.instructions, expected);
        assert_eq!(compiler.instructions, expected);
        assert_eq!(handwritten.kind, TraceKind::Handwritten);
        assert_eq!(compiler.kind, TraceKind::AlgorithmTreeCompiler);
        assert!(compiler.algorithm_tree_instruction_count >= 5);

        let handwritten_binary = assemble_trace(&handwritten, &labels(), 4).unwrap();
        let compiler_binary = assemble_trace(&compiler, &labels(), 4).unwrap();
        assert_eq!(handwritten_binary.program, compiler_binary.program);
        assert_eq!(handwritten_binary.sha256, compiler_binary.sha256);
        assert_ne!(handwritten_binary.sha256, [0; 32]);
    }

    #[test]
    fn compiler_accepts_exact_model_context_limit_and_fails_closed_above_it() {
        assert_eq!(
            build_compiler_trace(262_144).unwrap().model_context_limit,
            262_144
        );
        assert!(build_compiler_trace(0).unwrap_err().contains("positive"));
        assert!(build_compiler_trace(262_145)
            .unwrap_err()
            .contains("262144"));
    }

    #[test]
    fn assembler_rejects_mutated_or_unbound_trace() {
        let mut trace = build_handwritten_trace(512).unwrap();
        trace.instructions.swap(1, 2);
        assert!(assemble_trace(&trace, &labels(), 4)
            .unwrap_err()
            .contains("canonical"));

        let mut missing = labels();
        missing.remove("PARAM_MATRIX");
        assert!(assemble_trace(&build_handwritten_trace(512).unwrap(), &missing, 4).is_err());
    }

    #[test]
    fn selected_runtime_trace_always_cross_checks_both_builders() {
        let handwritten = build_selected_trace("handwritten", 262_144, &labels(), 4).unwrap();
        let compiler = build_selected_trace("compiler", 262_144, &labels(), 4).unwrap();
        assert_eq!(handwritten.selected_kind, TraceKind::Handwritten);
        assert_eq!(compiler.selected_kind, TraceKind::AlgorithmTreeCompiler);
        assert_eq!(handwritten.program, compiler.program);
        assert_eq!(handwritten.handwritten_sha256, handwritten.compiler_sha256);
        assert_eq!(compiler.handwritten_sha256, compiler.compiler_sha256);
        assert!(build_selected_trace("auto", 262_144, &labels(), 4)
            .unwrap_err()
            .contains("handwritten or compiler"));
    }
}
