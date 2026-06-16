use std::fmt;
use std::path::PathBuf;

pub(crate) const CXL_TYPE2_TMATMUL_UAPI_VERSION: u32 = 2;
pub(crate) const TMATMUL_DPA_MATRIX: u64 = 0x0000_0000;
pub(crate) const TMATMUL_DPA_INPUT: u64 = 0x0010_0000;
pub(crate) const TMATMUL_DPA_OUTPUT: u64 = 0x0020_0000;
pub(crate) const TMATMUL_DPA_PROGRAM: u64 = 0x0030_0000;
pub(crate) const TMATMUL_PROGRAM_BYTES: usize = 96;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CxlTmatmulError {
    MissingPtx,
    InvalidPtx,
    CompileFailed(String),
    Io(String),
    AllocationTooSmall {
        name: &'static str,
        have: usize,
        need: usize,
    },
    SizeOverflow,
}

impl fmt::Display for CxlTmatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPtx => write!(f, "missing PTX source"),
            Self::InvalidPtx => write!(f, "invalid PTX source"),
            Self::CompileFailed(msg) => write!(f, "PTX to tmatmul compile failed: {msg}"),
            Self::Io(msg) => write!(f, "tmatmul I/O failed: {msg}"),
            Self::AllocationTooSmall { name, have, need } => {
                write!(f, "{name} allocation too small: have {have}, need {need}")
            }
            Self::SizeOverflow => write!(f, "tmatmul size calculation overflowed"),
        }
    }
}

impl std::error::Error for CxlTmatmulError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompiledTmatmul {
    pub(crate) source_len: usize,
    pub(crate) assembly: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JitArtifacts {
    pub(crate) ptx_path: PathBuf,
    pub(crate) asm_path: PathBuf,
}

pub(crate) fn cxl_tmatmul_enabled() -> bool {
    env_flag("HETGPU_CXL_TMATMUL") || env_flag("HETGPU_TMATMUL_CXL")
}

pub(crate) fn ptx_looks_valid(ptx: &str) -> bool {
    let trimmed = ptx.trim_start();
    if trimmed.len() < 50 {
        return false;
    }
    if !(trimmed.starts_with(".version") || trimmed.starts_with("//")) {
        return false;
    }
    if !(trimmed.contains(".target ") && trimmed.contains(".address_size")) {
        return false;
    }
    if !trimmed.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with(".visible .entry ") || line.starts_with(".entry ")
    }) {
        return false;
    }
    !trimmed.bytes().any(|b| {
        b == 0 || b == 0x7f || (b < 0x20 && !matches!(b, b'\n' | b'\r' | b'\t'))
    })
}

pub(crate) fn compile_ptx_to_tmatmul_assembly(
    ptx_source: Option<&str>,
) -> Result<CompiledTmatmul, CxlTmatmulError> {
    let ptx_source = ptx_source.ok_or(CxlTmatmulError::MissingPtx)?;
    if !ptx_looks_valid(ptx_source) {
        return Err(CxlTmatmulError::InvalidPtx);
    }

    let source_len = ptx_source.len();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ptx::pass::ptx_to_tmatmul_assembly(ptx_source)
    }))
    .map_err(|_| CxlTmatmulError::CompileFailed("compiler panicked".to_string()))?;

    let assembly = result.map_err(|e| CxlTmatmulError::CompileFailed(e.to_string()))?;
    Ok(CompiledTmatmul {
        source_len,
        assembly,
    })
}

pub(crate) fn stage_jit_artifacts(
    kernel_name: &str,
    ptx_source: &str,
    compiled: &CompiledTmatmul,
) -> Result<JitArtifacts, CxlTmatmulError> {
    let base_dir = std::env::var("HETGPU_TMATMUL_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    std::fs::create_dir_all(&base_dir).map_err(|e| CxlTmatmulError::Io(e.to_string()))?;

    let stem = sanitize_kernel_name(kernel_name);
    let ptx_path = base_dir.join(format!("hetgpu_ptx_{stem}.ptx"));
    let asm_path = base_dir.join(format!("hetgpu_asm_{stem}.S"));

    std::fs::write(&ptx_path, ptx_source.as_bytes())
        .map_err(|e| CxlTmatmulError::Io(e.to_string()))?;
    std::fs::write(&asm_path, compiled.assembly.as_bytes())
        .map_err(|e| CxlTmatmulError::Io(e.to_string()))?;

    Ok(JitArtifacts { ptx_path, asm_path })
}

pub(crate) fn matrix_bytes(dim: usize) -> Result<usize, CxlTmatmulError> {
    dim.checked_mul(dim)
        .and_then(|bytes| bytes.checked_div(4))
        .ok_or(CxlTmatmulError::SizeOverflow)
}

pub(crate) fn vector_bytes(dim: usize) -> Result<usize, CxlTmatmulError> {
    dim.checked_mul(2).ok_or(CxlTmatmulError::SizeOverflow)
}

pub(crate) fn validate_allocations(
    dim: usize,
    input_bytes: usize,
    output_bytes: usize,
    matrix_bytes_have: usize,
) -> Result<(), CxlTmatmulError> {
    let vector_need = vector_bytes(dim)?;
    let matrix_need = matrix_bytes(dim)?;

    require_allocation("input", input_bytes, vector_need)?;
    require_allocation("output", output_bytes, vector_need)?;
    require_allocation("matrix", matrix_bytes_have, matrix_need)?;
    Ok(())
}

pub(crate) fn required_dax_len() -> usize {
    TMATMUL_DPA_PROGRAM as usize + TMATMUL_PROGRAM_BYTES
}

pub(crate) fn encode_smoke_program() -> Vec<u8> {
    let mut program = Vec::with_capacity(TMATMUL_PROGRAM_BYTES);
    program.extend_from_slice(&encode_instr(0b001, 0, 0, 0, 0, 0b01, 0, 0, TMATMUL_DPA_INPUT));
    program.extend_from_slice(&encode_instr(0b011, 0, 0, 0, 0, 0, 0b01, 0, 0));
    program.extend_from_slice(&encode_instr(0b011, 0, 0, 0, 0, 0, 0b10, 0, TMATMUL_DPA_MATRIX));
    program.extend_from_slice(&encode_instr(0b011, 0, 1, 1, 1, 0, 0b11, 0, 0));
    program.extend_from_slice(&encode_instr(0b001, 0, 1, 1, 1, 0b10, 0, 0, TMATMUL_DPA_OUTPUT));
    program.extend_from_slice(&encode_instr(0b101, 0, 0, 0, 0, 0, 0, 0, 0));
    debug_assert_eq!(program.len(), TMATMUL_PROGRAM_BYTES);
    program
}

fn encode_instr(
    fu: u8,
    op: u8,
    vy: u8,
    vb: u8,
    va: u8,
    ls: u8,
    tm: u8,
    rms: u8,
    addr: u64,
) -> [u8; 16] {
    let mut word = addr as u128;
    word |= ((rms & 0x7) as u128) << 64;
    word |= ((tm & 0x3) as u128) << 67;
    word |= ((ls & 0x3) as u128) << 69;
    word |= ((va & 0x7) as u128) << 71;
    word |= ((vb & 0x7) as u128) << 74;
    word |= ((vy & 0x7) as u128) << 77;
    word |= ((op & 0xf) as u128) << 80;
    word |= ((fu & 0x7) as u128) << 84;
    word.to_le_bytes()
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
        .unwrap_or(false)
}

fn require_allocation(
    name: &'static str,
    have: usize,
    need: usize,
) -> Result<(), CxlTmatmulError> {
    if have < need {
        Err(CxlTmatmulError::AllocationTooSmall { name, have, need })
    } else {
        Ok(())
    }
}

fn sanitize_kernel_name(kernel_name: &str) -> String {
    let mut out: String = kernel_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if out.is_empty() {
        out.push_str("kernel");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry simple_tmatmul_jit(
    .param .u64 input,
    .param .u64 output
) {
    .reg .f32 %f<4>;
    .reg .u64 %rd<3>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.f32 %f0, [%rd1];
    mul.f32 %f1, %f0, %f0;
    st.global.f32 [%rd2], %f1;

    ret;
}
"#;

    #[test]
    fn ptx_jit_requires_runtime_ptx_not_kernel_name() {
        let err = compile_ptx_to_tmatmul_assembly(None).unwrap_err();

        assert_eq!(err, CxlTmatmulError::MissingPtx);
    }

    #[test]
    fn ptx_jit_rejects_invalid_payload_without_name_fallback() {
        let err = compile_ptx_to_tmatmul_assembly(Some("_Zfake_matmul_kernel")).unwrap_err();

        assert_eq!(err, CxlTmatmulError::InvalidPtx);
    }

    #[test]
    fn ptx_jit_lowers_valid_runtime_ptx() {
        let compiled = compile_ptx_to_tmatmul_assembly(Some(SIMPLE_PTX)).unwrap();

        assert_eq!(compiled.source_len, SIMPLE_PTX.len());
        assert!(!compiled.assembly.trim().is_empty());
    }

    #[test]
    fn encode_smoke_program_matches_v2_runner() {
        let prog = encode_smoke_program();

        assert_eq!(prog.len(), 96);
        assert_eq!(&prog[0..8], &TMATMUL_DPA_INPUT.to_le_bytes());
        assert_eq!(&prog[32..40], &TMATMUL_DPA_MATRIX.to_le_bytes());
        assert_eq!(&prog[64..72], &TMATMUL_DPA_OUTPUT.to_le_bytes());
    }

    #[test]
    fn layout_requires_program_end() {
        assert_eq!(
            required_dax_len(),
            TMATMUL_DPA_PROGRAM as usize + TMATMUL_PROGRAM_BYTES
        );
    }

    #[test]
    fn validates_supported_sizes() {
        let dim = 512;

        assert_eq!(matrix_bytes(dim).unwrap(), 512 * 512 / 4);
        assert_eq!(vector_bytes(dim).unwrap(), 512 * 2);
        assert!(validate_allocations(dim, 1024, 1024, 65536).is_ok());
        assert!(validate_allocations(dim, 1023, 1024, 65536).is_err());
    }
}
