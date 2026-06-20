use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Read;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::ptr;

pub(crate) const CXL_TYPE2_TMATMUL_UAPI_VERSION: u32 = 2;
pub(crate) const TMATMUL_DPA_MATRIX: u64 = 0x0000_0000;
pub(crate) const TMATMUL_DPA_INPUT: u64 = 0x0010_0000;
pub(crate) const TMATMUL_DPA_OUTPUT: u64 = 0x0020_0000;
pub(crate) const TMATMUL_DPA_PROGRAM: u64 = 0x0030_0000;
pub(crate) const TMATMUL_PROGRAM_BYTES: usize = 96;
const DEFAULT_DEVICE_PATH: &str = "/dev/cxl_tmatmul3b000";
const DEFAULT_DAX_PATH: &str = "/dev/dax0.0";
const DEFAULT_TIMEOUT_MS: u32 = 10_000;
const CXL_TYPE2_TMATMUL_RESULT_STALLED: u32 = 1 << 0;
const CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR: u32 = 1 << 2;

#[cfg(unix)]
static MATRIX_STAGE_CACHE: std::sync::Mutex<Option<(usize, usize)>> = std::sync::Mutex::new(None);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CxlTmatmulError {
    MissingPtx,
    InvalidPtx,
    CompileFailed(String),
    AssembleFailed(String),
    Io(String),
    Device(String),
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
            Self::AssembleFailed(msg) => write!(f, "tmatmul assembly failed: {msg}"),
            Self::Io(msg) => write!(f, "tmatmul I/O failed: {msg}"),
            Self::Device(msg) => write!(f, "tmatmul device failed: {msg}"),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CxlTmatmulRunStatus {
    pub(crate) timeout_ms: u32,
    pub(crate) dma_status: u32,
    pub(crate) stall_status: u32,
    pub(crate) instr_count: u32,
    pub(crate) dim_d: u32,
    pub(crate) result_flags: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct CxlType2TmatmulInfo {
    version: u32,
    dev_id: u32,
    num_instances: u32,
    dim_d: u32,
    ddr_data_width: u32,
    mc_status: u32,
    reserved0: u32,
    reserved1: [u64; 2],
    reserved2: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct CxlType2TmatmulCsrRun {
    timeout_ms: u32,
    flags: u32,
    dma_status: u32,
    stall_status: u32,
    instr_count: u32,
    dim_d: u32,
    result_flags: u32,
    reserved0: u32,
    reserved1: [u64; 4],
}

impl From<&CxlType2TmatmulCsrRun> for CxlTmatmulRunStatus {
    fn from(run: &CxlType2TmatmulCsrRun) -> Self {
        Self {
            timeout_ms: run.timeout_ms,
            dma_status: run.dma_status,
            stall_status: run.stall_status,
            instr_count: run.instr_count,
            dim_d: run.dim_d,
            result_flags: run.result_flags,
        }
    }
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
    !trimmed
        .bytes()
        .any(|b| b == 0 || b == 0x7f || (b < 0x20 && !matches!(b, b'\n' | b'\r' | b'\t')))
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
    required_dax_len_for_program(TMATMUL_PROGRAM_BYTES).unwrap_or(usize::MAX)
}

pub(crate) fn encode_smoke_program() -> Vec<u8> {
    let mut program = Vec::with_capacity(TMATMUL_PROGRAM_BYTES);
    program.extend_from_slice(&encode_instr(
        0b001,
        0,
        0,
        0,
        0,
        0b01,
        0,
        0,
        TMATMUL_DPA_INPUT,
    ));
    program.extend_from_slice(&encode_instr(0b011, 0, 0, 0, 0, 0, 0b01, 0, 0));
    program.extend_from_slice(&encode_instr(
        0b011,
        0,
        0,
        0,
        0,
        0,
        0b10,
        0,
        TMATMUL_DPA_MATRIX,
    ));
    program.extend_from_slice(&encode_instr(0b011, 0, 1, 1, 1, 0, 0b11, 0, 0));
    program.extend_from_slice(&encode_instr(
        0b001,
        0,
        1,
        1,
        1,
        0b10,
        0,
        0,
        TMATMUL_DPA_OUTPUT,
    ));
    program.extend_from_slice(&encode_instr(0b101, 0, 0, 0, 0, 0, 0, 0, 0));
    debug_assert_eq!(program.len(), TMATMUL_PROGRAM_BYTES);
    program
}

pub(crate) fn assemble_tmatmul_program(
    assembly: &str,
    labels: &HashMap<String, u64>,
) -> Result<Vec<u8>, CxlTmatmulError> {
    let mut program = Vec::new();

    for (line_idx, raw_line) in assembly.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.split(';').next().unwrap_or("").trim();
        if line.is_empty()
            || line.starts_with('.')
            || line == "{"
            || line == "}"
            || line.ends_with(':')
        {
            continue;
        }

        let tokens: Vec<&str> = line
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|token| !token.is_empty())
            .collect();
        if tokens.is_empty() {
            continue;
        }

        let instr = assemble_tmatmul_instruction(line_no, &tokens, labels)?;
        program.extend_from_slice(&instr);
    }

    if program.is_empty() {
        return Err(CxlTmatmulError::AssembleFailed(
            "no AFU instructions found".to_string(),
        ));
    }

    Ok(program)
}

#[cfg(unix)]
pub(crate) unsafe fn submit_hardware_matmul_from_ptrs(
    assembly: &str,
    labels: &HashMap<String, u64>,
    matrix_ptr: *const u8,
    matrix_alloc: usize,
    input_ptr: *const u8,
    input_alloc: usize,
    output_ptr: *mut u8,
    output_alloc: usize,
    timeout_ms: u32,
) -> Result<CxlTmatmulRunStatus, CxlTmatmulError> {
    if matrix_ptr.is_null() || input_ptr.is_null() || output_ptr.is_null() {
        return Err(CxlTmatmulError::Device(
            "null matrix/input/output pointer".to_string(),
        ));
    }

    let program = assemble_tmatmul_program(assembly, labels)?;
    if program.len() != TMATMUL_PROGRAM_BYTES {
        return Err(CxlTmatmulError::Device(format!(
            "RUN_CSR_ONLY fetches exactly {TMATMUL_PROGRAM_BYTES} bytes, encoded program is {} bytes",
            program.len()
        )));
    }

    let device_path = cxl_tmatmul_device_path();
    let device = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&device_path)
        .map_err(|e| CxlTmatmulError::Io(format!("open {device_path}: {e}")))?;
    let info = get_info(&device)?;
    if info.version != CXL_TYPE2_TMATMUL_UAPI_VERSION {
        return Err(CxlTmatmulError::Device(format!(
            "unsupported UAPI version {}, expected {}",
            info.version, CXL_TYPE2_TMATMUL_UAPI_VERSION
        )));
    }
    if info.dim_d == 0 {
        return Err(CxlTmatmulError::Device(
            "device reports dim_d=0".to_string(),
        ));
    }

    let dim = usize::try_from(info.dim_d).map_err(|_| CxlTmatmulError::SizeOverflow)?;
    validate_allocations(dim, input_alloc, output_alloc, matrix_alloc)?;
    let matrix_len = matrix_bytes(dim)?;
    let vector_len = vector_bytes(dim)?;
    validate_fixed_layout(matrix_len, vector_len, program.len())?;

    let matrix = std::slice::from_raw_parts(matrix_ptr, matrix_len);
    let input = std::slice::from_raw_parts(input_ptr, vector_len);
    let output = std::slice::from_raw_parts_mut(output_ptr, vector_len);

    submit_prepared_hardware_matmul(
        &device, &program, matrix, input, output, timeout_ms, info.dim_d,
    )
}

#[cfg(not(unix))]
pub(crate) unsafe fn submit_hardware_matmul_from_ptrs(
    _assembly: &str,
    _labels: &HashMap<String, u64>,
    _matrix_ptr: *const u8,
    _matrix_alloc: usize,
    _input_ptr: *const u8,
    _input_alloc: usize,
    _output_ptr: *mut u8,
    _output_alloc: usize,
    _timeout_ms: u32,
) -> Result<CxlTmatmulRunStatus, CxlTmatmulError> {
    Err(CxlTmatmulError::Device(
        "CXL tmatmul ioctl submit is only implemented on Unix".to_string(),
    ))
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

fn assemble_tmatmul_instruction(
    line_no: usize,
    tokens: &[&str],
    labels: &HashMap<String, u64>,
) -> Result<[u8; 16], CxlTmatmulError> {
    let op = tokens[0].to_ascii_lowercase();
    match op.as_str() {
        "ldv" | "sv" => {
            require_arg_count(line_no, tokens, 3)?;
            let reg = parse_vector_register(line_no, tokens[1])?;
            let addr = resolve_address(line_no, tokens[2], labels)?;
            let ls = if op == "ldv" { 0b01 } else { 0b10 };
            Ok(encode_instr(0b001, 0, reg, reg, reg, ls, 0, 0, addr))
        }
        "add" | "sub" | "mul" | "div" => {
            require_arg_count(line_no, tokens, 4)?;
            let va = parse_vector_register(line_no, tokens[1])?;
            let vy = parse_vector_register(line_no, tokens[2])?;
            let vb = parse_vector_register(line_no, tokens[3])?;
            let op_code = match op.as_str() {
                "add" => 0b0001,
                "sub" => 0b0010,
                "mul" => 0b0011,
                "div" => 0b0100,
                _ => unreachable!(),
            };
            Ok(encode_instr(0b010, op_code, vy, vb, va, 0, 0, 0, 0))
        }
        "sig" | "csig" | "silu" => {
            require_arg_count(line_no, tokens, 3)?;
            let va = parse_vector_register(line_no, tokens[1])?;
            let vy_vb = parse_vector_register(line_no, tokens[2])?;
            let op_code = match op.as_str() {
                "sig" => 0b0101,
                "csig" => 0b0110,
                "silu" => 0b0111,
                _ => unreachable!(),
            };
            Ok(encode_instr(0b010, op_code, vy_vb, vy_vb, va, 0, 0, 0, 0))
        }
        "tmatmul_go" => {
            require_arg_count(line_no, tokens, 2)?;
            let addr = resolve_address(line_no, tokens[1], labels)?;
            Ok(encode_instr(0b011, 0, 0, 0, 0, 0, 0b10, 0, addr))
        }
        "tmatmul_import" | "tmatmul_export" => {
            require_arg_count(line_no, tokens, 2)?;
            let reg = parse_vector_register(line_no, tokens[1])?;
            let tm = if op == "tmatmul_import" { 0b01 } else { 0b11 };
            Ok(encode_instr(0b011, 0, reg, reg, reg, 0, tm, 0, 0))
        }
        "rms_clear" => {
            require_arg_count(line_no, tokens, 1)?;
            Ok(encode_instr(0b100, 0, 0, 0, 0, 0, 0, 0b001, 0))
        }
        "rms_accumulate" => {
            require_arg_count(line_no, tokens, 2)?;
            let reg = parse_vector_register(line_no, tokens[1])?;
            Ok(encode_instr(0b100, 0, reg, reg, reg, 0, 0, 0b010, 0))
        }
        "rms_finish_accumulate" => {
            require_arg_count(line_no, tokens, 2)?;
            let addr = resolve_address(line_no, tokens[1], labels)?;
            Ok(encode_instr(0b100, 0, 0, 0, 0, 0, 0, 0b011, addr))
        }
        "rms_norm" => {
            require_arg_count(line_no, tokens, 3)?;
            let va = parse_vector_register(line_no, tokens[1])?;
            let vy_vb = parse_vector_register(line_no, tokens[2])?;
            Ok(encode_instr(0b100, 0, vy_vb, vy_vb, va, 0, 0, 0b100, 0))
        }
        "stall" => {
            require_arg_count(line_no, tokens, 1)?;
            Ok(encode_instr(0b101, 0, 0, 0, 0, 0, 0, 0, 0))
        }
        _ => Err(CxlTmatmulError::AssembleFailed(format!(
            "line {line_no}: unsupported instruction '{}'",
            tokens[0]
        ))),
    }
}

fn require_arg_count(
    line_no: usize,
    tokens: &[&str],
    expected: usize,
) -> Result<(), CxlTmatmulError> {
    if tokens.len() == expected {
        Ok(())
    } else {
        Err(CxlTmatmulError::AssembleFailed(format!(
            "line {line_no}: '{}' expects {} tokens, got {}",
            tokens[0],
            expected,
            tokens.len()
        )))
    }
}

fn parse_vector_register(line_no: usize, token: &str) -> Result<u8, CxlTmatmulError> {
    let token = token.trim().to_ascii_lowercase();
    let Some(num) = token.strip_prefix('v') else {
        return Err(CxlTmatmulError::AssembleFailed(format!(
            "line {line_no}: expected vector register, got '{token}'"
        )));
    };
    let reg = num.parse::<u8>().map_err(|_| {
        CxlTmatmulError::AssembleFailed(format!(
            "line {line_no}: invalid vector register '{token}'"
        ))
    })?;
    if reg < 8 {
        Ok(reg)
    } else {
        Err(CxlTmatmulError::AssembleFailed(format!(
            "line {line_no}: vector register '{token}' exceeds AFU v0..v7 encoding"
        )))
    }
}

fn resolve_address(
    line_no: usize,
    token: &str,
    labels: &HashMap<String, u64>,
) -> Result<u64, CxlTmatmulError> {
    let token = token.trim().trim_start_matches('[').trim_end_matches(']');
    if let Some(value) = parse_u64_text(token) {
        return Ok(value);
    }
    if let Some(value) = labels.get(token) {
        return Ok(*value);
    }
    Err(CxlTmatmulError::AssembleFailed(format!(
        "line {line_no}: unresolved address label '{token}'"
    )))
}

fn parse_u64_text(text: &str) -> Option<u64> {
    let cleaned: String = text.chars().filter(|&c| c != '_').collect();
    let s = cleaned.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else if let Some(bin) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        u64::from_str_radix(bin, 2).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

fn required_dax_len_for_program(program_bytes: usize) -> Result<usize, CxlTmatmulError> {
    (TMATMUL_DPA_PROGRAM as usize)
        .checked_add(program_bytes)
        .ok_or(CxlTmatmulError::SizeOverflow)
}

fn validate_fixed_layout(
    matrix_len: usize,
    vector_len: usize,
    program_len: usize,
) -> Result<(), CxlTmatmulError> {
    let matrix_end = range_end(TMATMUL_DPA_MATRIX, matrix_len)?;
    let input_end = range_end(TMATMUL_DPA_INPUT, vector_len)?;
    let output_end = range_end(TMATMUL_DPA_OUTPUT, vector_len)?;
    let program_end = range_end(TMATMUL_DPA_PROGRAM, program_len)?;

    if matrix_end > TMATMUL_DPA_INPUT as usize {
        return Err(CxlTmatmulError::Device(format!(
            "fixed DAX layout overlaps matrix/input: matrix_end=0x{matrix_end:x} input=0x{TMATMUL_DPA_INPUT:x}"
        )));
    }
    if input_end > TMATMUL_DPA_OUTPUT as usize {
        return Err(CxlTmatmulError::Device(format!(
            "fixed DAX layout overlaps input/output: input_end=0x{input_end:x} output=0x{TMATMUL_DPA_OUTPUT:x}"
        )));
    }
    if output_end > TMATMUL_DPA_PROGRAM as usize {
        return Err(CxlTmatmulError::Device(format!(
            "fixed DAX layout overlaps output/program: output_end=0x{output_end:x} program=0x{TMATMUL_DPA_PROGRAM:x}"
        )));
    }
    if program_end < TMATMUL_DPA_PROGRAM as usize {
        return Err(CxlTmatmulError::SizeOverflow);
    }
    Ok(())
}

fn range_end(offset: u64, len: usize) -> Result<usize, CxlTmatmulError> {
    let offset = usize::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
    offset.checked_add(len).ok_or(CxlTmatmulError::SizeOverflow)
}

#[cfg(unix)]
fn get_info(device: &File) -> Result<CxlType2TmatmulInfo, CxlTmatmulError> {
    let mut info = CxlType2TmatmulInfo::default();
    let rc = unsafe {
        libc::ioctl(
            device.as_raw_fd(),
            cxl_type2_tmatmul_get_info_ioctl(),
            &mut info,
        )
    };
    if rc != 0 {
        return Err(CxlTmatmulError::Io(format!(
            "CXL_TYPE2_TMATMUL_GET_INFO: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(info)
}

#[cfg(unix)]
fn submit_prepared_hardware_matmul(
    device: &File,
    program: &[u8],
    matrix: &[u8],
    input: &[u8],
    output: &mut [u8],
    timeout_ms: u32,
    dim_d: u32,
) -> Result<CxlTmatmulRunStatus, CxlTmatmulError> {
    let used_len = [
        range_end(TMATMUL_DPA_MATRIX, matrix.len())?,
        range_end(TMATMUL_DPA_INPUT, input.len())?,
        range_end(TMATMUL_DPA_OUTPUT, output.len())?,
        range_end(TMATMUL_DPA_PROGRAM, program.len())?,
    ]
    .into_iter()
    .max()
    .ok_or(CxlTmatmulError::SizeOverflow)?;

    let staging = StagingMap::open(used_len)?;
    unsafe {
        let matrix_key = (matrix.as_ptr() as usize, matrix.len());
        let assume_static_matrix = env_flag("HETGPU_CXL_TMATMUL_ASSUME_STATIC_MATRIX");
        let matrix_already_staged = assume_static_matrix
            && MATRIX_STAGE_CACHE
                .lock()
                .map(|cache| *cache == Some(matrix_key))
                .unwrap_or(false);
        if !matrix_already_staged {
            stage_bytes(staging.ptr, TMATMUL_DPA_MATRIX, matrix)?;
            if assume_static_matrix {
                if let Ok(mut cache) = MATRIX_STAGE_CACHE.lock() {
                    *cache = Some(matrix_key);
                }
            }
        }
        stage_bytes(staging.ptr, TMATMUL_DPA_INPUT, input)?;
        ptr::write_bytes(
            staging.ptr.add(TMATMUL_DPA_OUTPUT as usize),
            0xa5,
            output.len(),
        );
        flush_range(staging.ptr.add(TMATMUL_DPA_OUTPUT as usize), output.len());
        stage_bytes(staging.ptr, TMATMUL_DPA_PROGRAM, program)?;
    }

    let mut run = CxlType2TmatmulCsrRun {
        timeout_ms: timeout_ms_or_default(timeout_ms),
        ..Default::default()
    };
    let mut rc = -1;
    let mut saved_error = std::io::Error::from_raw_os_error(0);
    for request in cxl_type2_tmatmul_run_csr_only_ioctl_requests() {
        let mut attempt = run;
        rc = unsafe { libc::ioctl(device.as_raw_fd(), request, &mut attempt) };
        saved_error = std::io::Error::last_os_error();
        run = attempt;
        if rc == 0 || saved_error.raw_os_error() != Some(libc::ENOTTY) {
            break;
        }
    }
    let status = CxlTmatmulRunStatus::from(&run);
    if rc != 0 {
        return Err(CxlTmatmulError::Device(format!(
            "CXL_TYPE2_TMATMUL_RUN_CSR_ONLY: {saved_error}; status={status:?}"
        )));
    }
    if (run.result_flags & CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR) != 0 {
        return Err(CxlTmatmulError::Device(format!(
            "DMA_ERROR after RUN_CSR_ONLY; status={status:?}"
        )));
    }
    if (run.result_flags & CXL_TYPE2_TMATMUL_RESULT_STALLED) == 0 {
        return Err(CxlTmatmulError::Device(format!(
            "device did not reach stall; status={status:?}"
        )));
    }
    if run.dim_d != dim_d {
        return Err(CxlTmatmulError::Device(format!(
            "RUN_CSR_ONLY dim changed from {dim_d} to {}; status={status:?}",
            run.dim_d
        )));
    }

    unsafe {
        invalidate_range(staging.ptr.add(TMATMUL_DPA_OUTPUT as usize), output.len());
        ptr::copy_nonoverlapping(
            staging.ptr.add(TMATMUL_DPA_OUTPUT as usize),
            output.as_mut_ptr(),
            output.len(),
        );
    }

    Ok(status)
}

#[cfg(unix)]
struct StagingMap {
    ptr: *mut u8,
    map_ptr: *mut u8,
    map_len: usize,
}

#[cfg(unix)]
impl StagingMap {
    fn open(used_len: usize) -> Result<Self, CxlTmatmulError> {
        if let Some(hpa_base) = cxl_tmatmul_hpa_base()? {
            let hpa_size = cxl_tmatmul_hpa_size()?;
            if let Some(hpa_size) = hpa_size {
                if used_len > hpa_size {
                    return Err(CxlTmatmulError::AllocationTooSmall {
                        name: "hpa staging window",
                        have: hpa_size,
                        need: used_len,
                    });
                }
            }
            return Self::open_physical(&cxl_tmatmul_mem_path(), hpa_base, used_len);
        }

        let dax_path = cxl_tmatmul_dax_path();
        let dax_size = read_dax_size(&dax_path)?;
        if used_len > dax_size {
            return Err(CxlTmatmulError::AllocationTooSmall {
                name: "dax window",
                have: dax_size,
                need: used_len,
            });
        }
        Self::open_file_window(&dax_path, 0, used_len, "dax")
    }

    fn open_physical(path: &str, hpa_base: u64, used_len: usize) -> Result<Self, CxlTmatmulError> {
        let page_size = page_size();
        let page_mask = u64::try_from(page_size - 1).map_err(|_| CxlTmatmulError::SizeOverflow)?;
        let page_offset =
            usize::try_from(hpa_base & page_mask).map_err(|_| CxlTmatmulError::SizeOverflow)?;
        let map_offset = hpa_base
            .checked_sub(u64::try_from(page_offset).map_err(|_| CxlTmatmulError::SizeOverflow)?)
            .ok_or(CxlTmatmulError::SizeOverflow)?;
        let map_len = page_align_len(
            used_len
                .checked_add(page_offset)
                .ok_or(CxlTmatmulError::SizeOverflow)?,
        )?;
        let mut map = Self::mmap_file(path, map_offset, map_len, "physical hpa")?;
        map.ptr = unsafe { map.map_ptr.add(page_offset) };
        Ok(map)
    }

    fn open_file_window(
        path: &str,
        offset: u64,
        used_len: usize,
        kind: &str,
    ) -> Result<Self, CxlTmatmulError> {
        let map_len = page_align_len(used_len)?;
        Self::mmap_file(path, offset, map_len, kind)
    }

    fn mmap_file(
        path: &str,
        offset: u64,
        map_len: usize,
        kind: &str,
    ) -> Result<Self, CxlTmatmulError> {
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        if kind == "physical hpa" {
            options.custom_flags(libc::O_SYNC);
        }
        let file = options
            .open(path)
            .map_err(|e| CxlTmatmulError::Io(format!("open {path}: {e}")))?;
        let offset = libc::off_t::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                offset,
            )
        };
        drop(file);
        if ptr == libc::MAP_FAILED {
            return Err(CxlTmatmulError::Io(format!(
                "mmap {kind} {path} offset=0x{offset:x} len=0x{map_len:x}: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self {
            ptr: ptr.cast(),
            map_ptr: ptr.cast(),
            map_len,
        })
    }
}

#[cfg(unix)]
impl Drop for StagingMap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.map_ptr.cast(), self.map_len);
        }
    }
}

#[cfg(unix)]
unsafe fn stage_bytes(base: *mut u8, offset: u64, data: &[u8]) -> Result<(), CxlTmatmulError> {
    let offset = usize::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
    ptr::copy_nonoverlapping(data.as_ptr(), base.add(offset), data.len());
    flush_range(base.add(offset), data.len());
    Ok(())
}

#[cfg(unix)]
fn read_dax_size(path: &str) -> Result<usize, CxlTmatmulError> {
    let base = path.rsplit('/').next().unwrap_or(path);
    let sysfs = format!("/sys/bus/dax/devices/{base}/size");
    let mut text = String::new();
    File::open(&sysfs)
        .and_then(|mut f| f.read_to_string(&mut text))
        .map_err(|e| CxlTmatmulError::Io(format!("read {sysfs}: {e}")))?;
    let value = parse_u64_text(text.trim()).ok_or_else(|| {
        CxlTmatmulError::Io(format!("invalid dax size in {sysfs}: {:?}", text.trim()))
    })?;
    usize::try_from(value).map_err(|_| CxlTmatmulError::SizeOverflow)
}

#[cfg(unix)]
fn cxl_tmatmul_device_path() -> String {
    std::env::var("HETGPU_CXL_TMATMUL_DEVICE")
        .or_else(|_| std::env::var("HETGPU_TMATMUL_DEVICE"))
        .or_else(|_| std::env::var("CXL_TMATMUL_DEVICE"))
        .unwrap_or_else(|_| DEFAULT_DEVICE_PATH.to_string())
}

#[cfg(unix)]
fn cxl_tmatmul_dax_path() -> String {
    std::env::var("HETGPU_CXL_TMATMUL_DAX")
        .or_else(|_| std::env::var("HETGPU_TMATMUL_DAX"))
        .or_else(|_| std::env::var("CXL_DAX_PATH"))
        .unwrap_or_else(|_| DEFAULT_DAX_PATH.to_string())
}

#[cfg(unix)]
fn cxl_tmatmul_hpa_base() -> Result<Option<u64>, CxlTmatmulError> {
    env_u64_any(&[
        "HETGPU_CXL_TMATMUL_HPA_BASE",
        "HETGPU_TMATMUL_HPA_BASE",
        "CXL_TMATMUL_HPA_BASE",
    ])
}

#[cfg(unix)]
fn cxl_tmatmul_hpa_size() -> Result<Option<usize>, CxlTmatmulError> {
    env_u64_any(&[
        "HETGPU_CXL_TMATMUL_HPA_SIZE",
        "HETGPU_TMATMUL_HPA_SIZE",
        "CXL_TMATMUL_HPA_SIZE",
    ])?
    .map(|value| usize::try_from(value).map_err(|_| CxlTmatmulError::SizeOverflow))
    .transpose()
}

#[cfg(unix)]
fn cxl_tmatmul_mem_path() -> String {
    std::env::var("HETGPU_CXL_TMATMUL_MEM_PATH")
        .or_else(|_| std::env::var("HETGPU_TMATMUL_MEM_PATH"))
        .unwrap_or_else(|_| "/dev/mem".to_string())
}

fn env_u64_any(keys: &[&str]) -> Result<Option<u64>, CxlTmatmulError> {
    for key in keys {
        match std::env::var(key) {
            Ok(value) => {
                let parsed = parse_u64_text(&value).ok_or_else(|| {
                    CxlTmatmulError::Io(format!("invalid {key}={value:?}, expected integer"))
                })?;
                return Ok(Some(parsed));
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(e) => {
                return Err(CxlTmatmulError::Io(format!("read {key}: {e}")));
            }
        }
    }
    Ok(None)
}

fn page_size() -> usize {
    #[cfg(unix)]
    {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size > 0 {
            return usize::try_from(page_size).unwrap_or(4096);
        }
    }
    4096
}

fn page_align_len(len: usize) -> Result<usize, CxlTmatmulError> {
    let page = page_size();
    len.checked_add(page - 1)
        .map(|value| value & !(page - 1))
        .ok_or(CxlTmatmulError::SizeOverflow)
}

fn timeout_ms_or_default(timeout_ms: u32) -> u32 {
    if timeout_ms == 0 {
        std::env::var("HETGPU_CXL_TMATMUL_TIMEOUT_MS")
            .ok()
            .and_then(|v| parse_u64_text(&v))
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(DEFAULT_TIMEOUT_MS)
    } else {
        timeout_ms
    }
}

#[cfg(unix)]
const fn ioctl_request(dir: u64, ty: u64, nr: u64, size: u64) -> libc::c_ulong {
    const IOC_NRBITS: u64 = 8;
    const IOC_TYPEBITS: u64 = 8;
    const IOC_SIZEBITS: u64 = 14;
    const IOC_NRSHIFT: u64 = 0;
    const IOC_TYPESHIFT: u64 = IOC_NRSHIFT + IOC_NRBITS;
    const IOC_SIZESHIFT: u64 = IOC_TYPESHIFT + IOC_TYPEBITS;
    const IOC_DIRSHIFT: u64 = IOC_SIZESHIFT + IOC_SIZEBITS;

    ((dir << IOC_DIRSHIFT) | (ty << IOC_TYPESHIFT) | (nr << IOC_NRSHIFT) | (size << IOC_SIZESHIFT))
        as libc::c_ulong
}

#[cfg(unix)]
fn cxl_type2_tmatmul_get_info_ioctl() -> libc::c_ulong {
    const IOC_READ: u64 = 2;
    ioctl_request(
        IOC_READ,
        0xCE,
        0x00,
        std::mem::size_of::<CxlType2TmatmulInfo>() as u64,
    )
}

#[cfg(unix)]
fn cxl_type2_tmatmul_run_csr_only_ioctl() -> libc::c_ulong {
    const IOC_READ: u64 = 2;
    const IOC_WRITE: u64 = 1;
    ioctl_request(
        IOC_READ | IOC_WRITE,
        0xCE,
        0x02,
        std::mem::size_of::<CxlType2TmatmulCsrRun>() as u64,
    )
}

#[cfg(unix)]
fn cxl_type2_tmatmul_run_csr_only_ioctl_compat() -> libc::c_ulong {
    const IOC_READ: u64 = 2;
    const IOC_WRITE: u64 = 1;
    ioctl_request(
        IOC_READ | IOC_WRITE,
        0xCE,
        0x03,
        std::mem::size_of::<CxlType2TmatmulCsrRun>() as u64,
    )
}

#[cfg(unix)]
fn cxl_type2_tmatmul_run_csr_only_ioctl_requests() -> [libc::c_ulong; 2] {
    [
        cxl_type2_tmatmul_run_csr_only_ioctl(),
        cxl_type2_tmatmul_run_csr_only_ioctl_compat(),
    ]
}

#[cfg(target_arch = "x86_64")]
unsafe fn flush_range(start: *const u8, len: usize) {
    use std::arch::x86_64::{_mm_clflush, _mm_sfence};

    if len == 0 {
        return;
    }
    let line = 64usize;
    let mut addr = (start as usize) & !(line - 1);
    let end = ((start as usize)
        .saturating_add(len)
        .saturating_add(line - 1))
        & !(line - 1);
    while addr < end {
        _mm_clflush(addr as *const u8);
        addr = addr.saturating_add(line);
    }
    _mm_sfence();
}

#[cfg(target_arch = "x86_64")]
unsafe fn invalidate_range(start: *const u8, len: usize) {
    use std::arch::x86_64::{_mm_clflush, _mm_lfence};

    if len == 0 {
        return;
    }
    let line = 64usize;
    let mut addr = (start as usize) & !(line - 1);
    let end = ((start as usize)
        .saturating_add(len)
        .saturating_add(line - 1))
        & !(line - 1);
    while addr < end {
        _mm_clflush(addr as *const u8);
        addr = addr.saturating_add(line);
    }
    _mm_lfence();
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn flush_range(_start: *const u8, _len: usize) {}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn invalidate_range(_start: *const u8, _len: usize) {}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn require_allocation(name: &'static str, have: usize, need: usize) -> Result<(), CxlTmatmulError> {
    if have < need {
        Err(CxlTmatmulError::AllocationTooSmall { name, have, need })
    } else {
        Ok(())
    }
}

fn sanitize_kernel_name(kernel_name: &str) -> String {
    let mut out: String = kernel_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
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
    fn assembles_smoke_program_from_tmatmul_assembly() {
        let asm = format!(
            "
            ; generated hardware matmul fallback
            ldv v0,{input:#x}
            tmatmul_import v0
            tmatmul_go {matrix:#x}
            tmatmul_export v1
            sv v1,{output:#x}
            stall
            ",
            input = TMATMUL_DPA_INPUT,
            matrix = TMATMUL_DPA_MATRIX,
            output = TMATMUL_DPA_OUTPUT,
        );

        let labels = std::collections::HashMap::new();
        let program = assemble_tmatmul_program(&asm, &labels).unwrap();

        assert_eq!(program, encode_smoke_program());
    }

    #[test]
    fn assembler_resolves_param_labels_for_hardware_fallback() {
        let asm = "
            ; BIND PARAM_0 matrix
            ; BIND PARAM_1 vector
            ; BIND PARAM_4 output
            ldv v0,PARAM_1
            tmatmul_import v0
            tmatmul_go PARAM_0
            tmatmul_export v1
            sv v1,PARAM_4
            stall
        ";
        let labels = std::collections::HashMap::from([
            ("PARAM_0".to_string(), TMATMUL_DPA_MATRIX),
            ("PARAM_1".to_string(), TMATMUL_DPA_INPUT),
            ("PARAM_4".to_string(), TMATMUL_DPA_OUTPUT),
        ]);

        let program = assemble_tmatmul_program(asm, &labels).unwrap();

        assert_eq!(program, encode_smoke_program());
    }

    #[test]
    fn layout_requires_program_end() {
        assert_eq!(
            required_dax_len(),
            TMATMUL_DPA_PROGRAM as usize + TMATMUL_PROGRAM_BYTES
        );
    }

    #[cfg(unix)]
    #[test]
    fn run_csr_only_ioctl_requests_cover_known_abis() {
        let requests = cxl_type2_tmatmul_run_csr_only_ioctl_requests();

        assert_eq!(requests[0], 0xC040CE02 as libc::c_ulong);
        assert_eq!(requests[1], 0xC040CE03 as libc::c_ulong);
    }

    #[test]
    fn staging_map_len_is_page_aligned() {
        let page = page_size();

        assert_eq!(page_align_len(1).unwrap(), page);
        assert_eq!(page_align_len(page).unwrap(), page);
        assert_eq!(page_align_len(page + 1).unwrap(), page * 2);
    }

    #[cfg(unix)]
    #[test]
    fn hardware_smoke_runs_when_requested() {
        if !env_flag("HETGPU_CXL_TMATMUL_HW_SMOKE") {
            return;
        }

        let (device, info, matrix, input, mut output, program) = hardware_smoke_setup();
        let started = std::time::Instant::now();
        let status = submit_prepared_hardware_matmul(
            &device,
            &program,
            &matrix,
            &input,
            &mut output,
            10_000,
            info.dim_d,
        )
        .unwrap();
        let elapsed = started.elapsed();

        assert!(
            output.iter().any(|&byte| byte != 0xa5),
            "output staging window was not overwritten"
        );
        eprintln!(
            "tmatmul hardware smoke dim={} matrix={}B vector={}B elapsed_us={} status={:?} output_prefix={:02x?}",
            info.dim_d,
            matrix.len(),
            input.len(),
            elapsed.as_micros(),
            status,
            &output[..output.len().min(16)]
        );
    }

    #[cfg(unix)]
    #[test]
    fn hardware_benchmark_runs_when_requested() {
        if !env_flag("HETGPU_CXL_TMATMUL_HW_BENCH") {
            return;
        }

        let iters = std::env::var("HETGPU_CXL_TMATMUL_HW_BENCH_ITERS")
            .ok()
            .and_then(|value| parse_u64_text(&value))
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .unwrap_or(5);
        let (device, info, matrix, input, mut output, program) = hardware_smoke_setup();
        let mut elapsed_us = Vec::with_capacity(iters);

        for iter in 0..iters {
            let started = std::time::Instant::now();
            let status = submit_prepared_hardware_matmul(
                &device,
                &program,
                &matrix,
                &input,
                &mut output,
                10_000,
                info.dim_d,
            )
            .unwrap();
            let elapsed = started.elapsed().as_micros();
            elapsed_us.push(elapsed);
            eprintln!(
                "tmatmul bench iter={} elapsed_us={} status={:?}",
                iter, elapsed, status
            );
        }

        let total: u128 = elapsed_us.iter().copied().sum();
        let min = elapsed_us.iter().copied().min().unwrap_or(0);
        let max = elapsed_us.iter().copied().max().unwrap_or(0);
        eprintln!(
            "tmatmul bench summary dim={} matrix={}B vector={}B iters={} avg_us={} min_us={} max_us={}",
            info.dim_d,
            matrix.len(),
            input.len(),
            elapsed_us.len(),
            total / elapsed_us.len() as u128,
            min,
            max
        );
    }

    #[cfg(unix)]
    fn hardware_smoke_setup() -> (
        File,
        CxlType2TmatmulInfo,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
    ) {
        let device_path = cxl_tmatmul_device_path();
        let device = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&device_path)
            .unwrap_or_else(|e| panic!("open {device_path}: {e}"));
        let info = get_info(&device).unwrap();
        assert_eq!(info.version, CXL_TYPE2_TMATMUL_UAPI_VERSION);
        assert_ne!(info.dim_d, 0);

        let dim = usize::try_from(info.dim_d).unwrap();
        let matrix_len = matrix_bytes(dim).unwrap();
        let vector_len = vector_bytes(dim).unwrap();
        let matrix = vec![0u8; matrix_len];
        let mut input = vec![0u8; vector_len];
        for value in input.chunks_exact_mut(2) {
            value.copy_from_slice(&0x0100u16.to_le_bytes());
        }
        let mut output = vec![0u8; vector_len];
        let program = encode_smoke_program();

        (device, info, matrix, input, output, program)
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
