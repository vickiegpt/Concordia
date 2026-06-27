use std::collections::HashMap;
#[cfg(unix)]
use std::ffi::CString;
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
const DEFAULT_PCI_ADDR: &str = "0000:3b:00.0";
const DEFAULT_BAR_INDEX: u32 = 0;
const DEFAULT_CSR_BASE: u32 = 0x1c0000;
const TMATMUL_CSR_DEV_ID: u32 = 0x544D4D31;
const CSR_PROBE_ADDR: usize = 0x10;
const CSR_PROBE_WDATA: usize = 0x14;
const CSR_PROBE_CTRL: usize = 0x18;
const CSR_PROBE_RDATA: usize = 0x1C;
const CSR_PROBE_STATUS: usize = 0x24;
const CSR_INST_BASE: usize = 0x100;
const CSR_INST_STALL_STATUS: usize = CSR_INST_BASE;
const CSR_INST_STALL_CLEAR: usize = CSR_INST_BASE + 0x04;
const CSR_INST_RST_TRIGGER: usize = CSR_INST_BASE + 0x08;
const CSR_INST_INSTR_SRC_LO: usize = CSR_INST_BASE + 0x10;
const CSR_INST_INSTR_SRC_HI: usize = CSR_INST_BASE + 0x14;
const CSR_INST_INSTR_LEN: usize = CSR_INST_BASE + 0x18;
const CSR_INST_INSTR_START: usize = CSR_INST_BASE + 0x1c;
const CSR_INST_INSTR_STATUS: usize = CSR_INST_BASE + 0x1c;
const CSR_INST_DBG_INSTR_CNT: usize = CSR_INST_BASE + 0x40;
const TMATMUL_DMA_ERROR_STATUS: u32 = 0xff;
const CXL_TYPE2_TMATMUL_RESULT_STALLED: u32 = 1 << 0;
const CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR: u32 = 1 << 2;
#[cfg(unix)]
const CUDA_IPC_HANDLE_BYTES: usize = 64;
#[cfg(unix)]
const CUDA_HOST_REGISTER_MAPPED: u32 = 0x02;
#[cfg(unix)]
const CUDA_MEMCPY_DEFAULT: i32 = 4;
#[cfg(unix)]
const CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: u32 = 1;

#[cfg(unix)]
static MATRIX_STAGE_CACHE: std::sync::Mutex<Option<(usize, usize, u64)>> =
    std::sync::Mutex::new(None);

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

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StagingBackend {
    Mmap,
    CsrProbe,
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatrixStageMode {
    Host,
    CudaDax,
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IoStageMode {
    Host,
    CudaDax,
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputStageDtype {
    F16,
    F32,
}

#[cfg(unix)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum CudaDaxSource {
    DevicePtr(usize),
    IpcHandle {
        bytes: [u8; CUDA_IPC_HANDLE_BYTES],
        hex: String,
    },
}

#[cfg(unix)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaDaxMatrixStage {
    source: CudaDaxSource,
    bytes: usize,
    dax_path: String,
    cxl_offset: u64,
    gpu: i32,
    stream: usize,
}

#[cfg(unix)]
enum MatrixStage<'a> {
    Host(&'a [u8]),
    CudaDax(CudaDaxMatrixStage),
}

#[cfg(unix)]
impl MatrixStage<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Host(bytes) => bytes.len(),
            Self::CudaDax(stage) => stage.bytes,
        }
    }

    fn cxl_offset(&self) -> u64 {
        match self {
            Self::Host(_) => matrix_dpa_offset().unwrap_or(TMATMUL_DPA_MATRIX),
            Self::CudaDax(stage) => stage.cxl_offset,
        }
    }

    fn mode(&self) -> MatrixStageMode {
        match self {
            Self::Host(_) => MatrixStageMode::Host,
            Self::CudaDax(_) => MatrixStageMode::CudaDax,
        }
    }
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

#[cfg(unix)]
fn parse_staging_backend(value: Option<&str>) -> Result<StagingBackend, CxlTmatmulError> {
    match value.unwrap_or("mmap") {
        "" | "mmap" | "dax" | "hpa" => Ok(StagingBackend::Mmap),
        "csr" | "csr_probe" | "probe" => Ok(StagingBackend::CsrProbe),
        other => Err(CxlTmatmulError::Io(format!(
            "invalid HETGPU_CXL_TMATMUL_STAGING={other:?}, expected mmap or csr_probe"
        ))),
    }
}

#[cfg(unix)]
fn cxl_tmatmul_staging_backend() -> Result<StagingBackend, CxlTmatmulError> {
    parse_staging_backend(
        std::env::var("HETGPU_CXL_TMATMUL_STAGING")
            .or_else(|_| std::env::var("HETGPU_TMATMUL_STAGING"))
            .ok()
            .as_deref(),
    )
}

#[cfg(unix)]
pub(crate) fn parse_matrix_stage_mode(
    value: Option<&str>,
) -> Result<MatrixStageMode, CxlTmatmulError> {
    match value.unwrap_or("host").trim().to_ascii_lowercase().as_str() {
        "" | "host" | "cpu" | "mmap" | "dax" => Ok(MatrixStageMode::Host),
        "cuda_dax" | "cuda-dax" | "nvgpu_dax" | "nvgpu-dax" => Ok(MatrixStageMode::CudaDax),
        other => Err(CxlTmatmulError::Io(format!(
            "invalid HETGPU_TMATMUL_MATRIX_STAGE={other:?}, expected host or cuda_dax"
        ))),
    }
}

#[cfg(unix)]
pub(crate) fn matrix_stage_mode() -> Result<MatrixStageMode, CxlTmatmulError> {
    parse_matrix_stage_mode(
        std::env::var("HETGPU_TMATMUL_MATRIX_STAGE")
            .or_else(|_| std::env::var("HETGPU_CXL_TMATMUL_MATRIX_STAGE"))
            .ok()
            .as_deref(),
    )
}

#[cfg(unix)]
pub(crate) fn matrix_stage_cuda_dax_enabled() -> bool {
    matches!(matrix_stage_mode(), Ok(MatrixStageMode::CudaDax))
}

#[cfg(unix)]
pub(crate) fn parse_io_stage_mode(value: Option<&str>) -> Result<IoStageMode, CxlTmatmulError> {
    match value.unwrap_or("host").trim().to_ascii_lowercase().as_str() {
        "" | "host" | "cpu" | "mmap" | "dax" => Ok(IoStageMode::Host),
        "cuda_dax" | "cuda-dax" | "nvgpu_dax" | "nvgpu-dax" => Ok(IoStageMode::CudaDax),
        other => Err(CxlTmatmulError::Io(format!(
            "invalid HETGPU_TMATMUL_IO_STAGE={other:?}, expected host or cuda_dax"
        ))),
    }
}

#[cfg(unix)]
pub(crate) fn io_stage_mode() -> Result<IoStageMode, CxlTmatmulError> {
    parse_io_stage_mode(
        std::env::var("HETGPU_TMATMUL_IO_STAGE")
            .or_else(|_| std::env::var("HETGPU_CXL_TMATMUL_IO_STAGE"))
            .ok()
            .as_deref(),
    )
}

#[cfg(unix)]
fn parse_output_stage_dtype(value: Option<&str>) -> Result<OutputStageDtype, CxlTmatmulError> {
    match value.unwrap_or("f16").trim().to_ascii_lowercase().as_str() {
        "" | "f16" | "half" | "fp16" => Ok(OutputStageDtype::F16),
        "f32" | "float" | "fp32" => Ok(OutputStageDtype::F32),
        other => Err(CxlTmatmulError::Io(format!(
            "invalid HETGPU_TMATMUL_OUTPUT_DTYPE={other:?}, expected f16 or f32"
        ))),
    }
}

#[cfg(unix)]
fn output_stage_dtype() -> Result<OutputStageDtype, CxlTmatmulError> {
    parse_output_stage_dtype(
        std::env::var("HETGPU_TMATMUL_OUTPUT_DTYPE")
            .or_else(|_| std::env::var("HETGPU_CXL_TMATMUL_OUTPUT_DTYPE"))
            .ok()
            .as_deref(),
    )
}

#[cfg(unix)]
pub(crate) fn matrix_dpa_offset() -> Result<u64, CxlTmatmulError> {
    Ok(env_u64_any(&[
        "HETGPU_TMATMUL_MATRIX_CXL_OFFSET",
        "HETGPU_CXL_TMATMUL_MATRIX_OFFSET",
        "HETGPU_TMATMUL_MATRIX_DPA",
    ])?
    .unwrap_or(TMATMUL_DPA_MATRIX))
}

#[cfg(unix)]
pub(crate) fn cuda_dax_bridge_param_json(
    cuda_device_ptr: u64,
    bytes: usize,
    count: usize,
) -> Result<String, CxlTmatmulError> {
    let source = cuda_dax_source_from_env(cuda_device_ptr as usize)?;
    let mut value = serde_json::json!({
        "file": "",
        "count": count,
        "is_pointer": true,
        "stage": "cuda_dax",
        "dtype": "nvint8",
        "bytes": bytes,
        "gpu": cuda_dax_gpu()?,
        "dax_path": cxl_tmatmul_dax_path(),
        "cxl_offset": matrix_dpa_offset()?,
    });
    match source {
        CudaDaxSource::DevicePtr(ptr) => {
            value["cuda_device_ptr"] = serde_json::Value::String(format!("0x{ptr:x}"));
        }
        CudaDaxSource::IpcHandle { hex, .. } => {
            value["cuda_ipc_handle"] = serde_json::Value::String(hex);
        }
    }
    let stream = cuda_dax_stream()?;
    if stream != 0 {
        value["stream"] = serde_json::Value::String(format!("0x{stream:x}"));
    }
    serde_json::to_string(&value).map_err(|e| CxlTmatmulError::Io(e.to_string()))
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
    let matrix_len = matrix_bytes(dim)?;
    let vector_len = vector_bytes(dim)?;
    let matrix_offset = matrix_dpa_offset()?;
    let matrix_stage_mode = matrix_stage_mode()?;
    let io_stage_mode = io_stage_mode()?;
    validate_fixed_layout_at_offsets(matrix_offset, matrix_len, vector_len, program.len())?;
    match (matrix_stage_mode, io_stage_mode) {
        (MatrixStageMode::Host, IoStageMode::Host) => {
            validate_allocations(dim, input_alloc, output_alloc, matrix_alloc)?;
        }
        (MatrixStageMode::CudaDax, IoStageMode::Host) => {
            require_allocation("input", input_alloc, vector_len)?;
            require_allocation("output", output_alloc, vector_len)?;
            if matrix_alloc != usize::MAX {
                require_allocation("matrix", matrix_alloc, matrix_len)?;
            }
        }
        (MatrixStageMode::Host, IoStageMode::CudaDax) => {
            require_allocation("matrix", matrix_alloc, matrix_len)?;
            if input_alloc != usize::MAX {
                require_allocation("input", input_alloc, vector_len)?;
            }
            if output_alloc != usize::MAX {
                require_allocation("output", output_alloc, vector_len)?;
            }
        }
        (MatrixStageMode::CudaDax, IoStageMode::CudaDax) => {
            if matrix_alloc != usize::MAX {
                require_allocation("matrix", matrix_alloc, matrix_len)?;
            }
            if input_alloc != usize::MAX {
                require_allocation("input", input_alloc, vector_len)?;
            }
            if output_alloc != usize::MAX {
                require_allocation("output", output_alloc, vector_len)?;
            }
        }
    }

    let matrix_stage = match matrix_stage_mode {
        MatrixStageMode::Host => {
            MatrixStage::Host(std::slice::from_raw_parts(matrix_ptr, matrix_len))
        }
        MatrixStageMode::CudaDax => MatrixStage::CudaDax(cuda_dax_matrix_stage(
            matrix_ptr as usize,
            matrix_len,
            matrix_offset,
        )?),
    };

    if io_stage_mode == IoStageMode::CudaDax {
        return submit_prepared_hardware_matmul_cuda_io(
            &device,
            &program,
            matrix_stage,
            input_ptr as usize,
            output_ptr as usize,
            vector_len,
            timeout_ms,
            info.dim_d,
        );
    }

    let input = std::slice::from_raw_parts(input_ptr, vector_len);
    let output = std::slice::from_raw_parts_mut(output_ptr, vector_len);

    submit_prepared_hardware_matmul(
        &device,
        &program,
        matrix_stage,
        input,
        output,
        timeout_ms,
        info.dim_d,
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
        "tmatmul_go" | "tmatmul_go_nvint8" => {
            if op == "tmatmul_go" {
                require_arg_count(line_no, tokens, 2)?;
            } else {
                require_arg_count(line_no, tokens, 3)?;
                let width = parse_u64_text(tokens[2]).ok_or_else(|| {
                    CxlTmatmulError::AssembleFailed(format!(
                        "line {line_no}: tmatmul_go_nvint8 expects numeric stage width, got '{}'",
                        tokens[2]
                    ))
                })?;
                if width != 4 {
                    return Err(CxlTmatmulError::AssembleFailed(format!(
                        "line {line_no}: tmatmul_go_nvint8 only supports stage width 4, got {width}"
                    )));
                }
            }
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
    validate_fixed_layout_at_offsets(TMATMUL_DPA_MATRIX, matrix_len, vector_len, program_len)
}

fn validate_fixed_layout_at_offsets(
    matrix_offset: u64,
    matrix_len: usize,
    vector_len: usize,
    program_len: usize,
) -> Result<(), CxlTmatmulError> {
    let matrix_end = range_end(matrix_offset, matrix_len)?;
    let input_end = range_end(TMATMUL_DPA_INPUT, vector_len)?;
    let output_end = range_end(TMATMUL_DPA_OUTPUT, vector_len)?;
    let program_end = range_end(TMATMUL_DPA_PROGRAM, program_len)?;

    if ranges_overlap(matrix_offset, matrix_len, TMATMUL_DPA_INPUT, vector_len)? {
        return Err(CxlTmatmulError::Device(format!(
            "fixed DAX layout overlaps matrix/input: matrix=[0x{matrix_offset:x},0x{matrix_end:x}) input=[0x{TMATMUL_DPA_INPUT:x},0x{input_end:x})"
        )));
    }
    if ranges_overlap(matrix_offset, matrix_len, TMATMUL_DPA_OUTPUT, vector_len)? {
        return Err(CxlTmatmulError::Device(format!(
            "fixed DAX layout overlaps matrix/output: matrix=[0x{matrix_offset:x},0x{matrix_end:x}) output=[0x{TMATMUL_DPA_OUTPUT:x},0x{output_end:x})"
        )));
    }
    if ranges_overlap(matrix_offset, matrix_len, TMATMUL_DPA_PROGRAM, program_len)? {
        return Err(CxlTmatmulError::Device(format!(
            "fixed DAX layout overlaps matrix/program: matrix=[0x{matrix_offset:x},0x{matrix_end:x}) program=[0x{TMATMUL_DPA_PROGRAM:x},0x{program_end:x})"
        )));
    }
    if ranges_overlap(
        TMATMUL_DPA_INPUT,
        vector_len,
        TMATMUL_DPA_OUTPUT,
        vector_len,
    )? {
        return Err(CxlTmatmulError::Device(format!(
            "fixed DAX layout overlaps input/output: input_end=0x{input_end:x} output=0x{TMATMUL_DPA_OUTPUT:x}"
        )));
    }
    if ranges_overlap(
        TMATMUL_DPA_OUTPUT,
        vector_len,
        TMATMUL_DPA_PROGRAM,
        program_len,
    )? {
        return Err(CxlTmatmulError::Device(format!(
            "fixed DAX layout overlaps output/program: output_end=0x{output_end:x} program=0x{TMATMUL_DPA_PROGRAM:x}"
        )));
    }
    if program_end < TMATMUL_DPA_PROGRAM as usize {
        return Err(CxlTmatmulError::SizeOverflow);
    }
    Ok(())
}

fn ranges_overlap(
    a_start: u64,
    a_len: usize,
    b_start: u64,
    b_len: usize,
) -> Result<bool, CxlTmatmulError> {
    let a_start = usize::try_from(a_start).map_err(|_| CxlTmatmulError::SizeOverflow)?;
    let b_start = usize::try_from(b_start).map_err(|_| CxlTmatmulError::SizeOverflow)?;
    let a_end = a_start
        .checked_add(a_len)
        .ok_or(CxlTmatmulError::SizeOverflow)?;
    let b_end = b_start
        .checked_add(b_len)
        .ok_or(CxlTmatmulError::SizeOverflow)?;
    Ok(a_start < b_end && b_start < a_end)
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
    matrix: MatrixStage<'_>,
    input: &[u8],
    output: &mut [u8],
    timeout_ms: u32,
    dim_d: u32,
) -> Result<CxlTmatmulRunStatus, CxlTmatmulError> {
    let matrix_offset = matrix.cxl_offset();
    let used_len = [
        range_end(matrix_offset, matrix.len())?,
        range_end(TMATMUL_DPA_INPUT, input.len())?,
        range_end(TMATMUL_DPA_OUTPUT, output.len())?,
        range_end(TMATMUL_DPA_PROGRAM, program.len())?,
    ]
    .into_iter()
    .max()
    .ok_or(CxlTmatmulError::SizeOverflow)?;

    let mut staging = StagingMap::open_for_matrix_stage(used_len, matrix.mode())?;
    {
        match &matrix {
            MatrixStage::Host(matrix) => {
                let matrix_key = (matrix.as_ptr() as usize, matrix.len(), matrix_offset);
                let assume_static_matrix = env_flag("HETGPU_CXL_TMATMUL_ASSUME_STATIC_MATRIX");
                let matrix_already_staged = assume_static_matrix
                    && MATRIX_STAGE_CACHE
                        .lock()
                        .map(|cache| *cache == Some(matrix_key))
                        .unwrap_or(false);
                if !matrix_already_staged {
                    staging.stage_bytes(matrix_offset, matrix)?;
                    if assume_static_matrix {
                        if let Ok(mut cache) = MATRIX_STAGE_CACHE.lock() {
                            *cache = Some(matrix_key);
                        }
                    }
                }
            }
            MatrixStage::CudaDax(stage) => {
                staging.stage_cuda_dax_matrix(stage)?;
            }
        }
        staging.stage_bytes(TMATMUL_DPA_INPUT, input)?;
        staging.fill_bytes(TMATMUL_DPA_OUTPUT, 0xa5, output.len())?;
        staging.stage_bytes(TMATMUL_DPA_PROGRAM, program)?;
    }

    let status =
        run_staged_hardware_matmul(device, &mut staging, program.len(), timeout_ms, dim_d)?;
    staging.read_bytes(TMATMUL_DPA_OUTPUT, output)?;
    Ok(status)
}

#[cfg(unix)]
fn submit_prepared_hardware_matmul_cuda_io(
    device: &File,
    program: &[u8],
    matrix: MatrixStage<'_>,
    input_device_ptr: usize,
    output_device_ptr: usize,
    vector_len: usize,
    timeout_ms: u32,
    dim_d: u32,
) -> Result<CxlTmatmulRunStatus, CxlTmatmulError> {
    let matrix_offset = matrix.cxl_offset();
    let output_dtype = output_stage_dtype()?;
    let used_len = [
        range_end(matrix_offset, matrix.len())?,
        range_end(TMATMUL_DPA_INPUT, vector_len)?,
        range_end(TMATMUL_DPA_OUTPUT, vector_len)?,
        range_end(TMATMUL_DPA_PROGRAM, program.len())?,
    ]
    .into_iter()
    .max()
    .ok_or(CxlTmatmulError::SizeOverflow)?;

    let mut staging = StagingMap::open_cuda_dax(used_len)?;
    {
        match &matrix {
            MatrixStage::Host(matrix) => {
                staging.stage_bytes(matrix_offset, matrix)?;
            }
            MatrixStage::CudaDax(stage) => {
                staging.stage_cuda_dax_matrix(stage)?;
            }
        }
        staging.stage_cuda_dax_device_to_offset(
            input_device_ptr,
            vector_len,
            TMATMUL_DPA_INPUT,
            "input",
        )?;
        staging.fill_bytes(TMATMUL_DPA_OUTPUT, 0xa5, vector_len)?;
        staging.stage_bytes(TMATMUL_DPA_PROGRAM, program)?;
    }

    let status =
        run_staged_hardware_matmul(device, &mut staging, program.len(), timeout_ms, dim_d)?;
    match output_dtype {
        OutputStageDtype::F16 => {
            staging.copy_cuda_dax_offset_to_device(
                output_device_ptr,
                vector_len,
                TMATMUL_DPA_OUTPUT,
                "output",
            )?;
        }
        OutputStageDtype::F32 => {
            staging.copy_cuda_dax_f16_offset_to_f32_device(
                output_device_ptr,
                vector_len,
                TMATMUL_DPA_OUTPUT,
                "output f16->f32",
            )?;
        }
    }
    Ok(status)
}

#[cfg(unix)]
fn run_staged_hardware_matmul(
    device: &File,
    staging: &mut StagingMap,
    program_len: usize,
    timeout_ms: u32,
    dim_d: u32,
) -> Result<CxlTmatmulRunStatus, CxlTmatmulError> {
    if cxl_tmatmul_bar_run_enabled() {
        return run_prepared_hardware_matmul_via_bar(program_len, timeout_ms, dim_d);
    }

    let max_retries = cxl_tmatmul_run_retries();
    for run_attempt in 0..=max_retries {
        staging.reset_instruction_engine_before_run()?;
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
            if should_retry_csr_run_timeout(&saved_error, &status, run_attempt, max_retries, dim_d)
            {
                eprintln!(
                    "[CXL TMatmul] RUN_CSR_ONLY timed out before stall; retrying {}/{}; status={:?}",
                    run_attempt + 1,
                    max_retries,
                    status
                );
                continue;
            }
            if should_fallback_to_bar_run(&saved_error, &status, dim_d)
                && !cxl_tmatmul_bar_fallback_disabled()
            {
                eprintln!(
                    "[CXL TMatmul] RUN_CSR_ONLY unavailable or stuck before stall; using BAR launch fallback; status={:?}",
                    status
                );
                return run_prepared_hardware_matmul_via_bar(program_len, timeout_ms, dim_d);
            }
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

        return Ok(status);
    }

    Err(CxlTmatmulError::Device(
        "RUN_CSR_ONLY retry loop exited unexpectedly".to_string(),
    ))
}

#[cfg(unix)]
fn run_prepared_hardware_matmul_via_bar(
    program_len: usize,
    timeout_ms: u32,
    dim_d: u32,
) -> Result<CxlTmatmulRunStatus, CxlTmatmulError> {
    let program_len = u32::try_from(program_len).map_err(|_| CxlTmatmulError::SizeOverflow)?;
    let bar = StagingMap::open_csr_probe(0)?;
    bar.reset_instruction_engine_before_run()?;
    bar.run_instance0_program(TMATMUL_DPA_PROGRAM, program_len, timeout_ms, dim_d)
}

#[cfg(unix)]
enum StagingMap {
    Mmap {
        ptr: *mut u8,
        map_ptr: *mut u8,
        map_len: usize,
    },
    CsrProbe {
        bar: *mut u8,
        map_len: usize,
        csr_base: u32,
    },
}

#[cfg(unix)]
impl StagingMap {
    fn open_for_matrix_stage(
        used_len: usize,
        matrix_stage: MatrixStageMode,
    ) -> Result<Self, CxlTmatmulError> {
        match matrix_stage {
            MatrixStageMode::Host => Self::open(used_len),
            MatrixStageMode::CudaDax => Self::open_cuda_dax(used_len),
        }
    }

    fn open(used_len: usize) -> Result<Self, CxlTmatmulError> {
        if cxl_tmatmul_staging_backend()? == StagingBackend::CsrProbe {
            return Self::open_csr_probe(used_len);
        }

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

    fn open_cuda_dax(used_len: usize) -> Result<Self, CxlTmatmulError> {
        if cxl_tmatmul_staging_backend()? == StagingBackend::CsrProbe {
            return Err(CxlTmatmulError::Device(
                "cuda_dax matrix staging requires DAX mmap staging, not csr_probe".to_string(),
            ));
        }
        if cxl_tmatmul_hpa_base()?.is_some() {
            return Err(CxlTmatmulError::Device(
                "cuda_dax matrix staging requires /dev/dax mapping; unset HETGPU_CXL_TMATMUL_HPA_BASE".to_string(),
            ));
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
        match Self::mmap_file(path, map_offset, map_len, "physical hpa")? {
            Self::Mmap {
                map_ptr, map_len, ..
            } => Ok(Self::Mmap {
                ptr: unsafe { map_ptr.add(page_offset) },
                map_ptr,
                map_len,
            }),
            Self::CsrProbe { .. } => unreachable!(),
        }
    }

    fn open_file_window(
        path: &str,
        offset: u64,
        used_len: usize,
        kind: &str,
    ) -> Result<Self, CxlTmatmulError> {
        let map_len = if kind == "dax" {
            align_len(used_len, read_dax_align(path)?)?
        } else {
            page_align_len(used_len)?
        };
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
        if kind == "physical hpa" || kind == "dax" {
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
        Ok(Self::Mmap {
            ptr: ptr.cast(),
            map_ptr: ptr.cast(),
            map_len,
        })
    }

    fn open_csr_probe(used_len: usize) -> Result<Self, CxlTmatmulError> {
        if used_len > u32::MAX as usize {
            return Err(CxlTmatmulError::AllocationTooSmall {
                name: "csr probe address window",
                have: u32::MAX as usize,
                need: used_len,
            });
        }

        let device_path = cxl_tmatmul_device_path();
        let pci_addr = cxl_tmatmul_pci_addr(&device_path);
        let bar_index = cxl_tmatmul_bar_index()?;
        let csr_base = cxl_tmatmul_csr_base()?;
        let resource_path = format!("/sys/bus/pci/devices/{pci_addr}/resource{bar_index}");

        let mut options = OpenOptions::new();
        options.read(true).write(true).custom_flags(libc::O_SYNC);
        let file = options
            .open(&resource_path)
            .map_err(|e| CxlTmatmulError::Io(format!("open {resource_path}: {e}")))?;
        let map_len = usize::try_from(
            file.metadata()
                .map_err(|e| CxlTmatmulError::Io(format!("metadata {resource_path}: {e}")))?
                .len(),
        )
        .map_err(|_| CxlTmatmulError::SizeOverflow)?;
        if (csr_base as usize)
            .checked_add(0x200)
            .ok_or(CxlTmatmulError::SizeOverflow)?
            > map_len
        {
            return Err(CxlTmatmulError::Io(format!(
                "BAR{bar_index} too small for CSR base 0x{csr_base:x}: size=0x{map_len:x}"
            )));
        }
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        drop(file);
        if ptr == libc::MAP_FAILED {
            return Err(CxlTmatmulError::Io(format!(
                "mmap csr probe {resource_path} len=0x{map_len:x}: {}",
                std::io::Error::last_os_error()
            )));
        }
        let map = Self::CsrProbe {
            bar: ptr.cast(),
            map_len,
            csr_base,
        };
        let dev_id = map.csr_mmio_rd32(0)?;
        if dev_id != TMATMUL_CSR_DEV_ID {
            return Err(CxlTmatmulError::Device(format!(
                "tmatmul CSR not found at {pci_addr} BAR{bar_index}+0x{csr_base:x}: dev_id=0x{dev_id:08x}"
            )));
        }
        Ok(map)
    }

    fn stage_bytes(&mut self, offset: u64, data: &[u8]) -> Result<(), CxlTmatmulError> {
        match self {
            Self::Mmap { ptr, .. } => {
                let offset = usize::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
                unsafe {
                    ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
                    flush_range(ptr.add(offset), data.len());
                }
                Ok(())
            }
            Self::CsrProbe { .. } => self.csr_write_bytes(offset, data),
        }
    }

    fn stage_cuda_dax_matrix(&mut self, stage: &CudaDaxMatrixStage) -> Result<(), CxlTmatmulError> {
        self.stage_cuda_dax_stage(stage, "matrix")
    }

    fn stage_cuda_dax_stage(
        &mut self,
        stage: &CudaDaxMatrixStage,
        label: &str,
    ) -> Result<(), CxlTmatmulError> {
        match self {
            Self::Mmap {
                map_ptr, map_len, ..
            } => {
                let dst_end = range_end(stage.cxl_offset, stage.bytes)?;
                if dst_end > *map_len {
                    return Err(CxlTmatmulError::AllocationTooSmall {
                        name: "cuda_dax mapped window",
                        have: *map_len,
                        need: dst_end,
                    });
                }
                unsafe {
                    CudaRuntime::load()?.copy_device_to_dax(stage, *map_ptr, *map_len, label)?;
                }
                Ok(())
            }
            Self::CsrProbe { .. } => Err(CxlTmatmulError::Device(
                "cuda_dax staging cannot target csr_probe staging".to_string(),
            )),
        }
    }

    fn stage_cuda_dax_device_to_offset(
        &mut self,
        device_ptr: usize,
        bytes: usize,
        cxl_offset: u64,
        label: &str,
    ) -> Result<(), CxlTmatmulError> {
        let stage = cuda_dax_device_stage(device_ptr, bytes, cxl_offset)?;
        self.stage_cuda_dax_stage(&stage, label)
    }

    fn copy_cuda_dax_offset_to_device(
        &mut self,
        device_ptr: usize,
        bytes: usize,
        cxl_offset: u64,
        label: &str,
    ) -> Result<(), CxlTmatmulError> {
        match self {
            Self::Mmap {
                map_ptr, map_len, ..
            } => {
                let src_end = range_end(cxl_offset, bytes)?;
                if src_end > *map_len {
                    return Err(CxlTmatmulError::AllocationTooSmall {
                        name: "cuda_dax mapped window",
                        have: *map_len,
                        need: src_end,
                    });
                }
                unsafe {
                    CudaRuntime::load()?.copy_dax_to_device(
                        device_ptr, bytes, cxl_offset, *map_ptr, *map_len, label,
                    )?;
                }
                Ok(())
            }
            Self::CsrProbe { .. } => Err(CxlTmatmulError::Device(
                "cuda_dax output copy cannot target csr_probe staging".to_string(),
            )),
        }
    }

    fn copy_cuda_dax_f16_offset_to_f32_device(
        &mut self,
        device_ptr: usize,
        f16_bytes: usize,
        cxl_offset: u64,
        label: &str,
    ) -> Result<(), CxlTmatmulError> {
        if (f16_bytes & 1) != 0 {
            return Err(CxlTmatmulError::Device(format!(
                "cuda_dax {label} source length must be even: {f16_bytes}"
            )));
        }
        match self {
            Self::Mmap { ptr, map_len, .. } => {
                let src_end = range_end(cxl_offset, f16_bytes)?;
                if src_end > *map_len {
                    return Err(CxlTmatmulError::AllocationTooSmall {
                        name: "cuda_dax mapped window",
                        have: *map_len,
                        need: src_end,
                    });
                }
                let offset =
                    usize::try_from(cxl_offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
                let element_count = f16_bytes / 2;
                let mut output = Vec::with_capacity(element_count);
                unsafe {
                    let src = ptr.add(offset);
                    invalidate_range(src, f16_bytes);
                    for i in 0..element_count {
                        let lo = ptr::read(src.add(i * 2));
                        let hi = ptr::read(src.add(i * 2 + 1));
                        output.push(cxl_f16_to_f32(u16::from_le_bytes([lo, hi])));
                    }
                    let bytes = std::slice::from_raw_parts(
                        output.as_ptr().cast::<u8>(),
                        output.len() * std::mem::size_of::<f32>(),
                    );
                    CudaRuntime::load()?.copy_host_to_device(device_ptr, bytes, label)?;
                }
                eprintln!(
                    "[CXL TMatmul] cuda_dax converted {} elems={} f16_bytes={} f32_bytes={}",
                    label,
                    element_count,
                    f16_bytes,
                    output.len() * std::mem::size_of::<f32>()
                );
                Ok(())
            }
            Self::CsrProbe { .. } => Err(CxlTmatmulError::Device(
                "cuda_dax output conversion cannot target csr_probe staging".to_string(),
            )),
        }
    }

    fn fill_bytes(&mut self, offset: u64, value: u8, len: usize) -> Result<(), CxlTmatmulError> {
        match self {
            Self::Mmap { ptr, .. } => {
                let offset = usize::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
                unsafe {
                    ptr::write_bytes(ptr.add(offset), value, len);
                    flush_range(ptr.add(offset), len);
                }
                Ok(())
            }
            Self::CsrProbe { .. } => self.csr_fill_bytes(offset, value, len),
        }
    }

    fn read_bytes(&mut self, offset: u64, out: &mut [u8]) -> Result<(), CxlTmatmulError> {
        match self {
            Self::Mmap { ptr, .. } => {
                let offset = usize::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
                unsafe {
                    invalidate_range(ptr.add(offset), out.len());
                    ptr::copy_nonoverlapping(ptr.add(offset), out.as_mut_ptr(), out.len());
                }
                Ok(())
            }
            Self::CsrProbe { .. } => self.csr_read_bytes(offset, out),
        }
    }

    fn csr_write_bytes(&self, offset: u64, data: &[u8]) -> Result<(), CxlTmatmulError> {
        require_csr_word_aligned(offset, data.len(), "write")?;
        let base = u32::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
        for (i, chunk) in data.chunks_exact(4).enumerate() {
            let addr = base
                .checked_add(u32::try_from(i * 4).map_err(|_| CxlTmatmulError::SizeOverflow)?)
                .ok_or(CxlTmatmulError::SizeOverflow)?;
            let value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            self.csr_write_word(addr, value)?;
        }
        Ok(())
    }

    fn csr_fill_bytes(&self, offset: u64, value: u8, len: usize) -> Result<(), CxlTmatmulError> {
        require_csr_word_aligned(offset, len, "fill")?;
        let base = u32::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
        let word = u32::from_le_bytes([value; 4]);
        for i in 0..(len / 4) {
            let addr = base
                .checked_add(u32::try_from(i * 4).map_err(|_| CxlTmatmulError::SizeOverflow)?)
                .ok_or(CxlTmatmulError::SizeOverflow)?;
            self.csr_write_word(addr, word)?;
        }
        Ok(())
    }

    fn csr_read_bytes(&self, offset: u64, out: &mut [u8]) -> Result<(), CxlTmatmulError> {
        require_csr_word_aligned(offset, out.len(), "read")?;
        let base = u32::try_from(offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
        for (i, chunk) in out.chunks_exact_mut(4).enumerate() {
            let addr = base
                .checked_add(u32::try_from(i * 4).map_err(|_| CxlTmatmulError::SizeOverflow)?)
                .ok_or(CxlTmatmulError::SizeOverflow)?;
            chunk.copy_from_slice(&self.csr_read_word(addr)?.to_le_bytes());
        }
        Ok(())
    }

    fn csr_write_word(&self, addr: u32, value: u32) -> Result<(), CxlTmatmulError> {
        self.csr_mmio_wr32(CSR_PROBE_ADDR, addr)?;
        self.csr_mmio_wr32(CSR_PROBE_WDATA, value)?;
        self.csr_mmio_wr32(CSR_PROBE_CTRL, 0x3)?;
        let status = self.wait_probe_status()?;
        if (status & 0x3) != 2 {
            return Err(CxlTmatmulError::Device(format!(
                "CSR DDR probe write failed at 0x{addr:x}: status=0x{status:08x}"
            )));
        }
        Ok(())
    }

    fn csr_read_word(&self, addr: u32) -> Result<u32, CxlTmatmulError> {
        self.csr_mmio_wr32(CSR_PROBE_ADDR, addr)?;
        self.csr_mmio_wr32(CSR_PROBE_CTRL, 0x1)?;
        let status = self.wait_probe_status()?;
        if (status & 0x3) != 2 {
            return Err(CxlTmatmulError::Device(format!(
                "CSR DDR probe read failed at 0x{addr:x}: status=0x{status:08x}"
            )));
        }
        self.csr_mmio_rd32(CSR_PROBE_RDATA)
    }

    fn wait_probe_status(&self) -> Result<u32, CxlTmatmulError> {
        // The FPGA holds the previous probe status until the CDC pulse for the
        // next request is observed. Match the kernel MEM_IO helper and the
        // working C++ BAR smoke so we do not consume a stale DONE status.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
        loop {
            let status = self.csr_mmio_rd32(CSR_PROBE_STATUS)?;
            if (status & 0x3) != 1 || std::time::Instant::now() >= deadline {
                return Ok(status);
            }
            std::hint::spin_loop();
        }
    }

    fn csr_mmio_rd32(&self, off: usize) -> Result<u32, CxlTmatmulError> {
        match self {
            Self::CsrProbe { bar, csr_base, .. } => {
                Ok(unsafe { ptr::read_volatile(bar.add(*csr_base as usize + off).cast::<u32>()) })
            }
            Self::Mmap { .. } => Err(CxlTmatmulError::Device(
                "CSR probe access requested for mmap staging".to_string(),
            )),
        }
    }

    fn csr_mmio_wr32(&self, off: usize, value: u32) -> Result<(), CxlTmatmulError> {
        match self {
            Self::CsrProbe { bar, csr_base, .. } => {
                unsafe {
                    ptr::write_volatile(bar.add(*csr_base as usize + off).cast::<u32>(), value);
                }
                Ok(())
            }
            Self::Mmap { .. } => Err(CxlTmatmulError::Device(
                "CSR probe access requested for mmap staging".to_string(),
            )),
        }
    }

    fn reset_instruction_engine_before_run(&self) -> Result<(), CxlTmatmulError> {
        if !matches!(self, Self::CsrProbe { .. }) {
            return Ok(());
        }
        for (off, value) in csr_probe_prelaunch_reset_sequence() {
            self.csr_mmio_wr32(off, value)?;
            if off == CSR_INST_RST_TRIGGER {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        Ok(())
    }

    fn run_instance0_program(
        &self,
        instr_addr: u64,
        instr_len: u32,
        timeout_ms: u32,
        dim_d: u32,
    ) -> Result<CxlTmatmulRunStatus, CxlTmatmulError> {
        if !matches!(self, Self::CsrProbe { .. }) {
            return Err(CxlTmatmulError::Device(
                "BAR launch requested without CSR BAR mapping".to_string(),
            ));
        }

        let timeout_ms = timeout_ms_or_default(timeout_ms);
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms as u64);
        self.csr_mmio_wr32(CSR_INST_STALL_CLEAR, 1)?;
        while self.csr_mmio_rd32(CSR_INST_STALL_STATUS)? != 0 {
            if std::time::Instant::now() >= deadline {
                let dma_status = self.csr_mmio_rd32(CSR_INST_INSTR_STATUS)?;
                let instr_count = self.csr_mmio_rd32(CSR_INST_DBG_INSTR_CNT)?;
                let status = bar_run_status(timeout_ms, dma_status, 1, instr_count, dim_d);
                return Err(CxlTmatmulError::Device(format!(
                    "BAR launch stall clear timed out; status={status:?}"
                )));
            }
            std::hint::spin_loop();
        }

        self.csr_mmio_wr32(CSR_INST_INSTR_SRC_LO, instr_addr as u32)?;
        self.csr_mmio_wr32(CSR_INST_INSTR_SRC_HI, (instr_addr >> 32) as u32)?;
        self.csr_mmio_wr32(CSR_INST_INSTR_LEN, instr_len)?;
        let before_count = self.csr_mmio_rd32(CSR_INST_DBG_INSTR_CNT)?;
        self.csr_mmio_wr32(CSR_INST_INSTR_START, 1)?;

        loop {
            let stall_status = self.csr_mmio_rd32(CSR_INST_STALL_STATUS)?;
            let dma_status = self.csr_mmio_rd32(CSR_INST_INSTR_STATUS)?;
            let after_count = self.csr_mmio_rd32(CSR_INST_DBG_INSTR_CNT)?;
            let instr_delta = after_count.wrapping_sub(before_count);
            let status = bar_run_status(timeout_ms, dma_status, stall_status, instr_delta, dim_d);
            if (status.result_flags & CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR) != 0 {
                return Err(CxlTmatmulError::Device(format!(
                    "BAR launch reported DMA_ERROR; status={status:?}"
                )));
            }
            if (status.result_flags & CXL_TYPE2_TMATMUL_RESULT_STALLED) != 0 {
                return Ok(status);
            }
            if std::time::Instant::now() >= deadline {
                return Err(CxlTmatmulError::Device(format!(
                    "BAR launch timed out before stall; status={status:?}"
                )));
            }
            std::hint::spin_loop();
        }
    }
}

#[cfg(unix)]
struct CudaRuntime {
    handle: *mut libc::c_void,
    cuda_set_device: unsafe extern "C" fn(libc::c_int) -> libc::c_int,
    cuda_host_register: unsafe extern "C" fn(*mut libc::c_void, usize, libc::c_uint) -> libc::c_int,
    cuda_host_get_device_pointer: unsafe extern "C" fn(
        *mut *mut libc::c_void,
        *mut libc::c_void,
        libc::c_uint,
    ) -> libc::c_int,
    cuda_memcpy_async: unsafe extern "C" fn(
        *mut libc::c_void,
        *const libc::c_void,
        usize,
        libc::c_int,
        *mut libc::c_void,
    ) -> libc::c_int,
    cuda_stream_synchronize: unsafe extern "C" fn(*mut libc::c_void) -> libc::c_int,
    cuda_host_unregister: unsafe extern "C" fn(*mut libc::c_void) -> libc::c_int,
    cuda_ipc_open_mem_handle: Option<
        unsafe extern "C" fn(
            *mut *mut libc::c_void,
            *mut libc::c_void,
            libc::c_uint,
        ) -> libc::c_int,
    >,
    cuda_ipc_close_mem_handle: Option<unsafe extern "C" fn(*mut libc::c_void) -> libc::c_int>,
}

#[cfg(unix)]
impl CudaRuntime {
    fn load() -> Result<Self, CxlTmatmulError> {
        let mut paths = Vec::new();
        if let Ok(path) = std::env::var("HETGPU_TMATMUL_CUDART") {
            paths.push(path);
        }
        if let Ok(path) = std::env::var("HETGPU_CUDART_PATH") {
            paths.push(path);
        }
        paths.push("libcudart.so.12".to_string());
        paths.push("libcudart.so".to_string());

        let mut last_error = String::new();
        for path in paths {
            let c_path =
                CString::new(path.clone()).map_err(|e| CxlTmatmulError::Io(e.to_string()))?;
            let handle =
                unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if handle.is_null() {
                last_error = unsafe {
                    let err = libc::dlerror();
                    if err.is_null() {
                        format!("dlopen {path} failed")
                    } else {
                        format!(
                            "dlopen {path}: {}",
                            std::ffi::CStr::from_ptr(err).to_string_lossy()
                        )
                    }
                };
                continue;
            }

            unsafe {
                return Ok(Self {
                    handle,
                    cuda_set_device: Self::required_symbol(handle, "cudaSetDevice")?,
                    cuda_host_register: Self::required_symbol(handle, "cudaHostRegister")?,
                    cuda_host_get_device_pointer: Self::required_symbol(
                        handle,
                        "cudaHostGetDevicePointer",
                    )?,
                    cuda_memcpy_async: Self::required_symbol(handle, "cudaMemcpyAsync")?,
                    cuda_stream_synchronize: Self::required_symbol(
                        handle,
                        "cudaStreamSynchronize",
                    )?,
                    cuda_host_unregister: Self::required_symbol(handle, "cudaHostUnregister")?,
                    cuda_ipc_open_mem_handle: Self::optional_symbol(handle, "cudaIpcOpenMemHandle"),
                    cuda_ipc_close_mem_handle: Self::optional_symbol(
                        handle,
                        "cudaIpcCloseMemHandle",
                    ),
                });
            }
        }

        Err(CxlTmatmulError::Device(format!(
            "cuda_dax matrix staging could not load libcudart: {last_error}"
        )))
    }

    unsafe fn required_symbol<T>(
        handle: *mut libc::c_void,
        name: &str,
    ) -> Result<T, CxlTmatmulError> {
        let c_name = CString::new(name).map_err(|e| CxlTmatmulError::Io(e.to_string()))?;
        let ptr = libc::dlsym(handle, c_name.as_ptr());
        if ptr.is_null() {
            return Err(CxlTmatmulError::Device(format!(
                "cuda_dax matrix staging missing libcudart symbol {name}"
            )));
        }
        Ok(std::mem::transmute_copy(&ptr))
    }

    unsafe fn optional_symbol<T>(handle: *mut libc::c_void, name: &str) -> Option<T> {
        let c_name = CString::new(name).ok()?;
        let ptr = libc::dlsym(handle, c_name.as_ptr());
        if ptr.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&ptr))
        }
    }

    unsafe fn check_cuda(&self, call: &str, rc: libc::c_int) -> Result<(), CxlTmatmulError> {
        if rc == 0 {
            Ok(())
        } else {
            Err(CxlTmatmulError::Device(format!(
                "{call} failed: cudaError={rc}"
            )))
        }
    }

    unsafe fn copy_device_to_dax(
        &self,
        stage: &CudaDaxMatrixStage,
        dax_host_base: *mut u8,
        map_len: usize,
        label: &str,
    ) -> Result<(), CxlTmatmulError> {
        self.check_cuda("cudaSetDevice", (self.cuda_set_device)(stage.gpu))?;
        self.check_cuda(
            "cudaHostRegister",
            (self.cuda_host_register)(dax_host_base.cast(), map_len, CUDA_HOST_REGISTER_MAPPED),
        )?;

        let mut ipc_opened: Option<*mut libc::c_void> = None;
        let result = (|| {
            let mut cxl_dev_base: *mut libc::c_void = std::ptr::null_mut();
            self.check_cuda(
                "cudaHostGetDevicePointer",
                (self.cuda_host_get_device_pointer)(&mut cxl_dev_base, dax_host_base.cast(), 0),
            )?;
            if cxl_dev_base.is_null() {
                return Err(CxlTmatmulError::Device(
                    "cudaHostGetDevicePointer returned null for DAX mapping".to_string(),
                ));
            }

            let src = match &stage.source {
                CudaDaxSource::DevicePtr(ptr) => *ptr as *mut libc::c_void,
                CudaDaxSource::IpcHandle { bytes, .. } => {
                    let Some(open) = self.cuda_ipc_open_mem_handle else {
                        return Err(CxlTmatmulError::Device(
                            "cuda_dax IPC source requested but cudaIpcOpenMemHandle is unavailable"
                                .to_string(),
                        ));
                    };
                    let mut dev_ptr: *mut libc::c_void = std::ptr::null_mut();
                    self.check_cuda(
                        "cudaIpcOpenMemHandle",
                        open(
                            &mut dev_ptr,
                            bytes.as_ptr() as *mut libc::c_void,
                            CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
                        ),
                    )?;
                    if dev_ptr.is_null() {
                        return Err(CxlTmatmulError::Device(
                            "cudaIpcOpenMemHandle returned null device pointer".to_string(),
                        ));
                    }
                    ipc_opened = Some(dev_ptr);
                    dev_ptr
                }
            };
            if src.is_null() {
                return Err(CxlTmatmulError::Device(
                    "cuda_dax matrix source pointer is null".to_string(),
                ));
            }

            let cxl_offset =
                usize::try_from(stage.cxl_offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
            let dst = (cxl_dev_base as *mut u8)
                .add(cxl_offset)
                .cast::<libc::c_void>();
            let stream = stage.stream as *mut libc::c_void;
            self.check_cuda(
                "cudaMemcpyAsync",
                (self.cuda_memcpy_async)(
                    dst,
                    src as *const libc::c_void,
                    stage.bytes,
                    CUDA_MEMCPY_DEFAULT,
                    stream,
                ),
            )?;
            self.check_cuda(
                "cudaStreamSynchronize",
                (self.cuda_stream_synchronize)(stream),
            )?;
            eprintln!(
                "[CXL TMatmul] cuda_dax staged {} bytes={} gpu={} cxl_offset=0x{:x} dax={} stream=0x{:x}",
                label, stage.bytes, stage.gpu, stage.cxl_offset, stage.dax_path, stage.stream
            );
            Ok(())
        })();

        if let Some(dev_ptr) = ipc_opened {
            if let Some(close) = self.cuda_ipc_close_mem_handle {
                let _ = close(dev_ptr);
            }
        }
        let _ = (self.cuda_host_unregister)(dax_host_base.cast());
        result
    }

    unsafe fn copy_dax_to_device(
        &self,
        device_ptr: usize,
        bytes: usize,
        cxl_offset: u64,
        dax_host_base: *mut u8,
        map_len: usize,
        label: &str,
    ) -> Result<(), CxlTmatmulError> {
        if device_ptr == 0 {
            return Err(CxlTmatmulError::Device(format!(
                "cuda_dax {label} destination pointer is null"
            )));
        }
        let gpu = cuda_dax_gpu()?;
        let stream = cuda_dax_stream()?;
        self.check_cuda("cudaSetDevice", (self.cuda_set_device)(gpu))?;
        self.check_cuda(
            "cudaHostRegister",
            (self.cuda_host_register)(dax_host_base.cast(), map_len, CUDA_HOST_REGISTER_MAPPED),
        )?;

        let result = (|| {
            let mut cxl_dev_base: *mut libc::c_void = std::ptr::null_mut();
            self.check_cuda(
                "cudaHostGetDevicePointer",
                (self.cuda_host_get_device_pointer)(&mut cxl_dev_base, dax_host_base.cast(), 0),
            )?;
            if cxl_dev_base.is_null() {
                return Err(CxlTmatmulError::Device(
                    "cudaHostGetDevicePointer returned null for DAX mapping".to_string(),
                ));
            }

            let cxl_offset =
                usize::try_from(cxl_offset).map_err(|_| CxlTmatmulError::SizeOverflow)?;
            let src = (cxl_dev_base as *const u8)
                .add(cxl_offset)
                .cast::<libc::c_void>();
            let dst = device_ptr as *mut libc::c_void;
            let stream_ptr = stream as *mut libc::c_void;
            self.check_cuda(
                "cudaMemcpyAsync",
                (self.cuda_memcpy_async)(
                    dst,
                    src as *const libc::c_void,
                    bytes,
                    CUDA_MEMCPY_DEFAULT,
                    stream_ptr,
                ),
            )?;
            self.check_cuda(
                "cudaStreamSynchronize",
                (self.cuda_stream_synchronize)(stream_ptr),
            )?;
            eprintln!(
                "[CXL TMatmul] cuda_dax copied {} bytes={} gpu={} cxl_offset=0x{:x} stream=0x{:x}",
                label, bytes, gpu, cxl_offset, stream
            );
            Ok(())
        })();

        let _ = (self.cuda_host_unregister)(dax_host_base.cast());
        result
    }

    unsafe fn copy_host_to_device(
        &self,
        device_ptr: usize,
        bytes: &[u8],
        label: &str,
    ) -> Result<(), CxlTmatmulError> {
        if device_ptr == 0 {
            return Err(CxlTmatmulError::Device(format!(
                "cuda_dax {label} destination pointer is null"
            )));
        }
        if bytes.is_empty() {
            return Err(CxlTmatmulError::Device(format!(
                "cuda_dax {label} host source is empty"
            )));
        }
        let gpu = cuda_dax_gpu()?;
        let stream = cuda_dax_stream()?;
        self.check_cuda("cudaSetDevice", (self.cuda_set_device)(gpu))?;
        let stream_ptr = stream as *mut libc::c_void;
        self.check_cuda(
            "cudaMemcpyAsync",
            (self.cuda_memcpy_async)(
                device_ptr as *mut libc::c_void,
                bytes.as_ptr().cast::<libc::c_void>(),
                bytes.len(),
                CUDA_MEMCPY_DEFAULT,
                stream_ptr,
            ),
        )?;
        self.check_cuda(
            "cudaStreamSynchronize",
            (self.cuda_stream_synchronize)(stream_ptr),
        )?;
        eprintln!(
            "[CXL TMatmul] cuda_dax copied {} host bytes={} gpu={} stream=0x{:x}",
            label,
            bytes.len(),
            gpu,
            stream
        );
        Ok(())
    }
}

#[cfg(unix)]
impl Drop for CudaRuntime {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle);
        }
    }
}

#[cfg(unix)]
impl Drop for StagingMap {
    fn drop(&mut self) {
        unsafe {
            match self {
                Self::Mmap {
                    map_ptr, map_len, ..
                } => {
                    libc::munmap((*map_ptr).cast(), *map_len);
                }
                Self::CsrProbe { bar, map_len, .. } => {
                    libc::munmap((*bar).cast(), *map_len);
                }
            }
        }
    }
}

#[cfg(unix)]
fn require_csr_word_aligned(offset: u64, len: usize, op: &str) -> Result<(), CxlTmatmulError> {
    if (offset & 0x3) != 0 || (len & 0x3) != 0 {
        return Err(CxlTmatmulError::Device(format!(
            "csr_probe {op} requires 4-byte aligned offset/len: offset=0x{offset:x} len=0x{len:x}"
        )));
    }
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
fn read_dax_align(path: &str) -> Result<usize, CxlTmatmulError> {
    let base = path.rsplit('/').next().unwrap_or(path);
    let sysfs = format!("/sys/bus/dax/devices/{base}/align");
    let mut text = String::new();
    File::open(&sysfs)
        .and_then(|mut f| f.read_to_string(&mut text))
        .map_err(|e| CxlTmatmulError::Io(format!("read {sysfs}: {e}")))?;
    let value = parse_u64_text(text.trim()).ok_or_else(|| {
        CxlTmatmulError::Io(format!("invalid dax align in {sysfs}: {:?}", text.trim()))
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
fn cxl_tmatmul_pci_addr(device_path: &str) -> String {
    std::env::var("HETGPU_CXL_TMATMUL_PCI_ADDR")
        .or_else(|_| std::env::var("HETGPU_TMATMUL_PCI_ADDR"))
        .or_else(|_| std::env::var("TMATMUL_PCI_ADDR"))
        .ok()
        .or_else(|| pci_addr_from_tmatmul_devnode(device_path))
        .unwrap_or_else(|| DEFAULT_PCI_ADDR.to_string())
}

#[cfg(unix)]
fn pci_addr_from_tmatmul_devnode(dev_path: &str) -> Option<String> {
    let base = dev_path.rsplit('/').next().unwrap_or(dev_path);
    let suffix = base.strip_prefix("cxl_tmatmul")?;
    if suffix.len() != 5 || !suffix.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!(
        "0000:{}{}:{}{}.{}",
        &suffix[0..1],
        &suffix[1..2],
        &suffix[2..3],
        &suffix[3..4],
        &suffix[4..5]
    ))
}

#[cfg(unix)]
fn cxl_tmatmul_bar_index() -> Result<u32, CxlTmatmulError> {
    env_u32_any(
        &[
            "HETGPU_CXL_TMATMUL_BAR",
            "HETGPU_TMATMUL_BAR",
            "TMATMUL_BAR",
        ],
        DEFAULT_BAR_INDEX,
    )
}

#[cfg(unix)]
fn cxl_tmatmul_csr_base() -> Result<u32, CxlTmatmulError> {
    env_u32_any(
        &[
            "HETGPU_CXL_TMATMUL_CSR_BASE",
            "HETGPU_TMATMUL_CSR_BASE",
            "TMATMUL_CSR_BASE",
        ],
        DEFAULT_CSR_BASE,
    )
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

fn env_u32_any(keys: &[&str], fallback: u32) -> Result<u32, CxlTmatmulError> {
    for key in keys {
        match std::env::var(key) {
            Ok(value) => {
                let parsed = parse_u64_text(&value).ok_or_else(|| {
                    CxlTmatmulError::Io(format!("invalid {key}={value:?}, expected integer"))
                })?;
                return u32::try_from(parsed).map_err(|_| CxlTmatmulError::SizeOverflow);
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(e) => {
                return Err(CxlTmatmulError::Io(format!("read {key}: {e}")));
            }
        }
    }
    Ok(fallback)
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
    align_len(len, page_size())
}

fn align_len(len: usize, align: usize) -> Result<usize, CxlTmatmulError> {
    if align == 0 {
        return Err(CxlTmatmulError::SizeOverflow);
    }
    let rem = len % align;
    if rem == 0 {
        return Ok(len);
    }
    len.checked_add(align - rem)
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
fn cuda_dax_source_from_env(default_device_ptr: usize) -> Result<CudaDaxSource, CxlTmatmulError> {
    if let Some(hex) = cuda_ipc_handle_hex_from_env() {
        return Ok(CudaDaxSource::IpcHandle {
            bytes: parse_cuda_ipc_handle_hex(&hex)?,
            hex: normalize_cuda_ipc_handle_hex(&hex)?,
        });
    }
    if default_device_ptr == 0 {
        return Err(CxlTmatmulError::Device(
            "cuda_dax matrix staging requires cuda_device_ptr or cuda_ipc_handle".to_string(),
        ));
    }
    Ok(CudaDaxSource::DevicePtr(default_device_ptr))
}

#[cfg(unix)]
fn cuda_ipc_handle_hex_from_env() -> Option<String> {
    [
        "HETGPU_TMATMUL_MATRIX_CUDA_IPC_HANDLE",
        "HETGPU_CXL_TMATMUL_MATRIX_CUDA_IPC_HANDLE",
    ]
    .iter()
    .find_map(|key| std::env::var(key).ok())
}

#[cfg(unix)]
fn normalize_cuda_ipc_handle_hex(value: &str) -> Result<String, CxlTmatmulError> {
    let cleaned = value
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X")
        .chars()
        .filter(|c| !matches!(c, ':' | '-' | '_' | ' ' | '\n' | '\t' | '\r'))
        .collect::<String>()
        .to_ascii_lowercase();
    if cleaned.len() != CUDA_IPC_HANDLE_BYTES * 2 || !cleaned.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(CxlTmatmulError::Io(format!(
            "invalid CUDA IPC handle length {}, expected {} hex chars",
            cleaned.len(),
            CUDA_IPC_HANDLE_BYTES * 2
        )));
    }
    Ok(cleaned)
}

#[cfg(unix)]
fn parse_cuda_ipc_handle_hex(value: &str) -> Result<[u8; CUDA_IPC_HANDLE_BYTES], CxlTmatmulError> {
    let normalized = normalize_cuda_ipc_handle_hex(value)?;
    let mut out = [0u8; CUDA_IPC_HANDLE_BYTES];
    for (i, slot) in out.iter_mut().enumerate() {
        let byte = &normalized[i * 2..i * 2 + 2];
        *slot = u8::from_str_radix(byte, 16)
            .map_err(|e| CxlTmatmulError::Io(format!("invalid CUDA IPC handle byte {i}: {e}")))?;
    }
    Ok(out)
}

#[cfg(unix)]
fn cuda_dax_gpu() -> Result<i32, CxlTmatmulError> {
    let value = env_u64_any(&[
        "HETGPU_TMATMUL_CUDA_GPU",
        "HETGPU_TMATMUL_MATRIX_GPU",
        "HETGPU_CXL_TMATMUL_CUDA_GPU",
    ])?
    .unwrap_or(0);
    i32::try_from(value).map_err(|_| CxlTmatmulError::SizeOverflow)
}

#[cfg(unix)]
fn cuda_dax_stream() -> Result<usize, CxlTmatmulError> {
    env_u64_any(&[
        "HETGPU_TMATMUL_CUDA_STREAM",
        "HETGPU_TMATMUL_MATRIX_CUDA_STREAM",
    ])?
    .map(|value| usize::try_from(value).map_err(|_| CxlTmatmulError::SizeOverflow))
    .transpose()
    .map(|value| value.unwrap_or(0))
}

#[cfg(unix)]
fn cuda_dax_matrix_stage(
    matrix_ptr: usize,
    bytes: usize,
    cxl_offset: u64,
) -> Result<CudaDaxMatrixStage, CxlTmatmulError> {
    if bytes == 0 {
        return Err(CxlTmatmulError::Device(
            "cuda_dax matrix staging requires non-zero bytes".to_string(),
        ));
    }
    Ok(CudaDaxMatrixStage {
        source: cuda_dax_source_from_env(matrix_ptr)?,
        bytes,
        dax_path: cxl_tmatmul_dax_path(),
        cxl_offset,
        gpu: cuda_dax_gpu()?,
        stream: cuda_dax_stream()?,
    })
}

#[cfg(unix)]
fn cuda_dax_device_stage(
    device_ptr: usize,
    bytes: usize,
    cxl_offset: u64,
) -> Result<CudaDaxMatrixStage, CxlTmatmulError> {
    if bytes == 0 {
        return Err(CxlTmatmulError::Device(
            "cuda_dax device staging requires non-zero bytes".to_string(),
        ));
    }
    if device_ptr == 0 {
        return Err(CxlTmatmulError::Device(
            "cuda_dax device staging source pointer is null".to_string(),
        ));
    }
    Ok(CudaDaxMatrixStage {
        source: CudaDaxSource::DevicePtr(device_ptr),
        bytes,
        dax_path: cxl_tmatmul_dax_path(),
        cxl_offset,
        gpu: cuda_dax_gpu()?,
        stream: cuda_dax_stream()?,
    })
}

#[cfg(unix)]
fn cxl_f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut exponent = -14i32;
            let mut significand = mant;
            while (significand & 0x0400) == 0 {
                significand <<= 1;
                exponent -= 1;
            }
            significand &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (significand << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | (((exp as u32) + 112) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

#[cfg(unix)]
fn cxl_tmatmul_bar_run_enabled() -> bool {
    env_flag("HETGPU_CXL_TMATMUL_BAR_RUN") || env_flag("HETGPU_TMATMUL_BAR_RUN")
}

#[cfg(unix)]
fn cxl_tmatmul_bar_fallback_disabled() -> bool {
    env_flag("HETGPU_CXL_TMATMUL_DISABLE_BAR_FALLBACK")
        || env_flag("HETGPU_TMATMUL_DISABLE_BAR_FALLBACK")
}

#[cfg(unix)]
fn bar_run_status(
    timeout_ms: u32,
    dma_status: u32,
    stall_status: u32,
    instr_count: u32,
    dim_d: u32,
) -> CxlTmatmulRunStatus {
    let mut result_flags = 0;
    if stall_status != 0 {
        result_flags |= CXL_TYPE2_TMATMUL_RESULT_STALLED;
    }
    if dma_status == TMATMUL_DMA_ERROR_STATUS {
        result_flags |= CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR;
    }
    CxlTmatmulRunStatus {
        timeout_ms,
        dma_status,
        stall_status,
        instr_count,
        dim_d,
        result_flags,
    }
}

#[cfg(unix)]
fn csr_probe_prelaunch_reset_sequence() -> [(usize, u32); 3] {
    [
        (CSR_INST_STALL_CLEAR, 1),
        (CSR_INST_RST_TRIGGER, 1),
        (CSR_INST_STALL_CLEAR, 1),
    ]
}

#[cfg(unix)]
fn cxl_tmatmul_run_retries() -> usize {
    for key in [
        "HETGPU_CXL_TMATMUL_RUN_RETRIES",
        "HETGPU_TMATMUL_RUN_RETRIES",
        "TMATMUL_RUN_RETRIES",
    ] {
        if let Ok(value) = std::env::var(key) {
            if let Some(parsed) =
                parse_u64_text(&value).and_then(|value| usize::try_from(value).ok())
            {
                return parsed;
            }
        }
    }
    1
}

#[cfg(unix)]
fn should_retry_csr_run_timeout(
    err: &std::io::Error,
    status: &CxlTmatmulRunStatus,
    run_attempt: usize,
    max_retries: usize,
    expected_dim: u32,
) -> bool {
    run_attempt < max_retries
        && err.raw_os_error() == Some(libc::ETIMEDOUT)
        && status.stall_status == 0
        && status.dim_d == expected_dim
        && (status.result_flags & CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR) == 0
        && (status.result_flags & CXL_TYPE2_TMATMUL_RESULT_STALLED) == 0
}

#[cfg(unix)]
fn should_fallback_to_bar_run(
    err: &std::io::Error,
    status: &CxlTmatmulRunStatus,
    expected_dim: u32,
) -> bool {
    matches!(
        err.raw_os_error(),
        Some(libc::ENOTTY) | Some(libc::ETIMEDOUT)
    ) && status.stall_status == 0
        && status.dim_d == expected_dim
        && (status.result_flags & CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR) == 0
        && (status.result_flags & CXL_TYPE2_TMATMUL_RESULT_STALLED) == 0
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
    fn encodes_stall_for_live_rtl_bit_order() {
        let stall = encode_instr(0b101, 0, 0, 0, 0, 0, 0, 0, 0);
        let words: Vec<u32> = stall
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        assert_eq!(words, vec![0, 0, 0x0050_0000, 0]);
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
    fn assembler_accepts_tmatmul_go_nvint8_alias_with_explicit_stage_width() {
        let matrix_offset = 0x0040_0000;
        let asm = "
            ldv v0,PARAM_1
            tmatmul_import v0
            tmatmul_go_nvint8 PARAM_0,4
            tmatmul_export v1
            sv v1,PARAM_2
            stall
        ";
        let labels = std::collections::HashMap::from([
            ("PARAM_0".to_string(), matrix_offset),
            ("PARAM_1".to_string(), TMATMUL_DPA_INPUT),
            ("PARAM_2".to_string(), TMATMUL_DPA_OUTPUT),
        ]);

        let program = assemble_tmatmul_program(asm, &labels).unwrap();

        assert_eq!(&program[32..40], &matrix_offset.to_le_bytes());
    }

    #[cfg(unix)]
    #[test]
    fn matrix_stage_mode_parser_accepts_cuda_dax() {
        assert_eq!(
            parse_matrix_stage_mode(Some("cuda_dax")).unwrap(),
            MatrixStageMode::CudaDax
        );
        assert_eq!(
            parse_matrix_stage_mode(Some("host")).unwrap(),
            MatrixStageMode::Host
        );
        assert_eq!(
            parse_matrix_stage_mode(None).unwrap(),
            MatrixStageMode::Host
        );
        assert!(parse_matrix_stage_mode(Some("bogus")).is_err());
    }

    #[test]
    fn io_stage_mode_parser_accepts_cuda_dax() {
        assert_eq!(
            parse_io_stage_mode(Some("cuda_dax")).unwrap(),
            IoStageMode::CudaDax
        );
        assert_eq!(
            parse_io_stage_mode(Some("host")).unwrap(),
            IoStageMode::Host
        );
        assert_eq!(parse_io_stage_mode(None).unwrap(), IoStageMode::Host);
        assert!(parse_io_stage_mode(Some("bogus")).is_err());
    }

    #[test]
    fn output_stage_dtype_parser_accepts_f32() {
        assert_eq!(
            parse_output_stage_dtype(Some("f32")).unwrap(),
            OutputStageDtype::F32
        );
        assert_eq!(
            parse_output_stage_dtype(Some("half")).unwrap(),
            OutputStageDtype::F16
        );
        assert_eq!(
            parse_output_stage_dtype(None).unwrap(),
            OutputStageDtype::F16
        );
        assert!(parse_output_stage_dtype(Some("int8")).is_err());
    }

    #[test]
    fn cxl_f16_to_f32_converts_common_values() {
        assert_eq!(cxl_f16_to_f32(0x0000), 0.0);
        assert_eq!(cxl_f16_to_f32(0x3c00), 1.0);
        assert_eq!(cxl_f16_to_f32(0xc000), -2.0);
        assert!(cxl_f16_to_f32(0x7c00).is_infinite());
        assert!(cxl_f16_to_f32(0x0001) > 0.0);
    }

    #[test]
    fn fixed_layout_allows_nondefault_matrix_offset_when_ranges_do_not_overlap() {
        validate_fixed_layout_at_offsets(0x0040_0000, 0x1000, 0x1000, TMATMUL_PROGRAM_BYTES)
            .unwrap();

        let err = validate_fixed_layout_at_offsets(
            TMATMUL_DPA_INPUT - 0x100,
            0x200,
            0x1000,
            TMATMUL_PROGRAM_BYTES,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("matrix/input"),
            "unexpected error: {err}"
        );
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
    fn csr_probe_prelaunch_reset_sequence_matches_bridge() {
        assert_eq!(
            csr_probe_prelaunch_reset_sequence(),
            [(0x104, 1), (0x108, 1), (0x104, 1)]
        );
    }

    #[cfg(unix)]
    #[test]
    fn run_csr_only_ioctl_requests_cover_known_abis() {
        let requests = cxl_type2_tmatmul_run_csr_only_ioctl_requests();

        assert_eq!(requests[0], 0xC040CE02 as libc::c_ulong);
        assert_eq!(requests[1], 0xC040CE03 as libc::c_ulong);
    }

    #[cfg(unix)]
    #[test]
    fn csr_run_timeout_without_progress_is_retryable_once() {
        let err = std::io::Error::from_raw_os_error(libc::ETIMEDOUT);
        let status = CxlTmatmulRunStatus {
            timeout_ms: DEFAULT_TIMEOUT_MS,
            dma_status: 1,
            stall_status: 0,
            instr_count: 0,
            dim_d: 2048,
            result_flags: 0,
        };

        assert!(should_retry_csr_run_timeout(&err, &status, 0, 1, 2048));
        assert!(!should_retry_csr_run_timeout(&err, &status, 1, 1, 2048));
    }

    #[cfg(unix)]
    #[test]
    fn csr_run_retry_policy_accepts_timeout_before_stall_without_dma_error() {
        let err = std::io::Error::from_raw_os_error(libc::ETIMEDOUT);
        let base = CxlTmatmulRunStatus {
            timeout_ms: DEFAULT_TIMEOUT_MS,
            dma_status: 1,
            stall_status: 0,
            instr_count: 0,
            dim_d: 2048,
            result_flags: 0,
        };

        let mut progressed = base.clone();
        progressed.instr_count = 5;
        assert!(should_retry_csr_run_timeout(&err, &progressed, 0, 1, 2048));

        let mut stalled = base.clone();
        stalled.stall_status = 1;
        assert!(!should_retry_csr_run_timeout(&err, &stalled, 0, 1, 2048));

        let mut dma_error = base.clone();
        dma_error.result_flags = CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR;
        assert!(!should_retry_csr_run_timeout(&err, &dma_error, 0, 1, 2048));

        let mut wrong_dim = base;
        wrong_dim.dim_d = 1024;
        assert!(!should_retry_csr_run_timeout(&err, &wrong_dim, 0, 1, 2048));
    }

    #[cfg(unix)]
    #[test]
    fn csr_run_errors_that_leave_no_stall_use_bar_fallback() {
        let base = CxlTmatmulRunStatus {
            timeout_ms: DEFAULT_TIMEOUT_MS,
            dma_status: 1,
            stall_status: 0,
            instr_count: 0,
            dim_d: 2048,
            result_flags: 0,
        };

        assert!(should_fallback_to_bar_run(
            &std::io::Error::from_raw_os_error(libc::ENOTTY),
            &base,
            2048
        ));
        assert!(should_fallback_to_bar_run(
            &std::io::Error::from_raw_os_error(libc::ETIMEDOUT),
            &base,
            2048
        ));

        let mut stalled = base.clone();
        stalled.result_flags = CXL_TYPE2_TMATMUL_RESULT_STALLED;
        assert!(!should_fallback_to_bar_run(
            &std::io::Error::from_raw_os_error(libc::ETIMEDOUT),
            &stalled,
            2048
        ));

        let mut dma_error = base;
        dma_error.result_flags = CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR;
        assert!(!should_fallback_to_bar_run(
            &std::io::Error::from_raw_os_error(libc::ETIMEDOUT),
            &dma_error,
            2048
        ));
    }

    #[cfg(unix)]
    #[test]
    fn bar_run_status_marks_stall_and_dma_error_flags() {
        let stalled = bar_run_status(DEFAULT_TIMEOUT_MS, 0, 2, 7, 2048);

        assert_eq!(stalled.stall_status, 2);
        assert_eq!(stalled.instr_count, 7);
        assert_eq!(stalled.result_flags, CXL_TYPE2_TMATMUL_RESULT_STALLED);

        let dma_error = bar_run_status(DEFAULT_TIMEOUT_MS, 0xff, 0, 0, 2048);
        assert_eq!(dma_error.result_flags, CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR);
    }

    #[cfg(unix)]
    #[test]
    fn staging_backend_parser_accepts_csr_probe_mode() {
        assert_eq!(parse_staging_backend(None).unwrap(), StagingBackend::Mmap);
        assert_eq!(
            parse_staging_backend(Some("csr_probe")).unwrap(),
            StagingBackend::CsrProbe
        );
        assert_eq!(
            parse_staging_backend(Some("csr")).unwrap(),
            StagingBackend::CsrProbe
        );
        assert!(parse_staging_backend(Some("bogus")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn tmatmul_devnode_maps_to_pci_bdf() {
        assert_eq!(
            pci_addr_from_tmatmul_devnode("/dev/cxl_tmatmul3b000").as_deref(),
            Some("0000:3b:00.0")
        );
        assert_eq!(
            pci_addr_from_tmatmul_devnode("cxl_tmatmulaf001").as_deref(),
            Some("0000:af:00.1")
        );
        assert!(pci_addr_from_tmatmul_devnode("/dev/not_tmatmul").is_none());
    }

    #[test]
    fn staging_map_len_is_page_aligned() {
        let page = page_size();

        assert_eq!(page_align_len(1).unwrap(), page);
        assert_eq!(page_align_len(page).unwrap(), page);
        assert_eq!(page_align_len(page + 1).unwrap(), page * 2);
    }

    #[test]
    fn staging_map_len_can_use_dax_alignment() {
        assert_eq!(align_len(0x301000, 0x200000).unwrap(), 0x400000);
        assert_eq!(align_len(0x400000, 0x200000).unwrap(), 0x400000);
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
            MatrixStage::Host(&matrix),
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
    fn hardware_csr_probe_stages_and_runs_stall_program_when_requested() {
        if !env_flag("HETGPU_CXL_TMATMUL_HW_STALL_PROBE") {
            return;
        }

        let device_path = cxl_tmatmul_device_path();
        let device = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&device_path)
            .unwrap_or_else(|e| panic!("open {device_path}: {e}"));
        let info = get_info(&device).unwrap();
        assert_eq!(info.version, CXL_TYPE2_TMATMUL_UAPI_VERSION);
        assert_ne!(info.dim_d, 0);

        let mut program = vec![0u8; TMATMUL_PROGRAM_BYTES];
        program[..16].copy_from_slice(&encode_instr(0b101, 0, 0, 0, 0, 0, 0, 0, 0));

        let mut staging =
            StagingMap::open_csr_probe((TMATMUL_DPA_PROGRAM as usize) + program.len()).unwrap();
        staging.stage_bytes(TMATMUL_DPA_PROGRAM, &program).unwrap();

        let mut readback = vec![0u8; program.len()];
        staging
            .read_bytes(TMATMUL_DPA_PROGRAM, &mut readback)
            .unwrap();
        assert_eq!(readback, program);

        staging.reset_instruction_engine_before_run().unwrap();
        let started = std::time::Instant::now();
        let status = staging
            .run_instance0_program(
                TMATMUL_DPA_PROGRAM,
                program.len() as u32,
                10_000,
                info.dim_d,
            )
            .unwrap();
        assert_ne!(status.result_flags & CXL_TYPE2_TMATMUL_RESULT_STALLED, 0);
        eprintln!(
            "tmatmul csr-probe stall smoke dim={} elapsed_us={} status={:?}",
            info.dim_d,
            started.elapsed().as_micros(),
            status
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
                MatrixStage::Host(&matrix),
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
