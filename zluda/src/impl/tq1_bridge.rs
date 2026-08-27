use super::tq1_tmatmul::{
    classify_tensor_name, ExpertRole, Tq1TensorRegistration, Tq1TensorSource,
};
use super::tq1_xrt::{self, Tq1MulMatIdOperation, Tq1XrtResult};
use cuda_types::cuda::{CUdeviceptr_v2, CUstream};
use serde_json::json;
use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub(crate) const HETGPU_TQ1_ABI_VERSION: u32 = 1;
pub(crate) const HETGPU_TQ1_NOT_HANDLED: i32 = 0;
pub(crate) const HETGPU_TQ1_HANDLED: i32 = 1;
pub(crate) const HETGPU_TQ1_ERROR: i32 = -1;
pub(crate) const GGML_TYPE_F32: u32 = 0;
pub(crate) const GGML_TYPE_I32: u32 = 26;
pub(crate) const GGML_TYPE_TQ1_0: u32 = 34;

const ROLE_GATE: u32 = 1;
const ROLE_UP: u32 = 2;
const ROLE_DOWN: u32 = 3;
const ROLE_GATE_UP: u32 = 4;

static EVIDENCE_LOCK: Mutex<()> = Mutex::new(());
static ELIGIBLE_OPERATIONS: AtomicU64 = AtomicU64::new(0);
static HANDLED_OPERATIONS: AtomicU64 = AtomicU64::new(0);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HetgpuTq1TensorV1 {
    pub abi_version: u32,
    pub ggml_type: u32,
    pub role: u32,
    pub file_index: u32,
    pub name: *const c_char,
    pub path: *const c_char,
    pub file_offset: u64,
    pub nbytes: u64,
    pub ne: [i64; 4],
    pub nb: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HetgpuTq1MulMatIdV1 {
    pub abi_version: u32,
    pub src0_type: u32,
    pub src1_type: u32,
    pub ids_type: u32,
    pub dst_type: u32,
    pub reserved: u32,
    pub src0_name: *const c_char,
    pub src1_device: *const c_void,
    pub ids_device: *const c_void,
    pub dst_device: *mut c_void,
    pub cuda_stream: *mut c_void,
    pub src0_ne: [i64; 4],
    pub src0_nb: [u64; 4],
    pub src1_ne: [i64; 4],
    pub src1_nb: [u64; 4],
    pub ids_ne: [i64; 4],
    pub ids_nb: [u64; 4],
    pub dst_ne: [i64; 4],
    pub dst_nb: [u64; 4],
}

pub(crate) trait CudaCopies {
    unsafe fn synchronize(&self, stream: *mut c_void) -> Result<(), String>;
    unsafe fn device_to_host(&self, dst: &mut [u8], src: *const c_void) -> Result<(), String>;
    unsafe fn host_to_device(&self, dst: *mut c_void, src: &[u8]) -> Result<(), String>;
}

struct NvidiaCudaCopies;

impl CudaCopies for NvidiaCudaCopies {
    unsafe fn synchronize(&self, stream: *mut c_void) -> Result<(), String> {
        let result = nvidia_runtime_sys::cuStreamSynchronize_ckpt(CUstream(stream.cast()));
        if result != 0 {
            return Err(format!("cuStreamSynchronize failed with code {result}"));
        }
        Ok(())
    }

    unsafe fn device_to_host(&self, dst: &mut [u8], src: *const c_void) -> Result<(), String> {
        let result = nvidia_runtime_sys::cuMemcpyDtoH_v2(
            dst.as_mut_ptr().cast(),
            CUdeviceptr_v2(src.cast_mut()),
            dst.len(),
        );
        if result != 0 {
            return Err(format!("cuMemcpyDtoH_v2 failed with code {result}"));
        }
        Ok(())
    }

    unsafe fn host_to_device(&self, dst: *mut c_void, src: &[u8]) -> Result<(), String> {
        let result = nvidia_runtime_sys::cuMemcpyHtoD_v2(
            CUdeviceptr_v2(dst),
            src.as_ptr().cast(),
            src.len(),
        );
        if result != 0 {
            return Err(format!("cuMemcpyHtoD_v2 failed with code {result}"));
        }
        Ok(())
    }
}

pub(crate) trait Tq1Executor {
    fn execute(&self, operation: &Tq1MulMatIdOperation) -> Result<Tq1XrtResult, String>;
}

struct PersistentXrtExecutor;

impl Tq1Executor for PersistentXrtExecutor {
    fn execute(&self, operation: &Tq1MulMatIdOperation) -> Result<Tq1XrtResult, String> {
        tq1_xrt::execute_mul_mat_id(operation)
    }
}

fn env_enabled(name: &str) -> Result<bool, String> {
    match std::env::var(name) {
        Ok(value) => match value.as_str() {
            "1" | "true" | "TRUE" | "on" => Ok(true),
            "0" | "false" | "FALSE" | "off" => Ok(false),
            _ => Err(format!("{name} must be a boolean value")),
        },
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(format!("read {name}: {error}")),
    }
}

fn evidence_path(strict: bool) -> Result<Option<PathBuf>, String> {
    match std::env::var("HETGPU_TQ1_EVIDENCE_LOG") {
        Ok(path) if !path.trim().is_empty() => Ok(Some(PathBuf::from(path.trim()))),
        Ok(_) | Err(std::env::VarError::NotPresent) if strict => {
            Err("HETGPU_TQ1_EVIDENCE_LOG is required in strict mode".to_string())
        }
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("read HETGPU_TQ1_EVIDENCE_LOG: {error}")),
    }
}

fn preflight_evidence(path: &Path) -> Result<(), String> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("open TQ1_0 evidence {}: {error}", path.display()))?;
    file.sync_data()
        .map_err(|error| format!("sync TQ1_0 evidence {}: {error}", path.display()))
}

fn append_evidence(path: &Path, record: &serde_json::Value) -> Result<(), String> {
    let _guard = EVIDENCE_LOCK
        .lock()
        .map_err(|_| "TQ1_0 evidence mutex poisoned".to_string())?;
    let mut bytes =
        serde_json::to_vec(record).map_err(|error| format!("serialize TQ1_0 evidence: {error}"))?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("open TQ1_0 evidence {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .map_err(|error| format!("write TQ1_0 evidence {}: {error}", path.display()))?;
    file.sync_data()
        .map_err(|error| format!("sync TQ1_0 evidence {}: {error}", path.display()))
}

fn positive_dimensions(values: [i64; 4], label: &str) -> Result<[usize; 4], String> {
    let mut result = [0usize; 4];
    for (index, value) in values.into_iter().enumerate() {
        if value <= 0 {
            return Err(format!("{label}.ne[{index}] must be positive"));
        }
        result[index] = usize::try_from(value)
            .map_err(|_| format!("{label}.ne[{index}] does not fit usize"))?;
    }
    Ok(result)
}

fn storage_bytes(ne: [usize; 4], nb: [u64; 4], element_bytes: usize) -> Result<usize, String> {
    if nb[0] != element_bytes as u64 {
        return Err(format!("innermost stride must equal {element_bytes}"));
    }
    for dimension in 1..4 {
        let preceding_extent = nb[dimension - 1]
            .checked_mul(ne[dimension - 1] as u64)
            .ok_or_else(|| "tensor stride extent overflow".to_string())?;
        if nb[dimension] < preceding_extent {
            return Err(format!(
                "tensor stride nb[{dimension}] overlaps the preceding dimension"
            ));
        }
    }
    let end = (0..4).try_fold(element_bytes as u64, |extent, dimension| {
        (ne[dimension] as u64)
            .checked_sub(1)
            .and_then(|count| count.checked_mul(nb[dimension]))
            .and_then(|offset| extent.checked_add(offset))
            .ok_or_else(|| "tensor storage extent overflow".to_string())
    })?;
    usize::try_from(end).map_err(|_| "tensor storage extent does not fit usize".to_string())
}

fn byte_offset(indices: [usize; 4], nb: [u64; 4], bytes: usize) -> Result<usize, String> {
    let offset = indices
        .into_iter()
        .zip(nb)
        .try_fold(0u64, |offset, (index, stride)| {
            (index as u64)
                .checked_mul(stride)
                .and_then(|value| offset.checked_add(value))
                .ok_or_else(|| "tensor byte offset overflow".to_string())
        })?;
    let end = offset
        .checked_add(bytes as u64)
        .ok_or_else(|| "tensor byte range overflow".to_string())?;
    usize::try_from(end - bytes as u64)
        .map_err(|_| "tensor byte offset does not fit usize".to_string())
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "f32 byte range overflow".to_string())?;
    let value = f32::from_le_bytes(
        bytes
            .get(offset..end)
            .ok_or_else(|| "f32 read exceeds copied source".to_string())?
            .try_into()
            .expect("four-byte f32 slice"),
    );
    if !value.is_finite() {
        return Err("TQ1_0 activation contains a non-finite value".to_string());
    }
    Ok(value)
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "i32 byte range overflow".to_string())?;
    Ok(i32::from_le_bytes(
        bytes
            .get(offset..end)
            .ok_or_else(|| "i32 read exceeds copied IDs".to_string())?
            .try_into()
            .expect("four-byte i32 slice"),
    ))
}

fn qualified_name(operation: &HetgpuTq1MulMatIdV1) -> Result<Option<String>, String> {
    if operation.src0_name.is_null() {
        return Err("TQ1_0 src0_name is null".to_string());
    }
    let name = unsafe { CStr::from_ptr(operation.src0_name) }
        .to_str()
        .map_err(|_| "TQ1_0 src0_name is not valid UTF-8".to_string())?;
    match classify_tensor_name(name.as_bytes()) {
        Ok(_) => Ok(Some(name.to_string())),
        Err(_) => Ok(None),
    }
}

fn build_operation(
    operation: &HetgpuTq1MulMatIdV1,
    copies: &impl CudaCopies,
) -> Result<Option<Tq1MulMatIdOperation>, String> {
    let Some(name) = qualified_name(operation)? else {
        return Ok(None);
    };
    if operation.abi_version != HETGPU_TQ1_ABI_VERSION || operation.reserved != 0 {
        return Err("TQ1_0 mul_mat_id ABI version or reserved field is invalid".to_string());
    }
    if operation.src0_type != GGML_TYPE_TQ1_0
        || operation.src1_type != GGML_TYPE_F32
        || operation.ids_type != GGML_TYPE_I32
        || operation.dst_type != GGML_TYPE_F32
    {
        return Err("TQ1_0 mul_mat_id has unsupported GGML types".to_string());
    }
    if operation.src1_device.is_null()
        || operation.ids_device.is_null()
        || operation.dst_device.is_null()
    {
        return Err("TQ1_0 mul_mat_id has a null data pointer".to_string());
    }
    let source = Tq1TensorSource::lookup(&name)?
        .ok_or_else(|| format!("TQ1_0 tensor {name} was not registered"))?;
    let src0_ne = positive_dimensions(operation.src0_ne, "src0")?;
    let src1_ne = positive_dimensions(operation.src1_ne, "src1")?;
    let ids_ne = positive_dimensions(operation.ids_ne, "ids")?;
    let dst_ne = positive_dimensions(operation.dst_ne, "dst")?;
    let registered_ne = source.identity.ne.map(|value| value as usize);
    if src0_ne != registered_ne || operation.src0_nb != source.identity.nb {
        return Err("TQ1_0 src0 shape or stride differs from its registration".to_string());
    }
    if src0_ne[0] != src1_ne[0]
        || ids_ne[1] != src1_ne[2]
        || ids_ne[0] % src1_ne[1] != 0
        || dst_ne != [src0_ne[1], ids_ne[0], ids_ne[1], 1]
        || src0_ne[3] != 1
        || src1_ne[3] != 1
        || ids_ne[2] != 1
        || ids_ne[3] != 1
    {
        return Err("TQ1_0 mul_mat_id shape contract is invalid".to_string());
    }
    let src1_bytes = storage_bytes(src1_ne, operation.src1_nb, 4)?;
    let ids_bytes = storage_bytes(ids_ne, operation.ids_nb, 4)?;
    let _dst_bytes = storage_bytes(dst_ne, operation.dst_nb, 4)?;
    let mut source_bytes = vec![0u8; src1_bytes];
    let mut id_bytes = vec![0u8; ids_bytes];
    unsafe {
        copies.synchronize(operation.cuda_stream)?;
        copies.device_to_host(&mut source_bytes, operation.src1_device)?;
        copies.device_to_host(&mut id_bytes, operation.ids_device)?;
    }

    let logical_groups = ids_ne[0]
        .checked_mul(ids_ne[1])
        .ok_or_else(|| "TQ1_0 logical group count overflow".to_string())?;
    let activation_count = logical_groups
        .checked_mul(src1_ne[0])
        .ok_or_else(|| "TQ1_0 activation count overflow".to_string())?;
    let mut activations = Vec::with_capacity(activation_count);
    let mut expert_ids = Vec::with_capacity(logical_groups);
    for token in 0..ids_ne[1] {
        let mut token_experts = HashSet::new();
        for expert_slot in 0..ids_ne[0] {
            let id_offset = byte_offset([expert_slot, token, 0, 0], operation.ids_nb, 4)?;
            let expert = read_i32(&id_bytes, id_offset)?;
            let expert = usize::try_from(expert)
                .map_err(|_| "TQ1_0 expert ID must be nonnegative".to_string())?;
            if expert >= src0_ne[2] || !token_experts.insert(expert) {
                return Err("TQ1_0 expert IDs are out of bounds or duplicated".to_string());
            }
            expert_ids.push(expert);
            let channel = expert_slot % src1_ne[1];
            for k in 0..src1_ne[0] {
                let offset = byte_offset([k, channel, token, 0], operation.src1_nb, 4)?;
                activations.push(read_f32(&source_bytes, offset)?);
            }
        }
    }
    Ok(Some(Tq1MulMatIdOperation {
        tensor: source,
        activations,
        expert_ids,
        token_count: ids_ne[1],
        expert_slots: ids_ne[0],
    }))
}

fn install_output(
    abi: &HetgpuTq1MulMatIdV1,
    result: &Tq1XrtResult,
    copies: &impl CudaCopies,
) -> Result<(), String> {
    let dst_ne = positive_dimensions(abi.dst_ne, "dst")?;
    let dst_bytes = storage_bytes(dst_ne, abi.dst_nb, 4)?;
    let expected = dst_ne[0]
        .checked_mul(dst_ne[1])
        .and_then(|value| value.checked_mul(dst_ne[2]))
        .ok_or_else(|| "TQ1_0 output element count overflow".to_string())?;
    if result.outputs.len() != expected {
        return Err(format!(
            "TQ1_0 executor returned {} outputs, expected {expected}",
            result.outputs.len()
        ));
    }
    let mut destination = vec![0u8; dst_bytes];
    for token in 0..dst_ne[2] {
        for expert_slot in 0..dst_ne[1] {
            for row in 0..dst_ne[0] {
                let source_index = (token * dst_ne[1] + expert_slot) * dst_ne[0] + row;
                let offset = byte_offset([row, expert_slot, token, 0], abi.dst_nb, 4)?;
                destination[offset..offset + 4]
                    .copy_from_slice(&result.outputs[source_index].to_le_bytes());
            }
        }
    }
    unsafe {
        copies.host_to_device(abi.dst_device, &destination)?;
        copies.synchronize(abi.cuda_stream)?;
    }
    Ok(())
}

fn ids_hash(ids: &[usize]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    ids.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn try_mul_mat_id_with(
    abi: &HetgpuTq1MulMatIdV1,
    copies: &impl CudaCopies,
    executor: &impl Tq1Executor,
) -> i32 {
    let enabled = match env_enabled("HETGPU_QWEN_TQ1_XRT") {
        Ok(enabled) => enabled,
        Err(error) => {
            eprintln!("[hetgpu-tq1] {error}");
            return HETGPU_TQ1_ERROR;
        }
    };
    if !enabled {
        return HETGPU_TQ1_NOT_HANDLED;
    }
    let strict = match env_enabled("HETGPU_QWEN_TQ1_STRICT") {
        Ok(strict) => strict,
        Err(error) => {
            eprintln!("[hetgpu-tq1] {error}");
            return HETGPU_TQ1_ERROR;
        }
    };
    let evidence = match evidence_path(strict) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("[hetgpu-tq1] {error}");
            return HETGPU_TQ1_ERROR;
        }
    };
    if let Some(path) = &evidence {
        if let Err(error) = preflight_evidence(path) {
            eprintln!("[hetgpu-tq1] {error}");
            return HETGPU_TQ1_ERROR;
        }
    }

    let operation = match build_operation(abi, copies) {
        Ok(Some(operation)) => operation,
        Ok(None) => return HETGPU_TQ1_NOT_HANDLED,
        Err(error) => {
            if let Some(path) = &evidence {
                let _ = append_evidence(path, &json!({ "route": "error", "error": error }));
            }
            return HETGPU_TQ1_ERROR;
        }
    };
    let eligible = ELIGIBLE_OPERATIONS.fetch_add(1, Ordering::SeqCst) + 1;
    let hash = ids_hash(&operation.expert_ids);
    let result = match executor.execute(&operation) {
        Ok(result) => result,
        Err(error) => {
            if let Some(path) = &evidence {
                let _ = append_evidence(
                    path,
                    &json!({
                        "route": "error",
                        "eligible_operations": eligible,
                        "ids_hash": hash,
                        "error": error,
                    }),
                );
            }
            return HETGPU_TQ1_ERROR;
        }
    };
    if let Err(error) = install_output(abi, &result, copies) {
        if let Some(path) = &evidence {
            let _ = append_evidence(
                path,
                &json!({
                    "route": "error",
                    "eligible_operations": eligible,
                    "ids_hash": hash,
                    "error": error,
                }),
            );
        }
        return HETGPU_TQ1_ERROR;
    }
    let handled = HANDLED_OPERATIONS.fetch_add(1, Ordering::SeqCst) + 1;
    if let Some(path) = &evidence {
        if let Err(error) = append_evidence(
            path,
            &json!({
                "route": "handled",
                "tensor": operation.tensor.identity.name,
                "dimensions": {
                    "k": operation.tensor.identity.ne[0],
                    "rows": operation.tensor.identity.ne[1],
                    "tokens": operation.token_count,
                    "expert_slots": operation.expert_slots,
                },
                "ids_hash": hash,
                "eligible_operations": eligible,
                "handled_operations": handled,
                "evidence": result.evidence,
            }),
        ) {
            eprintln!("[hetgpu-tq1] {error}");
            return HETGPU_TQ1_ERROR;
        }
    }
    HETGPU_TQ1_HANDLED
}

pub(crate) unsafe fn register_tensor(tensor: *const HetgpuTq1TensorV1) -> i32 {
    let result = (|| -> Result<(), String> {
        let tensor = tensor
            .as_ref()
            .ok_or_else(|| "TQ1_0 registration pointer is null".to_string())?;
        if tensor.abi_version != HETGPU_TQ1_ABI_VERSION || tensor.ggml_type != GGML_TYPE_TQ1_0 {
            return Err("TQ1_0 registration ABI version or GGML type is invalid".to_string());
        }
        let role = match tensor.role {
            ROLE_GATE => ExpertRole::Gate,
            ROLE_UP => ExpertRole::Up,
            ROLE_DOWN => ExpertRole::Down,
            ROLE_GATE_UP => ExpertRole::GateUp,
            _ => return Err("TQ1_0 registration role is invalid".to_string()),
        };
        if tensor.name.is_null() || tensor.path.is_null() {
            return Err("TQ1_0 registration name or path is null".to_string());
        }
        let name = CStr::from_ptr(tensor.name)
            .to_str()
            .map_err(|_| "TQ1_0 registration name is not UTF-8".to_string())?;
        let path = CStr::from_ptr(tensor.path)
            .to_str()
            .map_err(|_| "TQ1_0 registration path is not UTF-8".to_string())?;
        let mut ne = [0u64; 4];
        for (index, value) in tensor.ne.into_iter().enumerate() {
            ne[index] = u64::try_from(value)
                .map_err(|_| format!("TQ1_0 registration ne[{index}] must be nonnegative"))?;
        }
        Tq1TensorSource::register(Tq1TensorRegistration {
            path: PathBuf::from(path),
            file_offset: tensor.file_offset,
            nbytes: tensor.nbytes,
            name: name.to_string(),
            ne,
            nb: tensor.nb,
            role,
        })?;
        Ok(())
    })();
    match result {
        Ok(()) => HETGPU_TQ1_HANDLED,
        Err(error) => {
            eprintln!("[hetgpu-tq1] registration failed: {error}");
            HETGPU_TQ1_ERROR
        }
    }
}

pub(crate) unsafe fn try_mul_mat_id(operation: *const HetgpuTq1MulMatIdV1) -> i32 {
    let Some(operation) = operation.as_ref() else {
        return HETGPU_TQ1_ERROR;
    };
    try_mul_mat_id_with(operation, &NvidiaCudaCopies, &PersistentXrtExecutor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::tq1_tmatmul::{ExpertRole, Tq1TensorRegistration, Tq1TensorSource};
    use crate::r#impl::tq1_xrt::{Tq1XrtEvidence, Tq1XrtResult};
    use std::ffi::{c_void, CString};
    use std::io::Write;
    use std::mem::{align_of, size_of};
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    fn registered_name() -> &'static CString {
        static NAME: OnceLock<CString> = OnceLock::new();
        NAME.get_or_init(|| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            let mut block = [0xff; 54];
            block[52..54].copy_from_slice(&0x3c00u16.to_le_bytes());
            for _ in 0..4 {
                file.write_all(&block).unwrap();
            }
            file.flush().unwrap();
            let name = CString::new("blk.77.ffn_down_exps.weight").unwrap();
            Tq1TensorSource::register(Tq1TensorRegistration {
                path: file.path().to_path_buf(),
                file_offset: 0,
                nbytes: 216,
                name: name.to_str().unwrap().to_string(),
                ne: [1024, 1, 1, 1],
                nb: [54, 216, 216, 216],
                role: ExpertRole::Down,
            })
            .unwrap();
            name
        })
    }

    fn operation_fixture() -> HetgpuTq1MulMatIdV1 {
        HetgpuTq1MulMatIdV1 {
            abi_version: HETGPU_TQ1_ABI_VERSION,
            src0_type: GGML_TYPE_TQ1_0,
            src1_type: GGML_TYPE_F32,
            ids_type: GGML_TYPE_I32,
            dst_type: GGML_TYPE_F32,
            reserved: 0,
            src0_name: registered_name().as_ptr(),
            src1_device: 1usize as *const c_void,
            ids_device: 2usize as *const c_void,
            dst_device: 3usize as *mut c_void,
            cuda_stream: std::ptr::null_mut(),
            src0_ne: [1024, 1, 1, 1],
            src0_nb: [54, 216, 216, 216],
            src1_ne: [1024, 1, 1, 1],
            src1_nb: [4, 4096, 4096, 4096],
            ids_ne: [1, 1, 1, 1],
            ids_nb: [4, 4, 4, 4],
            dst_ne: [1, 1, 1, 1],
            dst_nb: [4, 4, 4, 4],
        }
    }

    #[derive(Default)]
    struct FakeCopies {
        activation: Vec<u8>,
        ids: Vec<u8>,
        destination: Mutex<Vec<u8>>,
        dtoh_calls: Mutex<usize>,
        fail_dtoh_call: Option<usize>,
        fail_htod: bool,
        fail_sync_call: Option<usize>,
        sync_calls: Mutex<usize>,
    }

    impl FakeCopies {
        fn valid() -> Self {
            Self {
                activation: (0..1024).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
                ids: 0i32.to_le_bytes().to_vec(),
                destination: Mutex::new(vec![0xa5; 4]),
                ..Self::default()
            }
        }
    }

    impl CudaCopies for FakeCopies {
        unsafe fn synchronize(&self, _stream: *mut c_void) -> Result<(), String> {
            let mut calls = self.sync_calls.lock().unwrap();
            *calls += 1;
            if self.fail_sync_call == Some(*calls) {
                return Err("injected synchronize failure".to_string());
            }
            Ok(())
        }

        unsafe fn device_to_host(&self, dst: &mut [u8], src: *const c_void) -> Result<(), String> {
            let mut calls = self.dtoh_calls.lock().unwrap();
            *calls += 1;
            if self.fail_dtoh_call == Some(*calls) {
                return Err("injected device-to-host failure".to_string());
            }
            let source = match src as usize {
                1 => &self.activation,
                2 => &self.ids,
                _ => return Err("unexpected fake device pointer".to_string()),
            };
            if source.len() != dst.len() {
                return Err("fake copy size mismatch".to_string());
            }
            dst.copy_from_slice(source);
            Ok(())
        }

        unsafe fn host_to_device(&self, _dst: *mut c_void, src: &[u8]) -> Result<(), String> {
            if self.fail_htod {
                return Err("injected host-to-device failure".to_string());
            }
            self.destination.lock().unwrap().copy_from_slice(src);
            Ok(())
        }
    }

    struct FakeExecutor {
        fail: bool,
    }

    impl Tq1Executor for FakeExecutor {
        fn execute(&self, _operation: &Tq1MulMatIdOperation) -> Result<Tq1XrtResult, String> {
            if self.fail {
                return Err("injected XRT failure".to_string());
            }
            Ok(Tq1XrtResult {
                outputs: vec![42.0],
                evidence: Tq1XrtEvidence {
                    backend: "fake-xrt",
                    eligible_operations: 1,
                    handled_operations: 1,
                    submission_count: 1,
                    completion_count: 1,
                    per_cu_submissions: vec![1],
                    per_cu_completions: vec![1],
                    stall_codes: vec![1],
                    raw_min: -1,
                    raw_max: 1,
                    matrix_bytes: 262_144,
                    input_bytes: 18_432,
                    output_bytes: 18_432,
                    program_bytes: 96,
                    dispatch_to_stall_ns: 500,
                    clock_hz: 370_000_000,
                    derived_accelerator_cycles: 185,
                    decode_ns: 1,
                    pack_ns: 1,
                    xrt_ns: 1,
                    reconstruct_ns: 1,
                },
            })
        }
    }

    struct WrongOutputExecutor;

    impl Tq1Executor for WrongOutputExecutor {
        fn execute(&self, operation: &Tq1MulMatIdOperation) -> Result<Tq1XrtResult, String> {
            let mut result = FakeExecutor { fail: false }.execute(operation)?;
            result.outputs.push(99.0);
            Ok(result)
        }
    }

    fn with_strict_env(test: impl FnOnce(&Path)) {
        let _lock = crate::r#impl::test_env::lock();
        let directory = tempfile::tempdir().unwrap();
        let log = directory.path().join("evidence.jsonl");
        std::env::set_var("HETGPU_QWEN_TQ1_XRT", "1");
        std::env::set_var("HETGPU_QWEN_TQ1_STRICT", "1");
        std::env::set_var("HETGPU_TQ1_EVIDENCE_LOG", &log);
        test(&log);
        std::env::remove_var("HETGPU_TQ1_EVIDENCE_LOG");
        std::env::remove_var("HETGPU_QWEN_TQ1_STRICT");
        std::env::remove_var("HETGPU_QWEN_TQ1_XRT");
    }

    #[test]
    fn abi_v1_layout_and_return_codes_are_stable() {
        assert_eq!(HETGPU_TQ1_ABI_VERSION, 1);
        assert_eq!(HETGPU_TQ1_NOT_HANDLED, 0);
        assert_eq!(HETGPU_TQ1_HANDLED, 1);
        assert_eq!(HETGPU_TQ1_ERROR, -1);
        assert_eq!(align_of::<HetgpuTq1MulMatIdV1>(), align_of::<u64>());
        assert_eq!(size_of::<HetgpuTq1TensorV1>(), 112);
        assert_eq!(size_of::<HetgpuTq1MulMatIdV1>(), 320);
    }

    #[test]
    fn strict_eligible_failure_never_becomes_not_handled_or_mutates_destination() {
        with_strict_env(|_| {
            let operation = operation_fixture();
            let copies = FakeCopies::valid();
            let result = try_mul_mat_id_with(&operation, &copies, &FakeExecutor { fail: true });
            assert_eq!(result, HETGPU_TQ1_ERROR);
            assert_eq!(*copies.destination.lock().unwrap(), vec![0xa5; 4]);
        });
    }

    #[test]
    fn complete_output_is_published_only_after_all_prior_gates() {
        with_strict_env(|log| {
            let operation = operation_fixture();
            let copies = FakeCopies::valid();
            assert_eq!(
                try_mul_mat_id_with(&operation, &copies, &FakeExecutor { fail: false }),
                HETGPU_TQ1_HANDLED
            );
            assert_eq!(
                f32::from_le_bytes(copies.destination.lock().unwrap()[..4].try_into().unwrap()),
                42.0
            );
            assert_eq!(*copies.sync_calls.lock().unwrap(), 2);
            let evidence = std::fs::read_to_string(log).unwrap();
            assert!(evidence.contains("\"route\":\"handled\""));
            assert!(evidence.contains("\"backend\":\"fake-xrt\""));
        });
    }

    #[test]
    fn cuda_copy_failures_leave_destination_unpublished() {
        with_strict_env(|_| {
            for (dtoh, htod, sync) in [
                (Some(1), false, None),
                (Some(2), false, None),
                (None, true, None),
                (None, false, Some(1)),
            ] {
                let mut copies = FakeCopies::valid();
                copies.fail_dtoh_call = dtoh;
                copies.fail_htod = htod;
                copies.fail_sync_call = sync;
                assert_eq!(
                    try_mul_mat_id_with(
                        &operation_fixture(),
                        &copies,
                        &FakeExecutor { fail: false }
                    ),
                    HETGPU_TQ1_ERROR
                );
                assert_eq!(*copies.destination.lock().unwrap(), vec![0xa5; 4]);
            }
        });
    }

    #[test]
    fn disabled_route_is_not_handled_but_eligible_validation_is_fail_closed() {
        let _lock = crate::r#impl::test_env::lock();
        std::env::remove_var("HETGPU_QWEN_TQ1_XRT");
        assert_eq!(
            try_mul_mat_id_with(
                &operation_fixture(),
                &FakeCopies::valid(),
                &FakeExecutor { fail: false }
            ),
            HETGPU_TQ1_NOT_HANDLED
        );
        drop(_lock);

        with_strict_env(|_| {
            let copies = FakeCopies::valid();
            let executor = FakeExecutor { fail: false };
            let mut operation = operation_fixture();
            operation.abi_version = 2;
            assert_eq!(try_mul_mat_id_with(&operation, &copies, &executor), -1);
            operation = operation_fixture();
            operation.src1_type = 99;
            assert_eq!(try_mul_mat_id_with(&operation, &copies, &executor), -1);
            operation = operation_fixture();
            operation.src1_ne[0] = 512;
            assert_eq!(try_mul_mat_id_with(&operation, &copies, &executor), -1);
            operation = operation_fixture();
            operation.ids_ne[0] = 2;
            assert_eq!(try_mul_mat_id_with(&operation, &copies, &executor), -1);
            operation = operation_fixture();
            operation.src1_device = std::ptr::null();
            assert_eq!(try_mul_mat_id_with(&operation, &copies, &executor), -1);
        });
    }

    #[test]
    fn qualified_unknown_duplicate_ids_and_wrong_output_are_errors() {
        with_strict_env(|_| {
            let executor = FakeExecutor { fail: false };
            let copies = FakeCopies::valid();

            let unknown = CString::new("blk.999.ffn_down_exps.weight").unwrap();
            let mut operation = operation_fixture();
            operation.src0_name = unknown.as_ptr();
            assert_eq!(try_mul_mat_id_with(&operation, &copies, &executor), -1);
            assert_eq!(*copies.destination.lock().unwrap(), vec![0xa5; 4]);

            let unqualified = CString::new("token_embd.weight").unwrap();
            operation = operation_fixture();
            operation.src0_name = unqualified.as_ptr();
            assert_eq!(
                try_mul_mat_id_with(&operation, &copies, &executor),
                HETGPU_TQ1_NOT_HANDLED
            );

            let mut duplicate_copies = FakeCopies::valid();
            duplicate_copies.ids = [0i32.to_le_bytes(), 0i32.to_le_bytes()].concat();
            duplicate_copies.destination = Mutex::new(vec![0xa5; 8]);
            operation = operation_fixture();
            operation.ids_ne = [2, 1, 1, 1];
            operation.ids_nb = [4, 8, 8, 8];
            operation.dst_ne = [1, 2, 1, 1];
            operation.dst_nb = [4, 4, 8, 8];
            assert_eq!(
                try_mul_mat_id_with(&operation, &duplicate_copies, &executor),
                HETGPU_TQ1_ERROR
            );
            assert_eq!(*duplicate_copies.destination.lock().unwrap(), vec![0xa5; 8]);

            let wrong_output_copies = FakeCopies::valid();
            assert_eq!(
                try_mul_mat_id_with(
                    &operation_fixture(),
                    &wrong_output_copies,
                    &WrongOutputExecutor
                ),
                HETGPU_TQ1_ERROR
            );
            assert_eq!(
                *wrong_output_copies.destination.lock().unwrap(),
                vec![0xa5; 4]
            );
        });
    }

    #[test]
    fn strict_evidence_path_is_preflighted_before_cuda_copies() {
        let _lock = crate::r#impl::test_env::lock();
        let directory = tempfile::tempdir().unwrap();
        std::env::set_var("HETGPU_QWEN_TQ1_XRT", "1");
        std::env::set_var("HETGPU_QWEN_TQ1_STRICT", "1");
        std::env::set_var("HETGPU_TQ1_EVIDENCE_LOG", directory.path());
        let copies = FakeCopies::valid();

        assert_eq!(
            try_mul_mat_id_with(&operation_fixture(), &copies, &FakeExecutor { fail: false }),
            HETGPU_TQ1_ERROR
        );
        assert_eq!(*copies.dtoh_calls.lock().unwrap(), 0);
        assert_eq!(*copies.destination.lock().unwrap(), vec![0xa5; 4]);
        std::env::remove_var("HETGPU_TQ1_EVIDENCE_LOG");
        std::env::remove_var("HETGPU_QWEN_TQ1_STRICT");
        std::env::remove_var("HETGPU_QWEN_TQ1_XRT");
    }

    #[test]
    fn registration_entrypoint_validates_and_is_idempotent() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; 216]).unwrap();
        file.flush().unwrap();
        let name = CString::new("blk.78.ffn_down_exps.weight").unwrap();
        let path = CString::new(file.path().to_str().unwrap()).unwrap();
        let mut tensor = HetgpuTq1TensorV1 {
            abi_version: HETGPU_TQ1_ABI_VERSION,
            ggml_type: GGML_TYPE_TQ1_0,
            role: ROLE_DOWN,
            file_index: 0,
            name: name.as_ptr(),
            path: path.as_ptr(),
            file_offset: 0,
            nbytes: 216,
            ne: [1024, 1, 1, 1],
            nb: [54, 216, 216, 216],
        };
        assert_eq!(unsafe { register_tensor(&tensor) }, HETGPU_TQ1_HANDLED);
        assert_eq!(unsafe { register_tensor(&tensor) }, HETGPU_TQ1_HANDLED);
        tensor.abi_version = 2;
        assert_eq!(unsafe { register_tensor(&tensor) }, HETGPU_TQ1_ERROR);
        assert_eq!(
            unsafe { register_tensor(std::ptr::null()) },
            HETGPU_TQ1_ERROR
        );
    }
}
