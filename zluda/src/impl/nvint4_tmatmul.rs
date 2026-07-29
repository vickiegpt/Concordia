use crate::r#impl::cxl_tmatmul::{self, CudaDaxContext, Nvint4HardwareStatus};
use crate::r#impl::nvidia_runtime_sys;
use cuda_types::cuda::{CUdevice, CUdeviceptr_v2, CUfunction, CUmodule, CUstream};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ptr;
use std::sync::{Mutex, OnceLock};

const NVINT4_DIM: u32 = 2048;
const NVINT4_INPUT_BYTES: usize = NVINT4_DIM as usize * 2;
const NVINT4_OUTPUT_BYTES: usize = NVINT4_DIM as usize * 8;
const CONVERTER_THREADS: u32 = 256;
const CONVERTER_PTX: &str = include_str!("nvint4_to_packed_ternary.ptx");
const CONVERTER_NAME: &str = "nvint4_to_packed_ternary";
pub(crate) const NVINT4_ENTRY: &str = "tmatmul_nvint4_dense";

static CUDA_STATES: OnceLock<Mutex<HashMap<i32, Nvint4CudaState>>> = OnceLock::new();

struct Nvint4CudaState {
    _module: CUmodule,
    function: CUfunction,
    scratch: CUdeviceptr_v2,
    scratch_bytes: usize,
}

unsafe impl Send for Nvint4CudaState {}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvertedMatrix {
    pub(crate) device_ptr: usize,
    pub(crate) bytes: usize,
    pub(crate) cuda_device: i32,
    pub(crate) stream: CUstream,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Nvint4Launch {
    pub(crate) packed_weights: usize,
    pub(crate) input_q8_8: usize,
    pub(crate) output_s64: usize,
    pub(crate) dim: u32,
    pub(crate) delta: u32,
    pub(crate) stream: CUstream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Nvint4Execution {
    status: Nvint4HardwareStatus,
    cuda_device: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Nvint4RouteConfig {
    enabled: bool,
    gpu_fallback: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureDisposition {
    StrictFailure,
    GpuFallback,
}

fn failure_disposition(fallback_enabled: bool) -> FailureDisposition {
    if fallback_enabled {
        FailureDisposition::GpuFallback
    } else {
        FailureDisposition::StrictFailure
    }
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub(crate) fn is_nvint4_entry(kernel_name: &str) -> bool {
    kernel_name == NVINT4_ENTRY
}

fn validate_launch_shape(grid: (u32, u32, u32), block: (u32, u32, u32)) -> Result<(), String> {
    if grid != (1, 1, 1) || block != (1, 1, 1) {
        return Err(format!(
            "{NVINT4_ENTRY} requires grid=(1,1,1) block=(1,1,1), got grid={grid:?} block={block:?}"
        ));
    }
    Ok(())
}

unsafe fn read_param<T: Copy>(
    kernel_params: *mut *mut c_void,
    index: usize,
    role: &str,
) -> Result<T, String> {
    if kernel_params.is_null() {
        return Err(format!("{NVINT4_ENTRY} has null kernel_params"));
    }
    let slot = *kernel_params.add(index);
    if slot.is_null() || (slot as usize) < 0x1000 {
        return Err(format!(
            "{NVINT4_ENTRY} PARAM_{index} ({role}) has invalid slot 0x{:x}",
            slot as usize
        ));
    }
    Ok((slot as *const T).read_unaligned())
}

pub(crate) unsafe fn parse_launch_params(
    kernel_params: *mut *mut c_void,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    stream: CUstream,
) -> Result<Nvint4Launch, String> {
    validate_launch_shape(grid, block)?;
    let launch = Nvint4Launch {
        packed_weights: read_param(kernel_params, 0, "packed_weights")?,
        input_q8_8: read_param(kernel_params, 1, "input_q8_8")?,
        output_s64: read_param(kernel_params, 2, "output_s64")?,
        dim: read_param(kernel_params, 3, "dim")?,
        delta: read_param(kernel_params, 4, "delta")?,
        stream,
    };
    if launch.packed_weights < 0x1000 {
        return Err(format!(
            "{NVINT4_ENTRY} packed_weights pointer is invalid: 0x{:x}",
            launch.packed_weights
        ));
    }
    if launch.input_q8_8 < 0x1000 {
        return Err(format!(
            "{NVINT4_ENTRY} input_q8_8 pointer is invalid: 0x{:x}",
            launch.input_q8_8
        ));
    }
    if launch.output_s64 < 0x1000 {
        return Err(format!(
            "{NVINT4_ENTRY} output_s64 pointer is invalid: 0x{:x}",
            launch.output_s64
        ));
    }
    if launch.dim != NVINT4_DIM {
        return Err(format!(
            "{NVINT4_ENTRY} requires dim={NVINT4_DIM}, got {}",
            launch.dim
        ));
    }
    if launch.delta > 7 {
        return Err(format!(
            "{NVINT4_ENTRY} delta must be in [0, 7], got {}",
            launch.delta
        ));
    }
    Ok(launch)
}

fn sign_extend_nibble(raw: u8) -> i8 {
    ((raw << 4) as i8) >> 4
}

fn ternary_code(value: i8, delta: u32) -> u8 {
    if value < -(delta as i8) {
        3
    } else if value > delta as i8 {
        1
    } else {
        0
    }
}

fn pack_ternary_codes(codes: [u8; 4]) -> u8 {
    codes
        .into_iter()
        .enumerate()
        .fold(0, |packed, (lane, code)| {
            packed | ((code & 0x3) << (2 * lane))
        })
}

fn nvint4_extents(dim: u32) -> Result<(usize, usize), String> {
    let elements = usize::try_from(dim)
        .ok()
        .and_then(|d| d.checked_mul(d))
        .ok_or_else(|| "NVINT4 dimension overflow".to_string())?;
    if elements % 4 != 0 {
        return Err(format!(
            "NVINT4 element count {elements} is not divisible by 4"
        ));
    }
    Ok((elements / 2, elements / 4))
}

fn range_fits_allocation(ptr: usize, bytes: usize, base: usize, allocation_bytes: usize) -> bool {
    ptr >= base
        && ptr
            .checked_add(bytes)
            .zip(base.checked_add(allocation_bytes))
            .is_some_and(|(end, allocation_end)| end <= allocation_end)
}

fn cuda_call(call: &str, result: i32) -> Result<(), String> {
    if result == 0 {
        Ok(())
    } else {
        Err(format!("{call} failed: cuResult={result}"))
    }
}

fn cuda_states() -> &'static Mutex<HashMap<i32, Nvint4CudaState>> {
    CUDA_STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn validate_cuda_allocation(ptr: usize, need: usize, role: &str) -> Result<(), String> {
    if ptr == 0 {
        return Err(format!("{role} pointer is null"));
    }
    let mut base = CUdeviceptr_v2(ptr::null_mut());
    let mut bytes = 0usize;
    cuda_call(
        "cuMemGetAddressRange_v2",
        nvidia_runtime_sys::cuMemGetAddressRange_v2(
            &mut base,
            &mut bytes,
            CUdeviceptr_v2(ptr as *mut c_void),
        ),
    )
    .map_err(|error| format!("{role} is not a tracked CUDA allocation: {error}"))?;
    let base = base.0 as usize;
    if !range_fits_allocation(ptr, need, base, bytes) {
        return Err(format!(
            "{role} allocation too small: ptr=0x{ptr:x} base=0x{base:x} have={bytes} need={need}"
        ));
    }
    Ok(())
}

fn create_cuda_state(scratch_bytes: usize) -> Result<Nvint4CudaState, String> {
    let ptx = CString::new(CONVERTER_PTX)
        .map_err(|error| format!("converter PTX contains NUL: {error}"))?;
    let name = CString::new(CONVERTER_NAME).expect("static converter name contains no NUL");
    let mut module = CUmodule(ptr::null_mut());
    cuda_call(
        "cuModuleLoadData(nvint4 converter)",
        nvidia_runtime_sys::cuModuleLoadData(&mut module, ptx.as_ptr().cast()),
    )?;

    let mut function = CUfunction(ptr::null_mut());
    if let Err(error) = cuda_call(
        "cuModuleGetFunction(nvint4 converter)",
        nvidia_runtime_sys::cuModuleGetFunction(&mut function, module, name.as_ptr()),
    ) {
        let _ = nvidia_runtime_sys::cuModuleUnload(module);
        return Err(error);
    }

    let mut scratch = CUdeviceptr_v2(ptr::null_mut());
    if let Err(error) = cuda_call(
        "cuMemAlloc_v2(nvint4 packed scratch)",
        nvidia_runtime_sys::cuMemAlloc_v2(&mut scratch, scratch_bytes),
    ) {
        let _ = nvidia_runtime_sys::cuModuleUnload(module);
        return Err(error);
    }

    Ok(Nvint4CudaState {
        _module: module,
        function,
        scratch,
        scratch_bytes,
    })
}

pub(crate) unsafe fn convert(
    packed_weights: usize,
    dim: u32,
    delta: u32,
    stream: CUstream,
) -> Result<ConvertedMatrix, String> {
    if dim != NVINT4_DIM {
        return Err(format!(
            "NVINT4 converter requires dim={NVINT4_DIM}, got {dim}"
        ));
    }
    if delta > 7 {
        return Err(format!("NVINT4 delta must be in [0, 7], got {delta}"));
    }
    let (source_bytes, packed_bytes) = nvint4_extents(dim)?;
    nvidia_runtime_sys::init()?;
    validate_cuda_allocation(packed_weights, source_bytes, "packed_weights")?;
    let mut device: CUdevice = 0;
    cuda_call(
        "cuCtxGetDevice",
        nvidia_runtime_sys::cuCtxGetDevice(&mut device),
    )?;

    let states = cuda_states();
    let mut states = states
        .lock()
        .map_err(|_| "NVINT4 CUDA state lock poisoned".to_string())?;
    if !states.contains_key(&device) {
        states.insert(device, create_cuda_state(packed_bytes)?);
    }
    let state = states
        .get_mut(&device)
        .expect("NVINT4 CUDA state inserted above");
    if state.scratch_bytes != packed_bytes {
        return Err(format!(
            "cached NVINT4 scratch has {} bytes, launch requires {packed_bytes}",
            state.scratch_bytes
        ));
    }

    let mut source = CUdeviceptr_v2(packed_weights as *mut c_void);
    let mut destination = state.scratch;
    let mut packed_bytes_u32 = u32::try_from(packed_bytes)
        .map_err(|_| format!("packed byte count {packed_bytes} does not fit u32"))?;
    let mut delta_value = delta;
    let mut params = [
        (&mut source as *mut CUdeviceptr_v2).cast::<c_void>(),
        (&mut destination as *mut CUdeviceptr_v2).cast::<c_void>(),
        (&mut packed_bytes_u32 as *mut u32).cast::<c_void>(),
        (&mut delta_value as *mut u32).cast::<c_void>(),
    ];
    let blocks = packed_bytes_u32.div_ceil(CONVERTER_THREADS);
    cuda_call(
        "cuLaunchKernel(nvint4 converter)",
        nvidia_runtime_sys::cuLaunchKernel(
            state.function,
            blocks,
            1,
            1,
            CONVERTER_THREADS,
            1,
            1,
            0,
            stream,
            params.as_mut_ptr(),
            ptr::null_mut(),
        ),
    )?;
    cuda_call(
        "cuStreamSynchronize(nvint4 converter)",
        nvidia_runtime_sys::cuStreamSynchronize_ckpt(stream),
    )?;

    Ok(ConvertedMatrix {
        device_ptr: state.scratch.0 as usize,
        bytes: packed_bytes,
        cuda_device: device,
        stream,
    })
}

unsafe fn execute_hardware(
    launch: Nvint4Launch,
    timeout_ms: u32,
) -> Result<Nvint4Execution, String> {
    validate_cuda_allocation(launch.input_q8_8, NVINT4_INPUT_BYTES, "input_q8_8")?;
    validate_cuda_allocation(launch.output_s64, NVINT4_OUTPUT_BYTES, "output_s64")?;
    let converted = convert(
        launch.packed_weights,
        launch.dim,
        launch.delta,
        launch.stream,
    )?;
    let status = cxl_tmatmul::submit_nvint4_packed_hardware_from_device_ptrs(
        converted.device_ptr,
        launch.input_q8_8,
        launch.output_s64,
        launch.dim,
        CudaDaxContext {
            gpu: converted.cuda_device,
            stream: launch.stream.0 as usize,
        },
        timeout_ms,
    )
    .map_err(|error| error.to_string())?;
    Ok(Nvint4Execution {
        status,
        cuda_device: converted.cuda_device,
    })
}

fn hardware_json(status: &Nvint4HardwareStatus) -> serde_json::Value {
    serde_json::json!({
        "instruction_dma_status": status.instruction_dma_status,
        "stall_status": status.stall_status,
        "wide_dma_status": status.wide_dma_status,
        "wide_dma_bytes": NVINT4_OUTPUT_BYTES,
        "exec_status": status.exec_status,
        "instruction_count": status.instruction_count,
        "ls_read_beats": status.ls_read_beats,
        "tmatmul_read_beats": status.tmatmul_read_beats,
        "elapsed_us": status.elapsed_us,
    })
}

fn route_record(
    launch: Option<Nvint4Launch>,
    stream: CUstream,
    final_status: &str,
    reason: &str,
    execution: Option<&Nvint4Execution>,
    total_us: u64,
) -> serde_json::Value {
    serde_json::json!({
        "route": "ptx_nvint4_to_dax_tmatmul",
        "kernel": NVINT4_ENTRY,
        "final_status": final_status,
        "reason": reason,
        "dim": launch.map(|value| value.dim),
        "delta": launch.map(|value| value.delta),
        "cuda_device": execution.map(|value| value.cuda_device),
        "cuda_stream": format!("0x{:x}", stream.0 as usize),
        "source_bytes": 2_097_152,
        "packed_bytes": 1_048_576,
        "input_bytes": NVINT4_INPUT_BYTES,
        "output_bytes": NVINT4_OUTPUT_BYTES,
        "dpa": {
            "matrix": "0x000000",
            "input": "0x400000",
            "output": "0x500000",
            "program": "0x600000",
        },
        "timing_us": {
            "total": total_us,
            "hardware": execution.map(|value| value.status.elapsed_us),
        },
        "hardware": execution.map(|value| hardware_json(&value.status)),
        "mismatch_count": serde_json::Value::Null,
    })
}

unsafe fn try_launch_with<E, L>(
    config: Nvint4RouteConfig,
    kernel_name: &str,
    kernel_params: *mut *mut c_void,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    stream: CUstream,
    executor: E,
    mut logger: L,
) -> Option<Result<(), String>>
where
    E: FnOnce(Nvint4Launch) -> Result<Nvint4Execution, String>,
    L: FnMut(&serde_json::Value) -> Result<(), String>,
{
    if !is_nvint4_entry(kernel_name) || !config.enabled {
        return None;
    }

    let started = std::time::Instant::now();
    let launch = match parse_launch_params(kernel_params, grid, block, stream) {
        Ok(launch) => launch,
        Err(reason) => {
            let final_status = if config.gpu_fallback {
                "gpu_fallback"
            } else {
                "fail"
            };
            let record = route_record(
                None,
                stream,
                final_status,
                &reason,
                None,
                u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            );
            if let Err(log_error) = logger(&record) {
                eprintln!("[NVINT4 TMatmul] route log failed: {log_error}");
            }
            return if config.gpu_fallback {
                None
            } else {
                Some(Err(reason))
            };
        }
    };

    match executor(launch) {
        Ok(execution) => {
            let record = route_record(
                Some(launch),
                stream,
                "pass",
                "hardware execution completed",
                Some(&execution),
                u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            );
            match logger(&record) {
                Ok(()) => Some(Ok(())),
                Err(error) => Some(Err(format!(
                    "NVINT4 hardware completed but route log failed: {error}"
                ))),
            }
        }
        Err(reason) => {
            let final_status = if config.gpu_fallback {
                "gpu_fallback"
            } else {
                "fail"
            };
            let record = route_record(
                Some(launch),
                stream,
                final_status,
                &reason,
                None,
                u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            );
            let log_result = logger(&record);
            if let Err(log_error) = &log_result {
                eprintln!("[NVINT4 TMatmul] route log failed: {log_error}");
            }
            if config.gpu_fallback {
                None
            } else {
                Some(Err(match log_result {
                    Ok(()) => reason,
                    Err(log_error) => format!("{reason}; route log failed: {log_error}"),
                }))
            }
        }
    }
}

fn open_route_log() -> Result<Option<File>, String> {
    let Ok(path) = std::env::var("HETGPU_NVINT4_ROUTE_LOG") else {
        return Ok(None);
    };
    if path.trim().is_empty() {
        return Ok(None);
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map(Some)
        .map_err(|error| format!("open HETGPU_NVINT4_ROUTE_LOG={path:?}: {error}"))
}

pub(crate) unsafe fn try_launch(
    kernel_name: &str,
    kernel_params: *mut *mut c_void,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    stream: CUstream,
) -> Option<Result<(), String>> {
    let config = Nvint4RouteConfig {
        enabled: env_truthy("HETGPU_NVINT4_TMATMUL"),
        gpu_fallback: env_truthy("HETGPU_NVINT4_GPU_FALLBACK"),
    };
    if !is_nvint4_entry(kernel_name) || !config.enabled {
        return None;
    }

    let mut route_log = match open_route_log() {
        Ok(file) => file,
        Err(reason) => {
            let final_status = if config.gpu_fallback {
                "gpu_fallback"
            } else {
                "fail"
            };
            let launch = parse_launch_params(kernel_params, grid, block, stream).ok();
            let record = route_record(launch, stream, final_status, &reason, None, 0);
            eprintln!("{record}");
            return if config.gpu_fallback {
                None
            } else {
                Some(Err(reason))
            };
        }
    };
    try_launch_with(
        config,
        kernel_name,
        kernel_params,
        grid,
        block,
        stream,
        |launch| execute_hardware(launch, 0),
        |record| {
            eprintln!("{record}");
            if let Some(file) = route_log.as_mut() {
                writeln!(file, "{record}").map_err(|error| error.to_string())?;
                file.flush().map_err(|error| error.to_string())?;
            }
            Ok(())
        },
    )
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_nvint4_convert_for_test(
    src: usize,
    dim: u32,
    delta: u32,
    stream: CUstream,
    scratch_out: *mut usize,
    bytes_out: *mut usize,
) -> i32 {
    if std::env::var("HETGPU_NVINT4_CONVERTER_TEST")
        .ok()
        .as_deref()
        != Some("1")
    {
        return 2;
    }
    if scratch_out.is_null() || bytes_out.is_null() {
        return 3;
    }
    match convert(src, dim, delta, stream) {
        Ok(converted) => {
            scratch_out.write(converted.device_ptr);
            bytes_out.write(converted.bytes);
            0
        }
        Err(error) => {
            eprintln!("[NVINT4 converter test] {error}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        failure_disposition, is_nvint4_entry, nvint4_extents, pack_ternary_codes,
        parse_launch_params, range_fits_allocation, sign_extend_nibble, ternary_code,
        validate_launch_shape, FailureDisposition, NVINT4_ENTRY,
    };
    use core::ffi::c_void;
    use cuda_types::cuda::CUstream;
    use std::ptr;

    #[test]
    fn all_nibbles_and_deltas_match_contract() {
        for delta in 0..=7 {
            for raw in 0u8..=15 {
                let signed = sign_extend_nibble(raw);
                let expected = if signed < -(delta as i8) {
                    3
                } else if signed > delta as i8 {
                    1
                } else {
                    0
                };
                assert_eq!(ternary_code(signed, delta), expected);
            }
        }
    }

    #[test]
    fn four_codes_pack_low_bits_first() {
        assert_eq!(pack_ternary_codes([3, 1, 3, 1]), 0x77);
        assert_eq!(pack_ternary_codes([1, 0, 3, 0]), 0x31);
    }

    #[test]
    fn fixed_extents_are_checked() {
        assert_eq!(nvint4_extents(2048).unwrap(), (2_097_152, 1_048_576));
        assert!(nvint4_extents(3).is_err());
    }

    #[test]
    fn allocation_ranges_accept_interior_pointer_and_reject_overflow() {
        assert!(range_fits_allocation(0x1000, 0x1000, 0x1000, 0x1000));
        assert!(range_fits_allocation(0x1200, 0x800, 0x1000, 0x1000));
        assert!(!range_fits_allocation(0x1200, 0xe01, 0x1000, 0x1000));
        assert!(!range_fits_allocation(0x0fff, 1, 0x1000, 0x1000));
        assert!(!range_fits_allocation(usize::MAX - 3, 8, 0, usize::MAX));
    }

    #[test]
    fn exact_entry_name_is_the_only_match() {
        assert!(is_nvint4_entry(NVINT4_ENTRY));
        assert!(!is_nvint4_entry("_Z22tmatmul_nvint4_densev"));
        assert!(!is_nvint4_entry("tmatmul_nvint4_dense_v2"));
        assert!(!is_nvint4_entry("mul_mat_q"));
    }

    #[test]
    fn exact_entry_requires_one_logical_invocation() {
        assert!(validate_launch_shape((1, 1, 1), (1, 1, 1)).is_ok());
        for (grid, block) in [
            ((2, 1, 1), (1, 1, 1)),
            ((1, 2, 1), (1, 1, 1)),
            ((1, 1, 2), (1, 1, 1)),
            ((1, 1, 1), (2, 1, 1)),
            ((1, 1, 1), (1, 2, 1)),
            ((1, 1, 1), (1, 1, 2)),
        ] {
            assert!(validate_launch_shape(grid, block).is_err());
        }
    }

    #[test]
    fn parser_reads_three_pointers_dim_and_delta() {
        let mut weights = 0x1000usize;
        let mut input = 0x2000usize;
        let mut output = 0x3000usize;
        let mut dim = 2048u32;
        let mut delta = 1u32;
        let mut params = [
            (&mut weights as *mut usize).cast::<c_void>(),
            (&mut input as *mut usize).cast::<c_void>(),
            (&mut output as *mut usize).cast::<c_void>(),
            (&mut dim as *mut u32).cast::<c_void>(),
            (&mut delta as *mut u32).cast::<c_void>(),
        ];

        let launch = unsafe {
            parse_launch_params(
                params.as_mut_ptr(),
                (1, 1, 1),
                (1, 1, 1),
                CUstream(ptr::null_mut()),
            )
            .unwrap()
        };
        assert_eq!(launch.packed_weights, weights);
        assert_eq!(launch.input_q8_8, input);
        assert_eq!(launch.output_s64, output);
        assert_eq!(launch.dim, 2048);
        assert_eq!(launch.delta, 1);
    }

    #[test]
    fn parser_rejects_null_slots_wrong_dim_and_delta() {
        let error = unsafe {
            parse_launch_params(
                ptr::null_mut(),
                (1, 1, 1),
                (1, 1, 1),
                CUstream(ptr::null_mut()),
            )
            .unwrap_err()
        };
        assert!(error.contains("null kernel_params"));

        let mut weights = 0x1000usize;
        let mut input = 0x2000usize;
        let mut output = 0x3000usize;
        for (dim_value, delta_value, expected) in [
            (1024u32, 1u32, "dim=2048"),
            (2048u32, 8u32, "delta must be in [0, 7]"),
        ] {
            let mut dim = dim_value;
            let mut delta = delta_value;
            let mut params = [
                (&mut weights as *mut usize).cast::<c_void>(),
                (&mut input as *mut usize).cast::<c_void>(),
                (&mut output as *mut usize).cast::<c_void>(),
                (&mut dim as *mut u32).cast::<c_void>(),
                (&mut delta as *mut u32).cast::<c_void>(),
            ];
            let error = unsafe {
                parse_launch_params(
                    params.as_mut_ptr(),
                    (1, 1, 1),
                    (1, 1, 1),
                    CUstream(ptr::null_mut()),
                )
                .unwrap_err()
            };
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn strict_is_default_and_fallback_requires_explicit_opt_in() {
        assert_eq!(
            failure_disposition(false),
            FailureDisposition::StrictFailure
        );
        assert_eq!(failure_disposition(true), FailureDisposition::GpuFallback);
    }
}

#[cfg(test)]
mod nvidia_nvint4_route_tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    fn fake_execution() -> Nvint4Execution {
        Nvint4Execution {
            status: Nvint4HardwareStatus {
                instruction_dma_status: 2,
                stall_status: 1,
                wide_dma_status: 2,
                exec_status: 0,
                instruction_count: 4,
                ls_read_beats: 64,
                tmatmul_read_beats: 16_384,
                elapsed_us: 20,
            },
            cuda_device: 0,
        }
    }

    #[test]
    fn exact_route_strict_failure_does_not_pass_through() {
        let mut weights = 0x1000usize;
        let mut input = 0x2000usize;
        let mut output = 0x3000usize;
        let mut dim = 2048u32;
        let mut delta = 1u32;
        let mut params = [
            (&mut weights as *mut usize).cast::<c_void>(),
            (&mut input as *mut usize).cast::<c_void>(),
            (&mut output as *mut usize).cast::<c_void>(),
            (&mut dim as *mut u32).cast::<c_void>(),
            (&mut delta as *mut u32).cast::<c_void>(),
        ];
        let records = RefCell::new(Vec::new());

        let result = unsafe {
            try_launch_with(
                Nvint4RouteConfig {
                    enabled: true,
                    gpu_fallback: false,
                },
                NVINT4_ENTRY,
                params.as_mut_ptr(),
                (1, 1, 1),
                (1, 1, 1),
                CUstream(ptr::null_mut()),
                |_| Err("injected hardware failure".to_string()),
                |record| {
                    records.borrow_mut().push(record.clone());
                    Ok(())
                },
            )
        };

        assert!(matches!(result, Some(Err(_))));
        assert_eq!(records.borrow()[0]["final_status"], "fail");
        assert_eq!(records.borrow()[0]["reason"], "injected hardware failure");
    }

    #[test]
    fn explicit_gpu_fallback_returns_none_and_records_reason() {
        let mut weights = 0x1000usize;
        let mut input = 0x2000usize;
        let mut output = 0x3000usize;
        let mut dim = 2048u32;
        let mut delta = 1u32;
        let mut params = [
            (&mut weights as *mut usize).cast::<c_void>(),
            (&mut input as *mut usize).cast::<c_void>(),
            (&mut output as *mut usize).cast::<c_void>(),
            (&mut dim as *mut u32).cast::<c_void>(),
            (&mut delta as *mut u32).cast::<c_void>(),
        ];
        let records = RefCell::new(Vec::new());

        let result = unsafe {
            try_launch_with(
                Nvint4RouteConfig {
                    enabled: true,
                    gpu_fallback: true,
                },
                NVINT4_ENTRY,
                params.as_mut_ptr(),
                (1, 1, 1),
                (1, 1, 1),
                CUstream(ptr::null_mut()),
                |_| Err("injected hardware failure".to_string()),
                |record| {
                    records.borrow_mut().push(record.clone());
                    Ok(())
                },
            )
        };

        assert!(result.is_none());
        assert_eq!(records.borrow()[0]["final_status"], "gpu_fallback");
        assert_eq!(records.borrow()[0]["reason"], "injected hardware failure");
    }

    #[test]
    fn successful_exact_route_records_hardware_and_consumes_launch() {
        let mut weights = 0x1000usize;
        let mut input = 0x2000usize;
        let mut output = 0x3000usize;
        let mut dim = 2048u32;
        let mut delta = 3u32;
        let mut params = [
            (&mut weights as *mut usize).cast::<c_void>(),
            (&mut input as *mut usize).cast::<c_void>(),
            (&mut output as *mut usize).cast::<c_void>(),
            (&mut dim as *mut u32).cast::<c_void>(),
            (&mut delta as *mut u32).cast::<c_void>(),
        ];
        let records = RefCell::new(Vec::new());

        let result = unsafe {
            try_launch_with(
                Nvint4RouteConfig {
                    enabled: true,
                    gpu_fallback: false,
                },
                NVINT4_ENTRY,
                params.as_mut_ptr(),
                (1, 1, 1),
                (1, 1, 1),
                CUstream(ptr::null_mut()),
                |_| Ok(fake_execution()),
                |record| {
                    records.borrow_mut().push(record.clone());
                    Ok(())
                },
            )
        };

        assert!(matches!(result, Some(Ok(()))));
        let records = records.borrow();
        assert_eq!(records[0]["final_status"], "pass");
        assert_eq!(records[0]["delta"], 3);
        assert_eq!(records[0]["hardware"]["wide_dma_status"], 2);
        assert_eq!(records[0]["hardware"]["wide_dma_bytes"], 16_384);
    }

    #[test]
    fn disabled_or_nonexact_route_never_calls_executor() {
        let called = Cell::new(false);
        let result = unsafe {
            try_launch_with(
                Nvint4RouteConfig {
                    enabled: true,
                    gpu_fallback: false,
                },
                "tmatmul_nvint4_dense_v2",
                ptr::null_mut(),
                (1, 1, 1),
                (1, 1, 1),
                CUstream(ptr::null_mut()),
                |_| {
                    called.set(true);
                    Ok(fake_execution())
                },
                |_| Ok(()),
            )
        };
        assert!(result.is_none());
        assert!(!called.get());

        let result = unsafe {
            try_launch_with(
                Nvint4RouteConfig {
                    enabled: false,
                    gpu_fallback: false,
                },
                NVINT4_ENTRY,
                ptr::null_mut(),
                (1, 1, 1),
                (1, 1, 1),
                CUstream(ptr::null_mut()),
                |_| {
                    called.set(true);
                    Ok(fake_execution())
                },
                |_| Ok(()),
            )
        };
        assert!(result.is_none());
        assert!(!called.get());
    }
}
