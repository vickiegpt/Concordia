use crate::r#impl::nvidia_runtime_sys;
use cuda_types::cuda::{CUdevice, CUdeviceptr_v2, CUfunction, CUmodule, CUstream};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::{Mutex, OnceLock};

const NVINT4_DIM: u32 = 2048;
const CONVERTER_THREADS: u32 = 256;
const CONVERTER_PTX: &str = include_str!("nvint4_to_packed_ternary.ptx");
const CONVERTER_NAME: &str = "nvint4_to_packed_ternary";

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
        nvint4_extents, pack_ternary_codes, range_fits_allocation, sign_extend_nibble, ternary_code,
    };

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
}
