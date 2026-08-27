pub(crate) mod r#impl;

/// Production-backed batch planner entrypoint used by external evaluation tests.
///
/// This intentionally exposes only the ordered logical slices, keeping the scheduler's
/// invariant-bearing types private to the runtime.
#[doc(hidden)]
#[cfg(all(
    unix,
    feature = "nvidia",
    feature = "evaluation",
    not(feature = "amd"),
    not(feature = "intel")
))]
pub fn hetgpu_v3_batch_plan_for_evaluation(
    logical_batch: u32,
    live_max_batch: u32,
    configured_limit: Option<&str>,
) -> std::result::Result<Vec<(u32, u32)>, String> {
    let config = crate::r#impl::batch_scheduler::BatchSchedulerConfig::parse(
        configured_limit,
        live_max_batch,
    )?;
    let plan = config.plan(logical_batch)?;
    Ok(plan
        .slices()
        .iter()
        .map(|slice| (slice.first(), slice.count()))
        .collect())
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_tq1_register_tensor_v1(
    tensor: *const crate::r#impl::tq1_bridge::HetgpuTq1TensorV1,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::tq1_bridge::register_tensor(tensor)
    }))
    .unwrap_or(crate::r#impl::tq1_bridge::HETGPU_TQ1_ERROR)
}

#[cfg(all(
    unix,
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_tq1_try_mul_mat_id_v1(
    operation: *const crate::r#impl::tq1_bridge::HetgpuTq1MulMatIdV1,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::r#impl::tq1_bridge::try_mul_mat_id(operation)
    }))
    .unwrap_or(crate::r#impl::tq1_bridge::HETGPU_TQ1_ERROR)
}
// Import necessary for FromCuda
use crate::r#impl::FromCuda;
// Import std::ptr for null_mut
use std::ptr;
// Import Ze types
#[cfg(feature = "intel")]
use ze_runtime_sys::ze_device_handle_t;
// Import CUerror for Result
use cuda_types::cuda::CUerror;
// Define Result type to match FromCuda error return type
type Result<T> = std::result::Result<T, CUerror>;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

#[cfg(unix)]
use libc::{dlsym, RTLD_DEFAULT};

// Note: The cudart shim symbols are force-linked via --whole-archive in build.rs,
// so we don't need an explicit anchor function here.

// FFI wrapper for LZ4 decompression called from cudart_shim.c
#[no_mangle]
pub extern "C" fn hetgpu_lz4_decompress(
    src: *const c_char,
    dst: *mut c_char,
    compressed_size: c_int,
    dst_capacity: c_int,
) -> c_int {
    unsafe { lz4_sys::LZ4_decompress_safe(src, dst, compressed_size, dst_capacity) }
}

// FFI wrapper for Zstandard PTX payloads in CUDA fatbins. Newer CUDA toolchains
// emit zstd-compressed entries while the legacy shim path only tried LZ4.
#[no_mangle]
pub extern "C" fn hetgpu_zstd_decompress(
    src: *const c_char,
    dst: *mut c_char,
    compressed_size: c_int,
    dst_capacity: c_int,
) -> c_int {
    if src.is_null() || dst.is_null() || compressed_size <= 0 || dst_capacity <= 0 {
        return -1;
    }

    use std::io::Read;

    let input = unsafe { std::slice::from_raw_parts(src as *const u8, compressed_size as usize) };
    let mut cursor = std::io::Cursor::new(input);
    let decoder = match ruzstd::StreamingDecoder::new(&mut cursor) {
        Ok(decoder) => decoder,
        Err(_) => return -2,
    };
    let mut limited = decoder.take(dst_capacity as u64 + 1);
    let mut decoded = Vec::new();
    if limited.read_to_end(&mut decoded).is_err() {
        return -3;
    }
    if decoded.len() > dst_capacity as usize {
        return -4;
    }

    unsafe {
        ptr::copy_nonoverlapping(decoded.as_ptr(), dst as *mut u8, decoded.len());
    }
    decoded.len() as c_int
}

#[cfg(feature = "intel")]
fn hetgpu_log_cuda_calls_enabled() -> bool {
    std::env::var("HETGPU_LOG_CUDA_CALLS")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(all(
    any(feature = "sifive", feature = "nvidia"),
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_launch_named_kernel(
    kernel_name: *const c_char,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32 {
    crate::r#impl::function::launch_named_kernel_c(
        kernel_name,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        stream,
        kernel_params,
        extra,
    )
}

#[cfg(all(
    any(feature = "sifive", feature = "nvidia"),
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_launch_named_kernel(
    kernel_name: *const c_char,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32 {
    hetgpu_launch_named_kernel(
        kernel_name,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        stream,
        kernel_params,
        extra,
    )
}

// Note: cudaDriverGetVersion and cudaGetDeviceCount are now in cudart_shim.c
// to get proper version tags (@@libcudart.so.12). The C implementations
// call through to cuDriverGetVersion/cuDeviceGetCount defined below.

// Note: cuGetProcAddress/cuGetProcAddress_v2/cuGetExportTable are provided
// via cuda_function_declarations! and implemented in r#impl::driver.

// Get device handle by index using the ordinal-to-handle mapping from driver.rs
#[cfg(feature = "intel")]
fn get_device_handle_by_index(index: usize) -> Result<ze_device_handle_t> {
    crate::r#impl::driver::get_ze_handle_by_ordinal(index as i32)
}

// Fix implementation of FromCuda for ze_device_handle_t
#[cfg(feature = "intel")]
impl FromCuda<'_, *mut i32> for *mut ze_device_handle_t {
    fn from_cuda(_: &*mut i32) -> Result<Self> {
        // Simplified implementation - just a placeholder
        Ok(ptr::null_mut())
    }
}

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                // Always log unimplemented calls to help debug initialization issues
                eprintln!("[hetGPU] Unimplemented CUDA call: {}", stringify!($fn_name));
                crate::r#impl::unimplemented()
            }
        )*
    };
}

#[cfg(feature = "amd")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id).unwrap()),*).unwrap();
                Ok(())
            }
        )*
    };
}
#[cfg(feature = "intel")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                if crate::hetgpu_log_cuda_calls_enabled() {
                    eprintln!("[hetGPU] {} called", stringify!($fn_name));
                }

                // Convert arguments with error handling
                let result = (|| -> std::result::Result<_, CUerror> {
                    let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name )($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                    Ok(crate::r#impl::into_cu_result(backend_ret))
                })();

                match result {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[hetGPU] {} failed with conversion error: {:?}", stringify!($fn_name), e);
                        Err(e)
                    }
                }
            }
        )*
    };
}
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, i32> for ze_device_handle_t {
    fn from_cuda(cuda_value: &'a i32) -> Result<Self> {
        // Logic to convert i32 to ze_device_handle_t
        if *cuda_value < 0 {
            return Err(CUerror::INVALID_VALUE); // Return an error, not CUresult
        }

        // Get device handle by index
        let device_handle = get_device_handle_by_index(*cuda_value as usize)?;
        Ok(device_handle)
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, cuda_types::cuda::CUdeviceptr_v2> for cuda_types::cuda::CUdeviceptr_v2 {
    fn from_cuda(cuda_value: &'a cuda_types::cuda::CUdeviceptr_v2) -> Result<Self> {
        // Logic to validate CUdeviceptr_v2
        if unsafe { cuda_value.0 as i64 } < 0 {
            return Err(CUerror::INVALID_HANDLE); // Return an error, not CUresult
        }

        Ok(*cuda_value)
    }
}
#[cfg(feature = "amd")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id).unwrap()),*).unwrap();
                Ok(())
            }
        )*
    };
}

#[cfg(feature = "intel")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name )($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                crate::r#impl::into_cu_result(backend_ret)
            }
        )*
    };
}

#[cfg(feature = "tenstorrent")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

#[cfg(feature = "tenstorrent")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

#[cfg(all(
    feature = "tmatmul",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "nvidia")
))]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

// SIFIVE backend macros
#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                crate::r#impl::into_cu_result(backend_ret)
            }
        )*
    };
}

#[cfg(all(
    feature = "sifive",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                let backend_ret = cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                crate::r#impl::into_cu_result(backend_ret)
            }
        )*
    };
}

// NVIDIA backend macros - pass through to real libcuda.so
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)
            }
        )*
    };
}

cuda_base::cuda_function_declarations!(
    unimplemented,
    implemented
        <= [
            cuCtxCreate_v2,
            cuCtxDestroy_v2,
            cuCtxGetLimit,
            cuCtxSetCurrent,
            cuCtxSetLimit,
            cuCtxSynchronize,
            cuCtxGetCurrent,
            cuCtxGetDevice,
            cuCtxPushCurrent_v2,
            cuCtxPopCurrent_v2,
            cuDeviceComputeCapability,
            cuDeviceGet,
            cuDeviceGetAttribute,
            cuDeviceGetCount,
            cuDeviceGetLuid,
            cuDeviceGetName,
            cuDevicePrimaryCtxRelease,
            cuDevicePrimaryCtxRetain,
            cuDevicePrimaryCtxGetState,
            cuDeviceGetProperties,
            cuDeviceGetUuid,
            cuDeviceGetUuid_v2,
            cuDeviceTotalMem_v2,
            cuDriverGetVersion,
            cuFuncGetAttribute,
            cuInit,
            cuMemAlloc_v2,
            cuMemFree_v2,
            cuMemcpyDtoH_v2,
            cuMemcpyHtoD_v2,
            cuModuleGetFunction,
            cuModuleGetLoadingMode,
            cuModuleLoadData,
            cuModuleUnload,
            cuLibraryLoadData,
            cuLibraryGetKernel,
            cuKernelGetFunction,
            cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
            cuMemAddressFree,
            cuMemAddressReserve,
            cuMemCreate,
            cuMemGetAllocationGranularity,
            cuMemMap,
            cuMemRelease,
            cuMemSetAccess,
            cuMemUnmap,
            cuPointerGetAttribute,
            cuMemGetAddressRange_v2,
            cuMemsetD32_v2,
            cuMemsetD8_v2,
            // Provide custom implementations in r#impl::driver
            cuGetProcAddress,
            cuGetProcAddress_v2,
            cuGetExportTable,
            cuGetErrorString,
            cuGetErrorName
        ],
    implemented_in_function <= [cuLaunchKernel, cuLaunchKernelEx,]
);
// cuGetErrorString/cuGetErrorName are implemented via r#impl::driver
