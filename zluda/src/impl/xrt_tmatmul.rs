use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fmt;
use std::path::PathBuf;

const AXI_INSTANCE_STRIDE: u32 = 0x4000;
const MM2S_DMACR: u32 = 0x0000;
const MM2S_SA: u32 = 0x0018;
const MM2S_LENGTH: u32 = 0x0028;
const STALL: u32 = 0x1000;
const INSTRUCTION_BYTES: usize = 16;
const XRT_BO_SYNC_TO_DEVICE: i32 = 0;
const XRT_BO_SYNC_FROM_DEVICE: i32 = 1;

type Handle = *mut libc::c_void;
type Xuid = [u8; 16];

type DeviceOpenFn = unsafe extern "C" fn(u32) -> Handle;
type DeviceCloseFn = unsafe extern "C" fn(Handle) -> i32;
type DeviceLoadXclbinFileFn = unsafe extern "C" fn(Handle, *const libc::c_char) -> i32;
type DeviceGetXclbinUuidFn = unsafe extern "C" fn(Handle, *mut u8) -> i32;
type PlKernelOpenExclusiveFn =
    unsafe extern "C" fn(Handle, *const u8, *const libc::c_char) -> Handle;
type KernelCloseFn = unsafe extern "C" fn(Handle) -> i32;
type KernelArgGroupIdFn = unsafe extern "C" fn(Handle, i32) -> i32;
type KernelReadRegisterFn = unsafe extern "C" fn(Handle, u32, *mut u32) -> i32;
type KernelWriteRegisterFn = unsafe extern "C" fn(Handle, u32, u32) -> i32;
type BoAllocFn = unsafe extern "C" fn(Handle, usize, u64, u32) -> Handle;
type BoFreeFn = unsafe extern "C" fn(Handle) -> i32;
type BoAddressFn = unsafe extern "C" fn(Handle) -> u64;
type BoWriteFn = unsafe extern "C" fn(Handle, *const libc::c_void, usize, usize) -> i32;
type BoReadFn = unsafe extern "C" fn(Handle, *mut libc::c_void, usize, usize) -> i32;
type BoSyncFn = unsafe extern "C" fn(Handle, i32, usize, usize) -> i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InstanceRegisters {
    dma_control: u32,
    dma_source_lo: u32,
    dma_source_hi: u32,
    dma_length: u32,
    stall: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XrtConfig {
    xclbin: PathBuf,
    device_index: u32,
    kernel_name: String,
    instance: u32,
    memory_arg: i32,
    timeout_ms: u32,
}

impl XrtConfig {
    fn from_env() -> Result<Self, XrtTmatmulError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self, XrtTmatmulError> {
        fn parse_u32(
            name: &'static str,
            value: Option<String>,
            default: u32,
        ) -> Result<u32, XrtTmatmulError> {
            value.map_or(Ok(default), |text| {
                text.parse::<u32>()
                    .map_err(|error| XrtTmatmulError::Config(format!("{name}={text:?}: {error}")))
            })
        }

        let xclbin = lookup("HETGPU_XRT_XCLBIN")
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| XrtTmatmulError::Config("HETGPU_XRT_XCLBIN is required".to_string()))?;
        let device_index = parse_u32(
            "HETGPU_XRT_DEVICE_INDEX",
            lookup("HETGPU_XRT_DEVICE_INDEX"),
            0,
        )?;
        let instance = parse_u32("HETGPU_XRT_INSTANCE", lookup("HETGPU_XRT_INSTANCE"), 0)?;
        let timeout_ms = parse_u32(
            "HETGPU_XRT_TIMEOUT_MS",
            lookup("HETGPU_XRT_TIMEOUT_MS"),
            10_000,
        )?;
        if timeout_ms == 0 {
            return Err(XrtTmatmulError::Config(
                "HETGPU_XRT_TIMEOUT_MS must be nonzero".to_string(),
            ));
        }
        let default_memory_arg = 14u32.checked_add(instance).ok_or_else(|| {
            XrtTmatmulError::Config("DDR_PTR argument index overflow".to_string())
        })?;
        let memory_arg_u32 = parse_u32(
            "HETGPU_XRT_MEMORY_ARG",
            lookup("HETGPU_XRT_MEMORY_ARG"),
            default_memory_arg,
        )?;
        let memory_arg = i32::try_from(memory_arg_u32).map_err(|_| {
            XrtTmatmulError::Config(
                "HETGPU_XRT_MEMORY_ARG does not fit an XRT argument index".to_string(),
            )
        })?;
        let kernel_name = lookup("HETGPU_XRT_KERNEL")
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "ternip_ip".to_string());

        Ok(Self {
            xclbin: PathBuf::from(xclbin),
            device_index,
            kernel_name,
            instance,
            memory_arg,
            timeout_ms,
        })
    }
}

trait XrtOps {
    fn device_open(&self, index: u32) -> Handle;
    fn device_close(&self, device: Handle) -> i32;
    fn load_xclbin_file(&self, device: Handle, path: &CStr) -> i32;
    fn get_xclbin_uuid(&self, device: Handle, uuid: &mut Xuid) -> i32;
    fn kernel_open_exclusive(&self, device: Handle, uuid: &Xuid, name: &CStr) -> Handle;
    fn kernel_close(&self, kernel: Handle) -> i32;
    fn kernel_arg_group_id(&self, kernel: Handle, arg: i32) -> i32;
    fn kernel_read_register(&self, kernel: Handle, offset: u32, value: &mut u32) -> i32;
    fn kernel_write_register(&self, kernel: Handle, offset: u32, value: u32) -> i32;
    fn bo_alloc(&self, device: Handle, size: usize, flags: u64, group: u32) -> Handle;
    fn bo_free(&self, bo: Handle) -> i32;
    fn bo_address(&self, bo: Handle) -> u64;
    fn bo_write(&self, bo: Handle, bytes: &[u8]) -> i32;
    fn bo_read(&self, bo: Handle, bytes: &mut [u8]) -> i32;
    fn bo_sync(&self, bo: Handle, direction: i32, size: usize, offset: usize) -> i32;
}

struct RealXrt {
    library: Handle,
    device_open: DeviceOpenFn,
    device_close: DeviceCloseFn,
    device_load_xclbin_file: DeviceLoadXclbinFileFn,
    device_get_xclbin_uuid: DeviceGetXclbinUuidFn,
    pl_kernel_open_exclusive: PlKernelOpenExclusiveFn,
    kernel_close: KernelCloseFn,
    kernel_arg_group_id: KernelArgGroupIdFn,
    kernel_read_register: KernelReadRegisterFn,
    kernel_write_register: KernelWriteRegisterFn,
    bo_alloc: BoAllocFn,
    bo_free: BoFreeFn,
    bo_address: BoAddressFn,
    bo_write: BoWriteFn,
    bo_read: BoReadFn,
    bo_sync: BoSyncFn,
}

impl RealXrt {
    fn load() -> Result<Self, XrtTmatmulError> {
        let mut failures = Vec::new();
        let mut library = std::ptr::null_mut();
        for candidate in ["libxrt_coreutil.so.2", "libxrt_coreutil.so"] {
            let candidate_c = CString::new(candidate).expect("XRT library name has no NUL");
            library =
                unsafe { libc::dlopen(candidate_c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !library.is_null() {
                break;
            }
            failures.push(format!("{candidate}: {}", dl_error_message()));
        }
        if library.is_null() {
            return Err(XrtTmatmulError::DynamicLoad(failures.join("; ")));
        }

        let result = unsafe {
            Ok(Self {
                library,
                device_open: load_symbol(library, c"xrtDeviceOpen")?,
                device_close: load_symbol(library, c"xrtDeviceClose")?,
                device_load_xclbin_file: load_symbol(library, c"xrtDeviceLoadXclbinFile")?,
                device_get_xclbin_uuid: load_symbol(library, c"xrtDeviceGetXclbinUUID")?,
                pl_kernel_open_exclusive: load_symbol(library, c"xrtPLKernelOpenExclusive")?,
                kernel_close: load_symbol(library, c"xrtKernelClose")?,
                kernel_arg_group_id: load_symbol(library, c"xrtKernelArgGroupId")?,
                kernel_read_register: load_symbol(library, c"xrtKernelReadRegister")?,
                kernel_write_register: load_symbol(library, c"xrtKernelWriteRegister")?,
                bo_alloc: load_symbol(library, c"xrtBOAlloc")?,
                bo_free: load_symbol(library, c"xrtBOFree")?,
                bo_address: load_symbol(library, c"xrtBOAddress")?,
                bo_write: load_symbol(library, c"xrtBOWrite")?,
                bo_read: load_symbol(library, c"xrtBORead")?,
                bo_sync: load_symbol(library, c"xrtBOSync")?,
            })
        };
        if result.is_err() {
            unsafe {
                libc::dlclose(library);
            }
        }
        result
    }
}

impl Drop for RealXrt {
    fn drop(&mut self) {
        if !self.library.is_null() {
            unsafe {
                libc::dlclose(self.library);
            }
        }
    }
}

impl XrtOps for RealXrt {
    fn device_open(&self, index: u32) -> Handle {
        unsafe { (self.device_open)(index) }
    }

    fn device_close(&self, device: Handle) -> i32 {
        unsafe { (self.device_close)(device) }
    }

    fn load_xclbin_file(&self, device: Handle, path: &CStr) -> i32 {
        unsafe { (self.device_load_xclbin_file)(device, path.as_ptr()) }
    }

    fn get_xclbin_uuid(&self, device: Handle, uuid: &mut Xuid) -> i32 {
        unsafe { (self.device_get_xclbin_uuid)(device, uuid.as_mut_ptr()) }
    }

    fn kernel_open_exclusive(&self, device: Handle, uuid: &Xuid, name: &CStr) -> Handle {
        unsafe { (self.pl_kernel_open_exclusive)(device, uuid.as_ptr(), name.as_ptr()) }
    }

    fn kernel_close(&self, kernel: Handle) -> i32 {
        unsafe { (self.kernel_close)(kernel) }
    }

    fn kernel_arg_group_id(&self, kernel: Handle, arg: i32) -> i32 {
        unsafe { (self.kernel_arg_group_id)(kernel, arg) }
    }

    fn kernel_read_register(&self, kernel: Handle, offset: u32, value: &mut u32) -> i32 {
        unsafe { (self.kernel_read_register)(kernel, offset, value) }
    }

    fn kernel_write_register(&self, kernel: Handle, offset: u32, value: u32) -> i32 {
        unsafe { (self.kernel_write_register)(kernel, offset, value) }
    }

    fn bo_alloc(&self, device: Handle, size: usize, flags: u64, group: u32) -> Handle {
        unsafe { (self.bo_alloc)(device, size, flags, group) }
    }

    fn bo_free(&self, bo: Handle) -> i32 {
        unsafe { (self.bo_free)(bo) }
    }

    fn bo_address(&self, bo: Handle) -> u64 {
        unsafe { (self.bo_address)(bo) }
    }

    fn bo_write(&self, bo: Handle, bytes: &[u8]) -> i32 {
        unsafe { (self.bo_write)(bo, bytes.as_ptr().cast(), bytes.len(), 0) }
    }

    fn bo_read(&self, bo: Handle, bytes: &mut [u8]) -> i32 {
        unsafe { (self.bo_read)(bo, bytes.as_mut_ptr().cast(), bytes.len(), 0) }
    }

    fn bo_sync(&self, bo: Handle, direction: i32, size: usize, offset: usize) -> i32 {
        unsafe { (self.bo_sync)(bo, direction, size, offset) }
    }
}

unsafe fn load_symbol<T: Copy>(library: Handle, name: &CStr) -> Result<T, XrtTmatmulError> {
    unsafe {
        libc::dlerror();
    }
    let symbol = unsafe { libc::dlsym(library, name.as_ptr()) };
    if symbol.is_null() {
        return Err(XrtTmatmulError::DynamicLoad(format!(
            "{}: {}",
            name.to_string_lossy(),
            dl_error_message()
        )));
    }
    if std::mem::size_of::<T>() != std::mem::size_of::<Handle>() {
        return Err(XrtTmatmulError::DynamicLoad(format!(
            "{}: incompatible function pointer size",
            name.to_string_lossy()
        )));
    }
    Ok(unsafe { std::mem::transmute_copy(&symbol) })
}

fn dl_error_message() -> String {
    let error = unsafe { libc::dlerror() };
    if error.is_null() {
        "unknown dynamic loader error".to_string()
    } else {
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum XrtTmatmulError {
    Config(String),
    DynamicLoad(String),
    Xrt { operation: &'static str, code: i32 },
    NullHandle(&'static str),
    InvalidProgram(String),
    InvalidBuffer(String),
    Assemble(String),
    Timeout { timeout_ms: u32 },
}

impl fmt::Display for XrtTmatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "XRT tmatmul configuration failed: {message}"),
            Self::DynamicLoad(message) => write!(f, "XRT runtime loading failed: {message}"),
            Self::Xrt { operation, code } => {
                write!(f, "XRT {operation} failed with code {code}")
            }
            Self::NullHandle(operation) => write!(f, "XRT {operation} returned a null handle"),
            Self::InvalidProgram(message) => write!(f, "invalid tmatmul program: {message}"),
            Self::InvalidBuffer(message) => write!(f, "invalid tmatmul buffer: {message}"),
            Self::Assemble(message) => write!(f, "tmatmul assembly failed: {message}"),
            Self::Timeout { timeout_ms } => {
                write!(f, "XRT tmatmul timed out after {timeout_ms} ms")
            }
        }
    }
}

impl std::error::Error for XrtTmatmulError {}

fn instance_registers(instance: u32) -> Result<InstanceRegisters, XrtTmatmulError> {
    let base = instance.checked_mul(AXI_INSTANCE_STRIDE).ok_or_else(|| {
        XrtTmatmulError::Config(format!(
            "instance {instance} overflows the register aperture"
        ))
    })?;
    let add = |offset: u32| {
        base.checked_add(offset).ok_or_else(|| {
            XrtTmatmulError::Config(format!("instance {instance} register offset overflow"))
        })
    };

    Ok(InstanceRegisters {
        dma_control: add(MM2S_DMACR)?,
        dma_source_lo: add(MM2S_SA)?,
        dma_source_hi: add(MM2S_SA + 4)?,
        dma_length: add(MM2S_LENGTH)?,
        stall: add(STALL)?,
    })
}

fn validate_program(program: &[u8]) -> Result<(), XrtTmatmulError> {
    if program.is_empty() {
        return Err(XrtTmatmulError::InvalidProgram(
            "program is empty".to_string(),
        ));
    }
    if program.len() % INSTRUCTION_BYTES != 0 {
        return Err(XrtTmatmulError::InvalidProgram(format!(
            "{} bytes is not a multiple of {INSTRUCTION_BYTES}",
            program.len()
        )));
    }
    u32::try_from(program.len()).map_err(|_| {
        XrtTmatmulError::InvalidProgram("program length does not fit MM2S_LENGTH".to_string())
    })?;
    Ok(())
}

fn bind_labels(
    matrix_label: &str,
    input_label: &str,
    output_label: &str,
    matrix_address: u64,
    input_address: u64,
    output_address: u64,
) -> Result<HashMap<String, u64>, XrtTmatmulError> {
    if matrix_label.is_empty() || input_label.is_empty() || output_label.is_empty() {
        return Err(XrtTmatmulError::Config(
            "BO labels must not be empty".to_string(),
        ));
    }
    if matrix_label == input_label || matrix_label == output_label || input_label == output_label {
        return Err(XrtTmatmulError::Config(
            "matrix, input, and output labels must be distinct".to_string(),
        ));
    }
    if matrix_address == 0 || input_address == 0 || output_address == 0 {
        return Err(XrtTmatmulError::InvalidBuffer(
            "XRT returned a zero BO address".to_string(),
        ));
    }

    Ok(HashMap::from([
        (matrix_label.to_owned(), matrix_address),
        (input_label.to_owned(), input_address),
        (output_label.to_owned(), output_address),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_registers_match_ternary_matmul_contract() {
        assert_eq!(
            instance_registers(0).unwrap(),
            InstanceRegisters {
                dma_control: 0x0000,
                dma_source_lo: 0x0018,
                dma_source_hi: 0x001c,
                dma_length: 0x0028,
                stall: 0x1000,
            }
        );
        assert_eq!(instance_registers(2).unwrap().stall, 0x9000);
    }

    #[test]
    fn program_requires_complete_128_bit_words() {
        assert!(validate_program(&[0; 16]).is_ok());
        assert!(matches!(
            validate_program(&[]),
            Err(XrtTmatmulError::InvalidProgram(_))
        ));
        assert!(matches!(
            validate_program(&[0; 17]),
            Err(XrtTmatmulError::InvalidProgram(_))
        ));
    }

    #[test]
    fn labels_bind_to_real_bo_addresses() {
        let labels = bind_labels("PARAM_0", "PARAM_1", "PARAM_2", 0x1000, 0x2000, 0x3000).unwrap();
        assert_eq!(labels["PARAM_0"], 0x1000);
        assert_eq!(labels["PARAM_1"], 0x2000);
        assert_eq!(labels["PARAM_2"], 0x3000);
    }

    #[test]
    fn duplicate_labels_are_rejected() {
        let error = bind_labels("PARAM_0", "PARAM_0", "PARAM_2", 1, 2, 3).unwrap_err();
        assert!(error.to_string().contains("distinct"));
    }

    #[test]
    fn config_uses_repo_compatible_defaults() {
        let config = XrtConfig::from_lookup(|name| match name {
            "HETGPU_XRT_XCLBIN" => Some("/tmp/kernel.xclbin".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(config.xclbin, PathBuf::from("/tmp/kernel.xclbin"));
        assert_eq!(config.device_index, 0);
        assert_eq!(config.kernel_name, "ternip_ip");
        assert_eq!(config.instance, 0);
        assert_eq!(config.memory_arg, 14);
        assert_eq!(config.timeout_ms, 10_000);
    }

    #[test]
    fn config_selects_instance_memory_pointer_arg() {
        let config = XrtConfig::from_lookup(|name| match name {
            "HETGPU_XRT_XCLBIN" => Some("kernel.xclbin".into()),
            "HETGPU_XRT_INSTANCE" => Some("2".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(config.memory_arg, 16);
    }

    #[test]
    fn xclbin_is_required() {
        let error = XrtConfig::from_lookup(|_| None).unwrap_err();
        assert!(error.to_string().contains("HETGPU_XRT_XCLBIN"));
    }

    #[test]
    #[ignore = "requires an installed XRT userspace runtime"]
    fn installed_xrt_symbols_resolve() {
        let api = RealXrt::load().unwrap();
        assert!(!api.library.is_null());
    }
}
