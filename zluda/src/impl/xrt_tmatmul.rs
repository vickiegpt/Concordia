use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const AXI_INSTANCE_STRIDE: u32 = 0x4000;
const MM2S_DMACR: u32 = 0x0000;
const MM2S_DMASR: u32 = 0x0004;
const MM2S_SA: u32 = 0x0018;
const MM2S_LENGTH: u32 = 0x0028;
const STALL: u32 = 0x1000;
const RESET: u32 = 0x2000;
const DMACR_RESET: u32 = 1 << 2;
const DMASR_HALTED: u32 = 1 << 0;
const DMASR_IDLE: u32 = 1 << 1;
const INSTRUCTION_BYTES: usize = 16;
const XRT_BO_SYNC_TO_DEVICE: i32 = 0;
const XRT_BO_SYNC_FROM_DEVICE: i32 = 1;
const AU250_DIM: usize = 1024;
const AU250_MATRIX_BYTES: usize = AU250_DIM * AU250_DIM / 4;
const AU250_TMATMUL_ASSEMBLY: &str = "ldv v0, PARAM_INPUT\ntmatmul_import v0\ntmatmul_go PARAM_MATRIX\ntmatmul_export v1\nsv v1, PARAM_OUTPUT\nstall\n";

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
type XclOpenFn = unsafe extern "C" fn(u32, *const libc::c_char, i32) -> Handle;
type XclCloseFn = unsafe extern "C" fn(Handle);
type XclIpNameToIndexFn = unsafe extern "C" fn(Handle, *const libc::c_char) -> i32;
type XclOpenContextFn = unsafe extern "C" fn(Handle, *const u8, u32, bool) -> i32;
type XclCloseContextFn = unsafe extern "C" fn(Handle, *const u8, u32) -> i32;
type XclRegReadFn = unsafe extern "C" fn(Handle, u32, u32, *mut u32) -> i32;
type XclRegWriteFn = unsafe extern "C" fn(Handle, u32, u32, u32) -> i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InstanceRegisters {
    dma_control: u32,
    dma_source_lo: u32,
    dma_source_hi: u32,
    dma_length: u32,
    stall: u32,
    reset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XrtConfig {
    xclbin: PathBuf,
    device_index: u32,
    kernel_name: String,
    ip_name: Option<String>,
    instance: u32,
    memory_arg: i32,
    memory_group: Option<u32>,
    num_vector_registers: u8,
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
        let ip_name = lookup("HETGPU_XRT_IP_NAME").filter(|value| !value.trim().is_empty());
        let memory_group = lookup("HETGPU_XRT_MEMORY_GROUP")
            .map(|text| {
                text.parse::<u32>().map_err(|error| {
                    XrtTmatmulError::Config(format!("HETGPU_XRT_MEMORY_GROUP={text:?}: {error}"))
                })
            })
            .transpose()?;
        if ip_name.is_some() && memory_group.is_none() {
            return Err(XrtTmatmulError::Config(
                "HETGPU_XRT_MEMORY_GROUP is required when HETGPU_XRT_IP_NAME is set".to_string(),
            ));
        }
        let num_vector_registers_u32 = parse_u32(
            "HETGPU_XRT_NUM_VECTOR_REGISTERS",
            lookup("HETGPU_XRT_NUM_VECTOR_REGISTERS"),
            8,
        )?;
        let num_vector_registers = u8::try_from(num_vector_registers_u32).map_err(|_| {
            XrtTmatmulError::Config(
                "HETGPU_XRT_NUM_VECTOR_REGISTERS does not fit in u8".to_string(),
            )
        })?;

        Ok(Self {
            xclbin: PathBuf::from(xclbin),
            device_index,
            kernel_name,
            ip_name,
            instance,
            memory_arg,
            memory_group,
            num_vector_registers,
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
    fn xcl_open(&self, index: u32) -> Handle;
    fn xcl_close(&self, device: Handle);
    fn xcl_ip_name_to_index(&self, device: Handle, name: &CStr) -> i32;
    fn xcl_open_context(&self, device: Handle, uuid: &Xuid, index: u32, shared: bool) -> i32;
    fn xcl_close_context(&self, device: Handle, uuid: &Xuid, index: u32) -> i32;
    fn xcl_reg_read(&self, device: Handle, index: u32, offset: u32, value: &mut u32) -> i32;
    fn xcl_reg_write(&self, device: Handle, index: u32, offset: u32, value: u32) -> i32;
    fn bo_alloc(&self, device: Handle, size: usize, flags: u64, group: u32) -> Handle;
    fn bo_free(&self, bo: Handle) -> i32;
    fn bo_address(&self, bo: Handle) -> u64;
    fn bo_write(&self, bo: Handle, bytes: &[u8]) -> i32;
    fn bo_read(&self, bo: Handle, bytes: &mut [u8]) -> i32;
    fn bo_sync(&self, bo: Handle, direction: i32, size: usize, offset: usize) -> i32;
}

struct NativeIpApi {
    library: Handle,
    open: XclOpenFn,
    close: XclCloseFn,
    ip_name_to_index: XclIpNameToIndexFn,
    open_context: XclOpenContextFn,
    close_context: XclCloseContextFn,
    reg_read: XclRegReadFn,
    reg_write: XclRegWriteFn,
}

impl NativeIpApi {
    fn load() -> Result<Self, XrtTmatmulError> {
        let library = open_library(&["libxrt_core.so.2", "libxrt_core.so"])?;
        let result: Result<Self, XrtTmatmulError> = unsafe {
            Ok(Self {
                library,
                open: load_symbol(library, c"xclOpen")?,
                close: load_symbol(library, c"xclClose")?,
                ip_name_to_index: load_symbol(library, c"xclIPName2Index")?,
                open_context: load_symbol(library, c"xclOpenContext")?,
                close_context: load_symbol(library, c"xclCloseContext")?,
                reg_read: load_symbol(library, c"xclRegRead")?,
                reg_write: load_symbol(library, c"xclRegWrite")?,
            })
        };
        if result.is_err() {
            unsafe {
                libc::dlclose(library);
            }
        }
        result.map_err(|error| {
            XrtTmatmulError::DynamicLoad(format!(
                "AP_CTRL_NONE requires the app215-compatible legacy xcl register API: {error}"
            ))
        })
    }
}

impl Drop for NativeIpApi {
    fn drop(&mut self) {
        if !self.library.is_null() {
            unsafe {
                libc::dlclose(self.library);
            }
        }
    }
}

struct RealXrt {
    library: Handle,
    native_ip: Option<NativeIpApi>,
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
    fn load(needs_native_ip: bool) -> Result<Self, XrtTmatmulError> {
        let library = open_library(&["libxrt_coreutil.so.2", "libxrt_coreutil.so"])?;
        let native_ip = match needs_native_ip.then(NativeIpApi::load).transpose() {
            Ok(api) => api,
            Err(error) => {
                unsafe {
                    libc::dlclose(library);
                }
                return Err(error);
            }
        };

        let result = unsafe {
            Ok(Self {
                library,
                native_ip,
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

    fn native_ip(&self) -> &NativeIpApi {
        self.native_ip
            .as_ref()
            .expect("native-IP API must be loaded before native-IP operations")
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

    fn xcl_open(&self, index: u32) -> Handle {
        unsafe { (self.native_ip().open)(index, std::ptr::null(), 0) }
    }

    fn xcl_close(&self, device: Handle) {
        unsafe { (self.native_ip().close)(device) }
    }

    fn xcl_ip_name_to_index(&self, device: Handle, name: &CStr) -> i32 {
        unsafe { (self.native_ip().ip_name_to_index)(device, name.as_ptr()) }
    }

    fn xcl_open_context(&self, device: Handle, uuid: &Xuid, index: u32, shared: bool) -> i32 {
        unsafe { (self.native_ip().open_context)(device, uuid.as_ptr(), index, shared) }
    }

    fn xcl_close_context(&self, device: Handle, uuid: &Xuid, index: u32) -> i32 {
        unsafe { (self.native_ip().close_context)(device, uuid.as_ptr(), index) }
    }

    fn xcl_reg_read(&self, device: Handle, index: u32, offset: u32, value: &mut u32) -> i32 {
        unsafe { (self.native_ip().reg_read)(device, index, offset, value) }
    }

    fn xcl_reg_write(&self, device: Handle, index: u32, offset: u32, value: u32) -> i32 {
        unsafe { (self.native_ip().reg_write)(device, index, offset, value) }
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

fn open_library(candidates: &[&str]) -> Result<Handle, XrtTmatmulError> {
    let mut failures = Vec::new();
    for candidate in candidates {
        let candidate_c = CString::new(*candidate).expect("XRT library name has no NUL");
        let library =
            unsafe { libc::dlopen(candidate_c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if !library.is_null() {
            return Ok(library);
        }
        failures.push(format!("{candidate}: {}", dl_error_message()));
    }
    Err(XrtTmatmulError::DynamicLoad(failures.join("; ")))
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
    Quiesce { primary: String, cleanup: String },
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
            Self::Quiesce { primary, cleanup } => write!(
                f,
                "XRT tmatmul failed ({primary}) and device quiescence could not be confirmed ({cleanup}); handles were retained to protect live BOs"
            ),
        }
    }
}

impl std::error::Error for XrtTmatmulError {}

pub(crate) struct XrtTmatmulRequest<'a> {
    pub(crate) assembly: &'a str,
    pub(crate) matrix_label: &'a str,
    pub(crate) input_label: &'a str,
    pub(crate) output_label: &'a str,
    pub(crate) matrix: &'a [u8],
    pub(crate) input: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct XrtTmatmulStatus {
    pub(crate) stall_code: u32,
    pub(crate) program_bytes: usize,
    pub(crate) matrix_address: u64,
    pub(crate) input_address: u64,
    pub(crate) output_address: u64,
    pub(crate) program_address: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct XrtCuTarget {
    pub(crate) ip_name: String,
    pub(crate) memory_group: u32,
    pub(crate) lanes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XrtWaveJob {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) matrix: Arc<[u8]>,
    pub(crate) input: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XrtWaveCompletion {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) stall_code: u32,
    pub(crate) output: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XrtPoolConfig {
    xclbin: PathBuf,
    device_index: u32,
    targets: Vec<XrtCuTarget>,
    num_vector_registers: u8,
    timeout_ms: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct XrtCuTable {
    version: u32,
    cus: Vec<XrtCuTarget>,
}

impl XrtPoolConfig {
    fn maxcores_targets() -> Vec<XrtCuTarget> {
        vec![
            XrtCuTarget {
                ip_name: "ternip_big:ternip_big_1".to_string(),
                memory_group: 0,
                lanes: 9,
            },
            XrtCuTarget {
                ip_name: "ternip_big:ternip_big_2".to_string(),
                memory_group: 3,
                lanes: 9,
            },
            XrtCuTarget {
                ip_name: "ternip_big:ternip_big_3".to_string(),
                memory_group: 2,
                lanes: 9,
            },
            XrtCuTarget {
                ip_name: "ternip_small:ternip_small_1".to_string(),
                memory_group: 1,
                lanes: 6,
            },
        ]
    }

    fn parse_cu_table(json: &str) -> Result<Vec<XrtCuTarget>, XrtTmatmulError> {
        let table: XrtCuTable = serde_json::from_str(json).map_err(|error| {
            XrtTmatmulError::Config(format!("HETGPU_XRT_CU_CONFIG is not valid JSON: {error}"))
        })?;
        if table.version != 1 {
            return Err(XrtTmatmulError::Config(format!(
                "unsupported HETGPU_XRT_CU_CONFIG version {}, expected 1",
                table.version
            )));
        }
        Self::validate_targets(&table.cus)?;
        Ok(table.cus)
    }

    fn validate_targets(targets: &[XrtCuTarget]) -> Result<(), XrtTmatmulError> {
        if targets.is_empty() || targets.len() > 4 {
            return Err(XrtTmatmulError::Config(format!(
                "XRT CU table must contain between 1 and 4 CUs, got {}",
                targets.len()
            )));
        }
        let mut names = HashSet::new();
        let mut groups = HashSet::new();
        for target in targets {
            if target.ip_name.trim().is_empty() || target.ip_name.as_bytes().contains(&0) {
                return Err(XrtTmatmulError::Config(
                    "XRT CU IP names must not be empty or contain NUL".to_string(),
                ));
            }
            if !names.insert(target.ip_name.as_str()) {
                return Err(XrtTmatmulError::Config(format!(
                    "duplicate XRT CU IP name {:?}",
                    target.ip_name
                )));
            }
            if !groups.insert(target.memory_group) {
                return Err(XrtTmatmulError::Config(format!(
                    "duplicate XRT CU memory group {}",
                    target.memory_group
                )));
            }
            if !matches!(target.lanes, 6 | 9) {
                return Err(XrtTmatmulError::Config(format!(
                    "XRT CU {:?} has {} lanes; expected 6 or 9",
                    target.ip_name, target.lanes
                )));
            }
        }
        Ok(())
    }

    fn from_env() -> Result<Self, XrtTmatmulError> {
        fn parse_u32(name: &'static str, default: u32) -> Result<u32, XrtTmatmulError> {
            std::env::var(name).map_or(Ok(default), |text| {
                text.parse::<u32>()
                    .map_err(|error| XrtTmatmulError::Config(format!("{name}={text:?}: {error}")))
            })
        }

        let xclbin = std::env::var("HETGPU_XRT_XCLBIN")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| XrtTmatmulError::Config("HETGPU_XRT_XCLBIN is required".to_string()))?;
        let device_index = parse_u32("HETGPU_XRT_DEVICE_INDEX", 0)?;
        let timeout_ms = parse_u32("HETGPU_XRT_TIMEOUT_MS", 10_000)?;
        if timeout_ms == 0 {
            return Err(XrtTmatmulError::Config(
                "HETGPU_XRT_TIMEOUT_MS must be nonzero".to_string(),
            ));
        }
        let num_vector_registers = u8::try_from(parse_u32("HETGPU_XRT_NUM_VECTOR_REGISTERS", 4)?)
            .map_err(|_| {
            XrtTmatmulError::Config(
                "HETGPU_XRT_NUM_VECTOR_REGISTERS does not fit in u8".to_string(),
            )
        })?;
        let targets = match std::env::var("HETGPU_XRT_CU_CONFIG") {
            Ok(json) if !json.trim().is_empty() => Self::parse_cu_table(&json)?,
            _ => Self::maxcores_targets(),
        };
        Self::validate_targets(&targets)?;
        Ok(Self {
            xclbin: PathBuf::from(xclbin),
            device_index,
            targets,
            num_vector_registers,
            timeout_ms,
        })
    }
}

fn expected_vector_bytes(lanes: usize) -> usize {
    AU250_DIM * lanes * 2
}

fn validate_wave_job(target: &XrtCuTarget, job: &XrtWaveJob) -> Result<(), XrtTmatmulError> {
    if job.matrix.len() != AU250_MATRIX_BYTES {
        return Err(XrtTmatmulError::InvalidBuffer(format!(
            "matrix has {} bytes, expected {AU250_MATRIX_BYTES}",
            job.matrix.len()
        )));
    }
    let expected = expected_vector_bytes(target.lanes);
    if job.input.len() != expected {
        return Err(XrtTmatmulError::InvalidBuffer(format!(
            "input has {} bytes, expected {expected}",
            job.input.len()
        )));
    }
    Ok(())
}

struct ReusableCu {
    target: XrtCuTarget,
    ip_device: Handle,
    ip_index: u32,
    matrix_bo: Handle,
    input_bo: Handle,
    output_bo: Handle,
    program_bo: Handle,
    program_address: u64,
    program_bytes: usize,
    release_handles: bool,
}

struct Pool<O: XrtOps> {
    ops: O,
    device: Handle,
    uuid: Xuid,
    cus: Vec<ReusableCu>,
    timeout_ms: u32,
    poisoned: bool,
    release_device: bool,
}

pub(crate) struct XrtTmatmulPool {
    inner: Pool<RealXrt>,
}

// SAFETY: every raw XRT handle is exclusively owned by this pool, all access is
// serialized by the IQ1_S executor's process-global mutex, and destruction is
// performed only by the owning pool. Raw-handle owners deliberately are not Sync.
unsafe impl Send for XrtTmatmulPool {}

impl XrtTmatmulPool {
    pub(crate) fn open_from_env() -> Result<Self, XrtTmatmulError> {
        let config = XrtPoolConfig::from_env()?;
        let ops = RealXrt::load(true)?;
        Ok(Self {
            inner: Pool::open_with_ops(ops, config)?,
        })
    }

    pub(crate) fn lane_capacities(&self) -> Vec<usize> {
        self.inner.cus.iter().map(|cu| cu.target.lanes).collect()
    }

    pub(crate) fn run_wave(
        &mut self,
        jobs: Vec<XrtWaveJob>,
    ) -> Result<Vec<XrtWaveCompletion>, XrtTmatmulError> {
        self.inner.run_wave(jobs)
    }
}

impl<O: XrtOps> Pool<O> {
    fn open_with_ops(ops: O, config: XrtPoolConfig) -> Result<Self, XrtTmatmulError> {
        XrtPoolConfig::validate_targets(&config.targets)?;
        let device = ops.device_open(config.device_index);
        if device.is_null() {
            return Err(XrtTmatmulError::NullHandle("xrtDeviceOpen"));
        }
        let mut pool = Self {
            ops,
            device,
            uuid: [0; 16],
            cus: Vec::with_capacity(config.targets.len()),
            timeout_ms: config.timeout_ms,
            poisoned: false,
            release_device: true,
        };

        let xclbin = config.xclbin.to_str().ok_or_else(|| {
            XrtTmatmulError::Config("HETGPU_XRT_XCLBIN must be valid UTF-8".to_string())
        })?;
        let xclbin = CString::new(xclbin).map_err(|_| {
            XrtTmatmulError::Config("HETGPU_XRT_XCLBIN contains a NUL byte".to_string())
        })?;
        check_xrt(
            "xrtDeviceLoadXclbinFile",
            pool.ops.load_xclbin_file(pool.device, &xclbin),
        )?;
        check_xrt(
            "xrtDeviceGetXclbinUUID",
            pool.ops.get_xclbin_uuid(pool.device, &mut pool.uuid),
        )?;

        for target in config.targets {
            let ip_device = pool.ops.xcl_open(config.device_index);
            if ip_device.is_null() {
                return Err(XrtTmatmulError::NullHandle("xclOpen"));
            }
            let ip_name = CString::new(target.ip_name.as_str()).map_err(|_| {
                XrtTmatmulError::Config("XRT CU IP name contains a NUL byte".to_string())
            })?;
            let raw_index = pool.ops.xcl_ip_name_to_index(ip_device, &ip_name);
            if raw_index < 0 {
                pool.ops.xcl_close(ip_device);
                return Err(XrtTmatmulError::Xrt {
                    operation: "xclIPName2Index",
                    code: raw_index,
                });
            }
            let ip_index = raw_index as u32;
            if let Err(error) = check_xrt(
                "xclOpenContext",
                pool.ops
                    .xcl_open_context(ip_device, &pool.uuid, ip_index, false),
            ) {
                pool.ops.xcl_close(ip_device);
                return Err(error);
            }

            pool.cus.push(ReusableCu {
                target,
                ip_device,
                ip_index,
                matrix_bo: std::ptr::null_mut(),
                input_bo: std::ptr::null_mut(),
                output_bo: std::ptr::null_mut(),
                program_bo: std::ptr::null_mut(),
                program_address: 0,
                program_bytes: 0,
                release_handles: true,
            });
            let cu_index = pool.cus.len() - 1;
            let group = pool.cus[cu_index].target.memory_group;
            let vector_bytes = expected_vector_bytes(pool.cus[cu_index].target.lanes);

            // This ordering is the persistent form of the four-BO ABI.
            let matrix_bo = pool.allocate_bo(AU250_MATRIX_BYTES, group, "xrtBOAlloc(matrix)")?;
            pool.cus[cu_index].matrix_bo = matrix_bo;
            let input_bo = pool.allocate_bo(vector_bytes, group, "xrtBOAlloc(input)")?;
            pool.cus[cu_index].input_bo = input_bo;
            let output_bo = pool.allocate_bo(vector_bytes, group, "xrtBOAlloc(output)")?;
            pool.cus[cu_index].output_bo = output_bo;

            let matrix_address = pool.bo_address(matrix_bo, "matrix")?;
            let input_address = pool.bo_address(input_bo, "input")?;
            let output_address = pool.bo_address(output_bo, "output")?;
            let labels = bind_labels(
                "PARAM_MATRIX",
                "PARAM_INPUT",
                "PARAM_OUTPUT",
                matrix_address,
                input_address,
                output_address,
            )?;
            let replay_safe_program =
                super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
                    AU250_TMATMUL_ASSEMBLY,
                    &labels,
                    config.num_vector_registers,
                )
                .map_err(|error| XrtTmatmulError::Assemble(error.to_string()))?;
            let program = compact_xrt_program(&replay_safe_program)?;
            validate_program(&program)?;
            let program_bo = pool.allocate_bo(program.len(), group, "xrtBOAlloc(program)")?;
            pool.cus[cu_index].program_bo = program_bo;
            pool.cus[cu_index].program_address = pool.bo_address(program_bo, "program")?;
            pool.cus[cu_index].program_bytes = program.len();
            bo_write_and_sync(&pool.ops, program_bo, &program, "program")?;
        }

        Ok(pool)
    }

    fn allocate_bo(
        &self,
        size: usize,
        group: u32,
        operation: &'static str,
    ) -> Result<Handle, XrtTmatmulError> {
        let bo = self.ops.bo_alloc(self.device, size, 0, group);
        if bo.is_null() {
            Err(XrtTmatmulError::NullHandle(operation))
        } else {
            Ok(bo)
        }
    }

    fn bo_address(&self, bo: Handle, name: &'static str) -> Result<u64, XrtTmatmulError> {
        let address = self.ops.bo_address(bo);
        if address == 0 || address == u64::MAX {
            Err(XrtTmatmulError::InvalidBuffer(format!(
                "xrtBOAddress returned 0x{address:016x} for {name}"
            )))
        } else {
            Ok(address)
        }
    }

    fn register_read(&self, cu_index: usize, offset: u32) -> Result<u32, XrtTmatmulError> {
        let cu = &self.cus[cu_index];
        let mut value = 0;
        check_xrt(
            "xclRegRead",
            self.ops
                .xcl_reg_read(cu.ip_device, cu.ip_index, offset, &mut value),
        )?;
        Ok(value)
    }

    fn register_write(
        &self,
        cu_index: usize,
        offset: u32,
        value: u32,
    ) -> Result<(), XrtTmatmulError> {
        let cu = &self.cus[cu_index];
        check_xrt(
            "xclRegWrite",
            self.ops
                .xcl_reg_write(cu.ip_device, cu.ip_index, offset, value),
        )
    }

    fn run_wave(
        &mut self,
        jobs: Vec<XrtWaveJob>,
    ) -> Result<Vec<XrtWaveCompletion>, XrtTmatmulError> {
        if self.poisoned {
            return Err(XrtTmatmulError::Config(
                "persistent XRT pool is poisoned after failed quiescence".to_string(),
            ));
        }
        let mut seen_cus = HashSet::new();
        let mut seen_requests = HashSet::new();
        for job in &jobs {
            let target = self.cus.get(job.cu_index).ok_or_else(|| {
                XrtTmatmulError::Config(format!(
                    "wave job {} selects missing CU {}",
                    job.request_id, job.cu_index
                ))
            })?;
            if !seen_cus.insert(job.cu_index) {
                return Err(XrtTmatmulError::Config(format!(
                    "wave contains more than one job for CU {}",
                    job.cu_index
                )));
            }
            if !seen_requests.insert(job.request_id) {
                return Err(XrtTmatmulError::Config(format!(
                    "wave contains duplicate request id {}",
                    job.request_id
                )));
            }
            validate_wave_job(&target.target, job)?;
        }

        for job in &jobs {
            let cu = &self.cus[job.cu_index];
            bo_write_and_sync(&self.ops, cu.matrix_bo, &job.matrix, "matrix")?;
            bo_write_and_sync(&self.ops, cu.input_bo, &job.input, "input")?;
        }

        let registers = instance_registers(0)?;
        let mut launched = Vec::with_capacity(jobs.len());
        for job in &jobs {
            launched.push(job.cu_index);
            let cu = &self.cus[job.cu_index];
            let start_result = (|| {
                self.register_write(job.cu_index, registers.reset, 0)?;
                self.register_write(job.cu_index, registers.dma_control, 1)?;
                self.register_write(
                    job.cu_index,
                    registers.dma_source_lo,
                    cu.program_address as u32,
                )?;
                self.register_write(
                    job.cu_index,
                    registers.dma_source_hi,
                    (cu.program_address >> 32) as u32,
                )?;
                self.register_write(job.cu_index, registers.dma_length, cu.program_bytes as u32)
            })();
            if let Err(error) = start_result {
                return self.fail_wave(error, &launched);
            }
        }

        let deadline = Instant::now() + Duration::from_millis(u64::from(self.timeout_ms));
        let mut stalls = vec![None; jobs.len()];
        let mut pending = jobs.len();
        while pending != 0 {
            for (job_index, job) in jobs.iter().enumerate() {
                if stalls[job_index].is_some() {
                    continue;
                }
                match self.register_read(job.cu_index, registers.stall) {
                    Ok(0) => {}
                    Ok(value) => {
                        stalls[job_index] = Some(value);
                        pending -= 1;
                    }
                    Err(error) => return self.fail_wave(error, &launched),
                }
            }
            if pending != 0 {
                if Instant::now() >= deadline {
                    return self.fail_wave(
                        XrtTmatmulError::Timeout {
                            timeout_ms: self.timeout_ms,
                        },
                        &launched,
                    );
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        let mut completions = Vec::with_capacity(jobs.len());
        for (job_index, job) in jobs.iter().enumerate() {
            let cu = &self.cus[job.cu_index];
            let mut output = vec![0u8; expected_vector_bytes(cu.target.lanes)];
            let finish_result = (|| {
                check_xrt(
                    "xrtBOSync(output, from-device)",
                    self.ops
                        .bo_sync(cu.output_bo, XRT_BO_SYNC_FROM_DEVICE, output.len(), 0),
                )?;
                check_xrt(
                    "xrtBORead(output)",
                    self.ops.bo_read(cu.output_bo, &mut output),
                )?;
                self.register_write(job.cu_index, registers.stall, 1)
            })();
            if let Err(error) = finish_result {
                return self.fail_wave(error, &launched);
            }
            completions.push(XrtWaveCompletion {
                request_id: job.request_id,
                cu_index: job.cu_index,
                stall_code: stalls[job_index].expect("all wave jobs completed"),
                output,
            });
        }
        Ok(completions)
    }

    fn fail_wave<T>(
        &mut self,
        primary: XrtTmatmulError,
        launched: &[usize],
    ) -> Result<T, XrtTmatmulError> {
        let mut cleanup_errors = Vec::new();
        let mut seen = HashSet::new();
        for &cu_index in launched {
            if seen.insert(cu_index) {
                if let Err(error) = self.quiesce_cu(cu_index) {
                    self.cus[cu_index].release_handles = false;
                    self.release_device = false;
                    cleanup_errors.push(format!("CU {cu_index}: {error}"));
                }
            }
        }
        if cleanup_errors.is_empty() {
            Err(primary)
        } else {
            self.poisoned = true;
            Err(XrtTmatmulError::Quiesce {
                primary: primary.to_string(),
                cleanup: cleanup_errors.join("; "),
            })
        }
    }

    fn quiesce_cu(&self, cu_index: usize) -> Result<(), XrtTmatmulError> {
        let registers = instance_registers(0)?;
        self.register_write(cu_index, registers.reset, 1)?;
        self.register_write(cu_index, registers.dma_control, DMACR_RESET)?;
        let timeout_ms = self.timeout_ms.max(100);
        let deadline = Instant::now() + Duration::from_millis(u64::from(timeout_ms));
        loop {
            let control = self.register_read(cu_index, registers.dma_control)?;
            if control & DMACR_RESET == 0 {
                let status = self.register_read(cu_index, registers.dma_control + MM2S_DMASR)?;
                if status & (DMASR_HALTED | DMASR_IDLE) != 0 {
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err(XrtTmatmulError::Timeout { timeout_ms });
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl<O: XrtOps> Drop for Pool<O> {
    fn drop(&mut self) {
        for cu in self.cus.iter_mut().rev() {
            if !cu.release_handles {
                continue;
            }
            for bo in [cu.program_bo, cu.output_bo, cu.input_bo, cu.matrix_bo] {
                if !bo.is_null() {
                    let _ = self.ops.bo_free(bo);
                }
            }
            let _ = self
                .ops
                .xcl_close_context(cu.ip_device, &self.uuid, cu.ip_index);
            if !cu.ip_device.is_null() {
                self.ops.xcl_close(cu.ip_device);
            }
        }
        if self.release_device && !self.device.is_null() {
            let _ = self.ops.device_close(self.device);
        }
    }
}

pub(crate) fn submit_xrt_tmatmul(
    request: XrtTmatmulRequest<'_>,
    output: &mut [u8],
) -> Result<XrtTmatmulStatus, XrtTmatmulError> {
    let config = XrtConfig::from_env()?;
    let xrt = RealXrt::load(config.ip_name.is_some())?;
    submit_with_ops(&xrt, &config, request, output)
}

struct Session<'a, O: XrtOps> {
    ops: &'a O,
    device: Handle,
    kernel: Handle,
    ip_device: Handle,
    ip_index: Option<u32>,
    uuid: Xuid,
    bos: Vec<Handle>,
    release_handles: bool,
}

impl<'a, O: XrtOps> Session<'a, O> {
    fn open(ops: &'a O, config: &XrtConfig) -> Result<Self, XrtTmatmulError> {
        let device = ops.device_open(config.device_index);
        if device.is_null() {
            return Err(XrtTmatmulError::NullHandle("xrtDeviceOpen"));
        }
        let mut session = Self {
            ops,
            device,
            kernel: std::ptr::null_mut(),
            ip_device: std::ptr::null_mut(),
            ip_index: None,
            uuid: [0; 16],
            bos: Vec::new(),
            release_handles: true,
        };

        let xclbin = config.xclbin.to_str().ok_or_else(|| {
            XrtTmatmulError::Config("HETGPU_XRT_XCLBIN must be valid UTF-8".to_string())
        })?;
        let xclbin = CString::new(xclbin).map_err(|_| {
            XrtTmatmulError::Config("HETGPU_XRT_XCLBIN contains a NUL byte".to_string())
        })?;
        check_xrt(
            "xrtDeviceLoadXclbinFile",
            ops.load_xclbin_file(device, &xclbin),
        )?;

        check_xrt(
            "xrtDeviceGetXclbinUUID",
            ops.get_xclbin_uuid(device, &mut session.uuid),
        )?;
        if let Some(ip_name) = config.ip_name.as_deref() {
            session.ip_device = ops.xcl_open(config.device_index);
            if session.ip_device.is_null() {
                return Err(XrtTmatmulError::NullHandle("xclOpen"));
            }
            let ip_name = CString::new(ip_name).map_err(|_| {
                XrtTmatmulError::Config("HETGPU_XRT_IP_NAME contains a NUL byte".to_string())
            })?;
            let ip_index = ops.xcl_ip_name_to_index(session.ip_device, &ip_name);
            if ip_index < 0 {
                return Err(XrtTmatmulError::Xrt {
                    operation: "xclIPName2Index",
                    code: ip_index,
                });
            }
            let ip_index = ip_index as u32;
            check_xrt(
                "xclOpenContext",
                ops.xcl_open_context(session.ip_device, &session.uuid, ip_index, false),
            )?;
            session.ip_index = Some(ip_index);
        } else {
            let kernel_name = CString::new(config.kernel_name.as_str()).map_err(|_| {
                XrtTmatmulError::Config("HETGPU_XRT_KERNEL contains a NUL byte".to_string())
            })?;
            session.kernel = ops.kernel_open_exclusive(device, &session.uuid, &kernel_name);
            if session.kernel.is_null() {
                return Err(XrtTmatmulError::NullHandle("xrtPLKernelOpenExclusive"));
            }
        }
        Ok(session)
    }

    fn allocate_bo(
        &mut self,
        size: usize,
        group: u32,
        name: &'static str,
    ) -> Result<Handle, XrtTmatmulError> {
        let bo = self.ops.bo_alloc(self.device, size, 0, group);
        if bo.is_null() {
            return Err(XrtTmatmulError::NullHandle(name));
        }
        self.bos.push(bo);
        Ok(bo)
    }

    fn bo_address(&self, bo: Handle, name: &'static str) -> Result<u64, XrtTmatmulError> {
        let address = self.ops.bo_address(bo);
        if address == 0 || address == u64::MAX {
            return Err(XrtTmatmulError::InvalidBuffer(format!(
                "xrtBOAddress returned 0x{address:016x} for {name}"
            )));
        }
        Ok(address)
    }

    fn register_read(
        &self,
        offset: u32,
        value: &mut u32,
        kernel_operation: &'static str,
        ip_operation: &'static str,
    ) -> Result<(), XrtTmatmulError> {
        if let Some(index) = self.ip_index {
            check_xrt(
                ip_operation,
                self.ops.xcl_reg_read(self.ip_device, index, offset, value),
            )
        } else {
            check_xrt(
                kernel_operation,
                self.ops.kernel_read_register(self.kernel, offset, value),
            )
        }
    }

    fn register_write(
        &self,
        offset: u32,
        value: u32,
        kernel_operation: &'static str,
        ip_operation: &'static str,
    ) -> Result<(), XrtTmatmulError> {
        if let Some(index) = self.ip_index {
            check_xrt(
                ip_operation,
                self.ops.xcl_reg_write(self.ip_device, index, offset, value),
            )
        } else {
            check_xrt(
                kernel_operation,
                self.ops.kernel_write_register(self.kernel, offset, value),
            )
        }
    }

    fn quiesce_after_launch(
        &mut self,
        registers: InstanceRegisters,
        timeout_ms: u32,
    ) -> Result<(), XrtTmatmulError> {
        // Until reset completion is observed, retaining the BO and device handles is safer
        // than allowing an independently running engine to access released storage.
        self.release_handles = false;
        self.register_write(
            registers.reset,
            1,
            "xrtKernelWriteRegister(RESET quiesce)",
            "xclRegWrite(RESET quiesce)",
        )?;
        self.register_write(
            registers.dma_control,
            DMACR_RESET,
            "xrtKernelWriteRegister(MM2S_DMACR reset)",
            "xclRegWrite(MM2S_DMACR reset)",
        )?;

        let deadline = Instant::now() + Duration::from_millis(u64::from(timeout_ms.max(100)));
        loop {
            let mut control = 0;
            self.register_read(
                registers.dma_control,
                &mut control,
                "xrtKernelReadRegister(MM2S_DMACR reset)",
                "xclRegRead(MM2S_DMACR reset)",
            )?;
            if control & DMACR_RESET == 0 {
                let mut status = 0;
                self.register_read(
                    registers.dma_control + MM2S_DMASR,
                    &mut status,
                    "xrtKernelReadRegister(MM2S_DMASR quiesce)",
                    "xclRegRead(MM2S_DMASR quiesce)",
                )?;
                if status & (DMASR_HALTED | DMASR_IDLE) != 0 {
                    self.release_handles = true;
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err(XrtTmatmulError::Timeout {
                    timeout_ms: timeout_ms.max(100),
                });
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl<O: XrtOps> Drop for Session<'_, O> {
    fn drop(&mut self) {
        if !self.release_handles {
            // The process/driver owns final recovery. Do not release or reuse addresses that
            // an engine may still be accessing when quiescence could not be confirmed.
            self.bos.clear();
            self.kernel = std::ptr::null_mut();
            self.ip_index = None;
            self.ip_device = std::ptr::null_mut();
            self.device = std::ptr::null_mut();
            return;
        }
        while let Some(bo) = self.bos.pop() {
            let _ = self.ops.bo_free(bo);
        }
        if !self.kernel.is_null() {
            let _ = self.ops.kernel_close(self.kernel);
        }
        if let Some(index) = self.ip_index {
            let _ = self
                .ops
                .xcl_close_context(self.ip_device, &self.uuid, index);
        }
        if !self.ip_device.is_null() {
            self.ops.xcl_close(self.ip_device);
        }
        if !self.device.is_null() {
            let _ = self.ops.device_close(self.device);
        }
    }
}

fn submit_with_ops<O: XrtOps>(
    ops: &O,
    config: &XrtConfig,
    request: XrtTmatmulRequest<'_>,
    output: &mut [u8],
) -> Result<XrtTmatmulStatus, XrtTmatmulError> {
    if request.matrix.is_empty() || request.input.is_empty() || output.is_empty() {
        return Err(XrtTmatmulError::InvalidBuffer(
            "matrix, input, and output BOs must be nonempty".to_string(),
        ));
    }

    let registers = instance_registers(config.instance)?;
    let mut session = Session::open(ops, config)?;
    let group = if let Some(group) = config.memory_group {
        group
    } else {
        let group = ops.kernel_arg_group_id(session.kernel, config.memory_arg);
        if group < 0 {
            return Err(XrtTmatmulError::Xrt {
                operation: "xrtKernelArgGroupId",
                code: group,
            });
        }
        group as u32
    };

    // This ordering is the four-BO ABI: matrix, input, output, then program.
    let matrix_bo = session.allocate_bo(request.matrix.len(), group, "xrtBOAlloc(matrix)")?;
    let input_bo = session.allocate_bo(request.input.len(), group, "xrtBOAlloc(input)")?;
    let output_bo = session.allocate_bo(output.len(), group, "xrtBOAlloc(output)")?;
    let matrix_address = session.bo_address(matrix_bo, "matrix")?;
    let input_address = session.bo_address(input_bo, "input")?;
    let output_address = session.bo_address(output_bo, "output")?;

    let labels = bind_labels(
        request.matrix_label,
        request.input_label,
        request.output_label,
        matrix_address,
        input_address,
        output_address,
    )?;
    let replay_safe_program = super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
        request.assembly,
        &labels,
        config.num_vector_registers,
    )
    .map_err(|error| XrtTmatmulError::Assemble(error.to_string()))?;
    let program = compact_xrt_program(&replay_safe_program)?;
    validate_program(&program)?;
    let program_bo = session.allocate_bo(program.len(), group, "xrtBOAlloc(program)")?;
    let program_address = session.bo_address(program_bo, "program")?;

    bo_write_and_sync(ops, matrix_bo, request.matrix, "matrix")?;
    bo_write_and_sync(ops, input_bo, request.input, "input")?;
    bo_write_and_sync(ops, program_bo, &program, "program")?;

    session.register_write(
        registers.reset,
        0,
        "xrtKernelWriteRegister(RESET)",
        "xclRegWrite(RESET)",
    )?;
    session.register_write(
        registers.dma_control,
        1,
        "xrtKernelWriteRegister(MM2S_DMACR)",
        "xclRegWrite(MM2S_DMACR)",
    )?;
    session.register_write(
        registers.dma_source_lo,
        program_address as u32,
        "xrtKernelWriteRegister(MM2S_SA low)",
        "xclRegWrite(MM2S_SA low)",
    )?;
    session.register_write(
        registers.dma_source_hi,
        (program_address >> 32) as u32,
        "xrtKernelWriteRegister(MM2S_SA high)",
        "xclRegWrite(MM2S_SA high)",
    )?;
    session.register_write(
        registers.dma_length,
        program.len() as u32,
        "xrtKernelWriteRegister(MM2S_LENGTH)",
        "xclRegWrite(MM2S_LENGTH)",
    )?;

    let deadline = Instant::now() + Duration::from_millis(u64::from(config.timeout_ms));
    let wait_result = (|| {
        loop {
            let mut value = 0;
            session.register_read(
                registers.stall,
                &mut value,
                "xrtKernelReadRegister(STALL)",
                "xclRegRead(STALL)",
            )?;
            if value != 0 {
                return Ok(value);
            }
            if Instant::now() >= deadline {
                return Err(XrtTmatmulError::Timeout {
                    timeout_ms: config.timeout_ms,
                });
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    })();
    let stall_code = match wait_result {
        Ok(value) => value,
        Err(primary) => {
            if let Err(cleanup) = session.quiesce_after_launch(registers, config.timeout_ms) {
                return Err(XrtTmatmulError::Quiesce {
                    primary: primary.to_string(),
                    cleanup: cleanup.to_string(),
                });
            }
            return Err(primary);
        }
    };

    check_xrt(
        "xrtBOSync(output, from-device)",
        ops.bo_sync(output_bo, XRT_BO_SYNC_FROM_DEVICE, output.len(), 0),
    )?;
    let mut completed_output = vec![0u8; output.len()];
    check_xrt(
        "xrtBORead(output)",
        ops.bo_read(output_bo, &mut completed_output),
    )?;
    session.register_write(
        registers.stall,
        1,
        "xrtKernelWriteRegister(STALL)",
        "xclRegWrite(STALL)",
    )?;
    output.copy_from_slice(&completed_output);

    Ok(XrtTmatmulStatus {
        stall_code,
        program_bytes: program.len(),
        matrix_address,
        input_address,
        output_address,
        program_address,
    })
}

fn bo_write_and_sync<O: XrtOps>(
    ops: &O,
    bo: Handle,
    bytes: &[u8],
    name: &'static str,
) -> Result<(), XrtTmatmulError> {
    check_xrt(
        match name {
            "matrix" => "xrtBOWrite(matrix)",
            "input" => "xrtBOWrite(input)",
            "program" => "xrtBOWrite(program)",
            _ => "xrtBOWrite",
        },
        ops.bo_write(bo, bytes),
    )?;
    check_xrt(
        match name {
            "matrix" => "xrtBOSync(matrix, to-device)",
            "input" => "xrtBOSync(input, to-device)",
            "program" => "xrtBOSync(program, to-device)",
            _ => "xrtBOSync(to-device)",
        },
        ops.bo_sync(bo, XRT_BO_SYNC_TO_DEVICE, bytes.len(), 0),
    )
}

fn check_xrt(operation: &'static str, code: i32) -> Result<(), XrtTmatmulError> {
    if code == 0 {
        Ok(())
    } else {
        Err(XrtTmatmulError::Xrt { operation, code })
    }
}

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
        reset: add(RESET)?,
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

fn compact_xrt_program(replay_safe_program: &[u8]) -> Result<Vec<u8>, XrtTmatmulError> {
    validate_program(replay_safe_program)?;
    let terminal_offset = replay_safe_program.len() - INSTRUCTION_BYTES;
    let padding_offset = replay_safe_program[..terminal_offset]
        .chunks_exact(INSTRUCTION_BYTES)
        .position(|instruction| instruction.iter().all(|byte| *byte == 0))
        .map_or(terminal_offset, |slot| slot * INSTRUCTION_BYTES);
    if replay_safe_program[padding_offset..terminal_offset]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(XrtTmatmulError::InvalidProgram(
            "replay-safe padding contains a nonzero instruction after its first zero slot"
                .to_string(),
        ));
    }

    let mut compact = Vec::with_capacity(padding_offset + INSTRUCTION_BYTES);
    compact.extend_from_slice(&replay_safe_program[..padding_offset]);
    compact.extend_from_slice(&replay_safe_program[terminal_offset..]);
    validate_program(&compact)?;
    Ok(compact)
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
    if [matrix_address, input_address, output_address]
        .into_iter()
        .any(|address| address == 0 || address == u64::MAX)
    {
        return Err(XrtTmatmulError::InvalidBuffer(
            "XRT returned an invalid BO address".to_string(),
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
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::Arc;

    const DEVICE_HANDLE: usize = 1;
    const KERNEL_HANDLE: usize = 2;
    const IP_DEVICE_HANDLE: usize = 3;
    const FIRST_BO_HANDLE: usize = 100;
    const AU250_BATCH_SIZE: usize = 9;
    const AU250_VECTOR_LENGTH: usize = 1024;
    const AU250_VECTOR_ELEMENTS: usize = AU250_BATCH_SIZE * AU250_VECTOR_LENGTH;
    const TEST_ASSEMBLY: &str = r#"
        ldv v0, PARAM_INPUT
        tmatmul_import v0
        tmatmul_go PARAM_MATRIX
        tmatmul_export v1
        sv v1, PARAM_OUTPUT
        stall
    "#;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Event {
        DeviceOpen(u32),
        LoadXclbin(String),
        GetUuid,
        KernelOpen(String),
        GroupId(i32),
        XclOpen(u32),
        IpNameToIndex(String),
        OpenContext {
            index: u32,
            shared: bool,
        },
        BoAlloc {
            bo: usize,
            size: usize,
            group: u32,
        },
        BoWrite {
            bo: usize,
            bytes: Vec<u8>,
        },
        BoSync {
            bo: usize,
            direction: i32,
            size: usize,
        },
        RegisterWrite {
            offset: u32,
            value: u32,
        },
        RegisterRead(u32),
        IpRegisterWrite {
            offset: u32,
            value: u32,
        },
        IpRegisterRead(u32),
        BoRead {
            bo: usize,
            size: usize,
        },
        BoFree(usize),
        KernelClose,
        CloseContext(u32),
        XclClose,
        DeviceClose,
    }

    struct FakeState {
        events: Vec<Event>,
        next_bo: usize,
        stall_reads: VecDeque<u32>,
        stall_reads_by_ip: HashMap<u32, VecDeque<u32>>,
        fail_register_write: Option<u32>,
        fail_register_read: Option<u32>,
        output_pattern: u8,
    }

    struct FakeXrt {
        state: RefCell<FakeState>,
    }

    impl FakeXrt {
        fn new(stall_reads: impl IntoIterator<Item = u32>) -> Self {
            Self {
                state: RefCell::new(FakeState {
                    events: Vec::new(),
                    next_bo: FIRST_BO_HANDLE,
                    stall_reads: stall_reads.into_iter().collect(),
                    stall_reads_by_ip: HashMap::new(),
                    fail_register_write: None,
                    fail_register_read: None,
                    output_pattern: 0x5a,
                }),
            }
        }

        fn with_per_ip_stalls(stalls: Vec<Vec<u32>>) -> Self {
            let xrt = Self::new([]);
            xrt.state.borrow_mut().stall_reads_by_ip = stalls
                .into_iter()
                .enumerate()
                .map(|(index, values)| (index as u32, values.into_iter().collect()))
                .collect();
            xrt
        }

        fn fail_register_write(&self, offset: u32) {
            self.state.borrow_mut().fail_register_write = Some(offset);
        }

        fn fail_register_read(&self, offset: u32) {
            self.state.borrow_mut().fail_register_read = Some(offset);
        }

        fn events(&self) -> Vec<Event> {
            self.state.borrow().events.clone()
        }
    }

    impl XrtOps for FakeXrt {
        fn device_open(&self, index: u32) -> Handle {
            self.state
                .borrow_mut()
                .events
                .push(Event::DeviceOpen(index));
            DEVICE_HANDLE as Handle
        }

        fn device_close(&self, _device: Handle) -> i32 {
            self.state.borrow_mut().events.push(Event::DeviceClose);
            0
        }

        fn load_xclbin_file(&self, _device: Handle, path: &CStr) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::LoadXclbin(path.to_string_lossy().into_owned()));
            0
        }

        fn get_xclbin_uuid(&self, _device: Handle, uuid: &mut Xuid) -> i32 {
            *uuid = [0x2a; 16];
            self.state.borrow_mut().events.push(Event::GetUuid);
            0
        }

        fn kernel_open_exclusive(&self, _device: Handle, _uuid: &Xuid, name: &CStr) -> Handle {
            self.state
                .borrow_mut()
                .events
                .push(Event::KernelOpen(name.to_string_lossy().into_owned()));
            KERNEL_HANDLE as Handle
        }

        fn kernel_close(&self, _kernel: Handle) -> i32 {
            self.state.borrow_mut().events.push(Event::KernelClose);
            0
        }

        fn kernel_arg_group_id(&self, _kernel: Handle, arg: i32) -> i32 {
            self.state.borrow_mut().events.push(Event::GroupId(arg));
            7
        }

        fn kernel_read_register(&self, _kernel: Handle, offset: u32, value: &mut u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::RegisterRead(offset));
            if state.fail_register_read == Some(offset) {
                return -5;
            }
            *value = match offset {
                STALL => state.stall_reads.pop_front().unwrap_or(0),
                MM2S_DMACR => 0,
                MM2S_DMASR => 1,
                _ => 0,
            };
            0
        }

        fn kernel_write_register(&self, _kernel: Handle, offset: u32, value: u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::RegisterWrite { offset, value });
            if state.fail_register_write == Some(offset) {
                -5
            } else {
                0
            }
        }

        fn xcl_open(&self, index: u32) -> Handle {
            self.state.borrow_mut().events.push(Event::XclOpen(index));
            IP_DEVICE_HANDLE as Handle
        }

        fn xcl_close(&self, _device: Handle) {
            self.state.borrow_mut().events.push(Event::XclClose);
        }

        fn xcl_ip_name_to_index(&self, _device: Handle, name: &CStr) -> i32 {
            let name = name.to_string_lossy().into_owned();
            self.state
                .borrow_mut()
                .events
                .push(Event::IpNameToIndex(name.clone()));
            if name.ends_with("big_1") {
                0
            } else if name.ends_with("big_2") {
                1
            } else if name.ends_with("big_3") {
                2
            } else if name.ends_with("small_1") {
                3
            } else {
                -1
            }
        }

        fn xcl_open_context(&self, _device: Handle, _uuid: &Xuid, index: u32, shared: bool) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::OpenContext { index, shared });
            0
        }

        fn xcl_close_context(&self, _device: Handle, _uuid: &Xuid, index: u32) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::CloseContext(index));
            0
        }

        fn xcl_reg_read(&self, _device: Handle, index: u32, offset: u32, value: &mut u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::IpRegisterRead(offset));
            if state.fail_register_read == Some(offset) {
                return -5;
            }
            *value = match offset {
                STALL => {
                    if let Some(reads) = state.stall_reads_by_ip.get_mut(&index) {
                        reads.pop_front().unwrap_or(0)
                    } else {
                        state.stall_reads.pop_front().unwrap_or(0)
                    }
                }
                MM2S_DMACR => 0,
                MM2S_DMASR => 1,
                _ => 0,
            };
            0
        }

        fn xcl_reg_write(&self, _device: Handle, _index: u32, offset: u32, value: u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::IpRegisterWrite { offset, value });
            if state.fail_register_write == Some(offset) {
                -5
            } else {
                0
            }
        }

        fn bo_alloc(&self, _device: Handle, size: usize, _flags: u64, group: u32) -> Handle {
            let mut state = self.state.borrow_mut();
            let bo = state.next_bo;
            state.next_bo += 1;
            state.events.push(Event::BoAlloc { bo, size, group });
            bo as Handle
        }

        fn bo_free(&self, bo: Handle) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::BoFree(bo as usize));
            0
        }

        fn bo_address(&self, bo: Handle) -> u64 {
            0x1000 * (bo as usize - FIRST_BO_HANDLE + 1) as u64
        }

        fn bo_write(&self, bo: Handle, bytes: &[u8]) -> i32 {
            self.state.borrow_mut().events.push(Event::BoWrite {
                bo: bo as usize,
                bytes: bytes.to_vec(),
            });
            0
        }

        fn bo_read(&self, bo: Handle, bytes: &mut [u8]) -> i32 {
            let mut state = self.state.borrow_mut();
            bytes.fill(state.output_pattern);
            state.events.push(Event::BoRead {
                bo: bo as usize,
                size: bytes.len(),
            });
            0
        }

        fn bo_sync(&self, bo: Handle, direction: i32, size: usize, _offset: usize) -> i32 {
            self.state.borrow_mut().events.push(Event::BoSync {
                bo: bo as usize,
                direction,
                size,
            });
            0
        }
    }

    fn test_config(timeout_ms: u32) -> XrtConfig {
        XrtConfig {
            xclbin: PathBuf::from("/tmp/ternary_matmul.xclbin"),
            device_index: 0,
            kernel_name: "ternip_ip".to_string(),
            ip_name: None,
            instance: 0,
            memory_arg: 14,
            memory_group: None,
            num_vector_registers: 8,
            timeout_ms,
        }
    }

    fn test_request() -> XrtTmatmulRequest<'static> {
        XrtTmatmulRequest {
            assembly: TEST_ASSEMBLY,
            matrix_label: "PARAM_MATRIX",
            input_label: "PARAM_INPUT",
            output_label: "PARAM_OUTPUT",
            matrix: &[1, 2, 3, 4],
            input: &[5, 6, 7, 8],
        }
    }

    fn native_ip_config(timeout_ms: u32) -> XrtConfig {
        XrtConfig {
            ip_name: Some("ternip_big:ternip_big_1".to_string()),
            memory_group: Some(0),
            num_vector_registers: 4,
            ..test_config(timeout_ms)
        }
    }

    fn test_wave_jobs() -> Vec<XrtWaveJob> {
        XrtPoolConfig::maxcores_targets()
            .into_iter()
            .enumerate()
            .map(|(cu_index, target)| XrtWaveJob {
                request_id: 10 + cu_index as u64,
                cu_index,
                matrix: Arc::from(vec![0_u8; AU250_MATRIX_BYTES]),
                input: vec![0_u8; expected_vector_bytes(target.lanes)],
            })
            .collect()
    }

    fn pool_test_config() -> XrtPoolConfig {
        XrtPoolConfig {
            xclbin: PathBuf::from("/tmp/MaxCores_370M.xclbin"),
            device_index: 0,
            targets: XrtPoolConfig::maxcores_targets(),
            num_vector_registers: 4,
            timeout_ms: 20,
        }
    }

    #[test]
    fn maxcores_targets_match_xclbin_layout() {
        assert_eq!(
            XrtPoolConfig::maxcores_targets(),
            vec![
                XrtCuTarget {
                    ip_name: "ternip_big:ternip_big_1".into(),
                    memory_group: 0,
                    lanes: 9,
                },
                XrtCuTarget {
                    ip_name: "ternip_big:ternip_big_2".into(),
                    memory_group: 3,
                    lanes: 9,
                },
                XrtCuTarget {
                    ip_name: "ternip_big:ternip_big_3".into(),
                    memory_group: 2,
                    lanes: 9,
                },
                XrtCuTarget {
                    ip_name: "ternip_small:ternip_small_1".into(),
                    memory_group: 1,
                    lanes: 6,
                },
            ]
        );
    }

    #[test]
    fn cu_table_override_is_one_versioned_atomic_value() {
        let json = r#"{"version":1,"cus":[{"ip_name":"ternip_big:ternip_big_1","memory_group":0,"lanes":9}]}"#;
        let parsed = XrtPoolConfig::parse_cu_table(json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(XrtPoolConfig::parse_cu_table(r#"{"version":2,"cus":[]}"#).is_err());
        assert!(XrtPoolConfig::parse_cu_table(r#"{"version":1,"cus":[]}"#).is_err());
    }

    #[test]
    fn pool_loads_xclbin_once_and_allocates_four_bos_per_cu() {
        let xrt = FakeXrt::new([1, 1, 1, 1]);
        let pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
        assert_eq!(
            pool.ops
                .events()
                .iter()
                .filter(|event| matches!(event, Event::LoadXclbin(_)))
                .count(),
            1
        );
        assert_eq!(
            pool.ops
                .events()
                .iter()
                .filter(|event| matches!(event, Event::BoAlloc { .. }))
                .count(),
            16
        );
    }

    #[test]
    fn wave_returns_request_order_after_out_of_order_stalls() {
        let xrt =
            FakeXrt::with_per_ip_stalls(vec![vec![0, 0, 1], vec![1], vec![0, 0, 0, 1], vec![0, 1]]);
        let mut pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
        let completions = pool.run_wave(test_wave_jobs()).unwrap();
        assert_eq!(
            completions
                .iter()
                .map(|completion| completion.request_id)
                .collect::<Vec<_>>(),
            vec![10, 11, 12, 13]
        );
    }

    #[test]
    fn wave_rejects_wrong_payload_sizes_before_register_writes() {
        let xrt = FakeXrt::new([1]);
        let mut pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
        let mut jobs = test_wave_jobs();
        jobs.truncate(1);
        jobs[0].input.pop();

        let error = pool.run_wave(jobs).unwrap_err();
        assert!(error.to_string().contains("input has"), "{error}");
        assert!(!pool.ops.events().iter().any(|event| matches!(
            event,
            Event::IpRegisterWrite { .. } | Event::IpRegisterRead(_)
        )));
    }

    #[test]
    fn failed_quiescence_poisons_pool_and_retains_uncertain_cu() {
        let xrt = FakeXrt::new([1]);
        let mut pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
        pool.ops.fail_register_write(RESET);
        let mut jobs = test_wave_jobs();
        jobs.truncate(1);

        let error = pool.run_wave(jobs.clone()).unwrap_err();
        assert!(matches!(error, XrtTmatmulError::Quiesce { .. }));
        let retry = pool.run_wave(jobs).unwrap_err();
        assert!(retry.to_string().contains("poisoned"), "{retry}");
        assert!(!pool.cus[0].release_handles);
        assert!(!pool.release_device);
    }

    fn au250_vector_add_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut vector_a = Vec::with_capacity(AU250_VECTOR_ELEMENTS * 2);
        let mut vector_b = Vec::with_capacity(AU250_VECTOR_ELEMENTS * 2);
        let mut expected = Vec::with_capacity(AU250_VECTOR_ELEMENTS * 2);

        for index in 0..AU250_VECTOR_ELEMENTS {
            let a_raw = ((index % 8) as i16) * 32;
            let b_raw = 48i16;
            vector_a.extend_from_slice(&a_raw.to_le_bytes());
            vector_b.extend_from_slice(&b_raw.to_le_bytes());
            expected.extend_from_slice(&(a_raw + b_raw).to_le_bytes());
        }
        (vector_a, vector_b, expected)
    }

    fn au250_tmatmul_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut matrix = vec![0u8; AU250_VECTOR_LENGTH * AU250_VECTOR_LENGTH / 4];
        matrix[..2 * AU250_VECTOR_LENGTH / 4].fill(0x55);

        let mut input = vec![0u8; AU250_VECTOR_ELEMENTS * 2];
        let mut expected = vec![0u8; AU250_VECTOR_ELEMENTS * 2];
        for lane in 0..AU250_BATCH_SIZE {
            for column in [lane, lane + 1] {
                let offset = (column * AU250_BATCH_SIZE + lane) * 2;
                input[offset..offset + 2].copy_from_slice(&32i16.to_le_bytes());
            }
            for row in 0..2 {
                let offset = (row * AU250_BATCH_SIZE + lane) * 2;
                expected[offset..offset + 2].copy_from_slice(&64i16.to_le_bytes());
            }
        }
        (matrix, input, expected)
    }

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
                reset: 0x2000,
            }
        );
        assert_eq!(instance_registers(2).unwrap().stall, 0x9000);
        assert_eq!(instance_registers(2).unwrap().reset, 0xa000);
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
        assert_eq!(config.ip_name, None);
        assert_eq!(config.instance, 0);
        assert_eq!(config.memory_arg, 14);
        assert_eq!(config.memory_group, None);
        assert_eq!(config.num_vector_registers, 8);
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
    fn config_selects_native_ip_and_explicit_memory_group() {
        let config = XrtConfig::from_lookup(|name| match name {
            "HETGPU_XRT_XCLBIN" => Some("kernel.xclbin".into()),
            "HETGPU_XRT_IP_NAME" => Some("ternip_big:ternip_big_1".into()),
            "HETGPU_XRT_MEMORY_GROUP" => Some("0".into()),
            "HETGPU_XRT_NUM_VECTOR_REGISTERS" => Some("4".into()),
            _ => None,
        })
        .unwrap();

        assert_eq!(config.ip_name.as_deref(), Some("ternip_big:ternip_big_1"));
        assert_eq!(config.memory_group, Some(0));
        assert_eq!(config.num_vector_registers, 4);
    }

    #[test]
    fn native_ip_requires_explicit_memory_group() {
        let error = XrtConfig::from_lookup(|name| match name {
            "HETGPU_XRT_XCLBIN" => Some("kernel.xclbin".into()),
            "HETGPU_XRT_IP_NAME" => Some("ternip_big:ternip_big_1".into()),
            _ => None,
        })
        .unwrap_err();

        assert!(error.to_string().contains("HETGPU_XRT_MEMORY_GROUP"));
    }

    #[test]
    fn xclbin_is_required() {
        let error = XrtConfig::from_lookup(|_| None).unwrap_err();
        assert!(error.to_string().contains("HETGPU_XRT_XCLBIN"));
    }

    #[test]
    fn au250_vector_add_fixture_matches_known_good_example() {
        let (vector_a, vector_b, expected) = au250_vector_add_fixture();

        assert_eq!(vector_a.len(), 9 * 1024 * 2);
        assert_eq!(vector_b.len(), vector_a.len());
        assert_eq!(expected.len(), vector_a.len());

        let first_a: Vec<i16> = vector_a[..16]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let first_b: Vec<i16> = vector_b[..16]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let first_expected: Vec<i16> = expected[..16]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();

        assert_eq!(first_a, vec![0, 32, 64, 96, 128, 160, 192, 224]);
        assert_eq!(first_b, vec![48; 8]);
        assert_eq!(first_expected, vec![48, 80, 112, 144, 176, 208, 240, 272]);
    }

    #[test]
    fn xrt_compaction_matches_known_good_five_instruction_image() {
        let labels = HashMap::from([
            ("PARAM_MATRIX".to_string(), 0x1000),
            ("PARAM_INPUT".to_string(), 0x2000),
            ("PARAM_OUTPUT".to_string(), 0x3000),
        ]);
        let replay_safe = super::super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
            r#"
                ldv v0, PARAM_MATRIX
                ldv v1, PARAM_INPUT
                add v2, v0, v1
                sv v2, PARAM_OUTPUT
                stall
            "#,
            &labels,
            4,
        )
        .unwrap();

        let compact = compact_xrt_program(&replay_safe).unwrap();
        let expected = [
            [
                0x00, 0x10, 0, 0, 0, 0, 0, 0, 0x20, 0x00, 0x02, 0, 0, 0, 0, 0,
            ],
            [
                0x00, 0x20, 0, 0, 0, 0, 0, 0, 0xa0, 0x0a, 0x02, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x23, 0x04, 0, 0, 0, 0, 0],
            [
                0x00, 0x30, 0, 0, 0, 0, 0, 0, 0x40, 0x15, 0x02, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00, 0x0a, 0, 0, 0, 0, 0],
        ]
        .concat();

        assert_eq!(compact, expected);
        assert_eq!(compact.len(), 5 * INSTRUCTION_BYTES);
    }

    #[test]
    fn au250_tmatmul_fixture_matches_repository_canonical_case() {
        let (matrix, input, expected) = au250_tmatmul_fixture();

        assert_eq!(matrix.len(), 1024 * 1024 / 4);
        assert!(matrix[..512].iter().all(|byte| *byte == 0x55));
        assert!(matrix[512..].iter().all(|byte| *byte == 0));
        let input_raw: Vec<i16> = input
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let expected_raw: Vec<i16> = expected
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(input_raw.iter().filter(|value| **value == 32).count(), 18);
        for lane in 0..9 {
            assert_eq!(input_raw[lane * AU250_BATCH_SIZE + lane], 32);
            assert_eq!(input_raw[(lane + 1) * AU250_BATCH_SIZE + lane], 32);
            assert_eq!(expected_raw[lane], 64);
            assert_eq!(expected_raw[AU250_BATCH_SIZE + lane], 64);
        }
        assert!(
            expected_raw[2 * AU250_BATCH_SIZE..]
                .iter()
                .all(|value| *value == 0)
        );
    }

    #[test]
    fn tmatmul_image_matches_existing_xcu250_assembler() {
        let labels = HashMap::from([
            ("PARAM_MATRIX".to_string(), 0x1000),
            ("PARAM_INPUT".to_string(), 0x2000),
            ("PARAM_OUTPUT".to_string(), 0x3000),
        ]);
        let replay_safe = super::super::cxl_tmatmul::assemble_tmatmul_program_for_vector_registers(
            TEST_ASSEMBLY,
            &labels,
            4,
        )
        .unwrap();
        let compact = compact_xrt_program(&replay_safe).unwrap();
        let expected = [
            [
                0x00, 0x20, 0, 0, 0, 0, 0, 0, 0x20, 0x00, 0x02, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x08, 0x00, 0x06, 0, 0, 0, 0, 0],
            [
                0x00, 0x10, 0, 0, 0, 0, 0, 0, 0x10, 0x00, 0x06, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x98, 0x0a, 0x06, 0, 0, 0, 0, 0],
            [
                0x00, 0x30, 0, 0, 0, 0, 0, 0, 0xc0, 0x0a, 0x02, 0, 0, 0, 0, 0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00, 0x0a, 0, 0, 0, 0, 0],
        ]
        .concat();

        assert_eq!(compact, expected);
        assert_eq!(compact.len(), 6 * INSTRUCTION_BYTES);
    }

    #[test]
    #[ignore = "requires the app215 XRT legacy xcl register API used by AU250"]
    fn installed_xrt_symbols_resolve() {
        let api = RealXrt::load(true).unwrap();
        assert!(!api.library.is_null());
        assert!(!api.native_ip.as_ref().unwrap().library.is_null());
    }

    #[test]
    fn submit_uses_four_bo_abi_and_embeds_real_addresses() {
        let xrt = FakeXrt::new([0, 1]);
        let mut output = [0u8; 8];

        let status = submit_with_ops(&xrt, &test_config(20), test_request(), &mut output).unwrap();

        assert_eq!(output, [0x5a; 8]);
        assert_eq!(status.program_bytes, 96);
        assert_eq!(status.matrix_address, 0x1000);
        assert_eq!(status.input_address, 0x2000);
        assert_eq!(status.output_address, 0x3000);
        assert_eq!(status.program_address, 0x4000);
        assert_eq!(status.stall_code, 1);

        let events = xrt.events();
        let allocations: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::BoAlloc { bo, size, group } => Some((*bo, *size, *group)),
                _ => None,
            })
            .collect();
        assert_eq!(
            allocations,
            vec![(100, 4, 7), (101, 4, 7), (102, 8, 7), (103, 96, 7)]
        );

        let program = events
            .iter()
            .find_map(|event| match event {
                Event::BoWrite { bo: 103, bytes } => Some(bytes),
                _ => None,
            })
            .unwrap();
        assert_eq!(program.len(), 96);
        assert_eq!(
            u64::from_le_bytes(program[0..8].try_into().unwrap()),
            0x2000
        );
        assert_eq!(
            u64::from_le_bytes(program[32..40].try_into().unwrap()),
            0x1000
        );
        assert_eq!(
            u64::from_le_bytes(program[64..72].try_into().unwrap()),
            0x3000
        );

        let syncs: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::BoSync {
                    bo,
                    direction,
                    size,
                } => Some((*bo, *direction, *size)),
                _ => None,
            })
            .collect();
        assert_eq!(
            syncs,
            vec![
                (100, XRT_BO_SYNC_TO_DEVICE, 4),
                (101, XRT_BO_SYNC_TO_DEVICE, 4),
                (103, XRT_BO_SYNC_TO_DEVICE, 96),
                (102, XRT_BO_SYNC_FROM_DEVICE, 8),
            ]
        );

        let writes: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::RegisterWrite { offset, value } => Some((*offset, *value)),
                _ => None,
            })
            .collect();
        assert_eq!(
            writes,
            vec![
                (0x2000, 0),
                (0x0000, 1),
                (0x0018, 0x4000),
                (0x001c, 0),
                (0x0028, 96),
                (0x1000, 1)
            ]
        );
        assert_eq!(
            &events[events.len() - 6..],
            &[
                Event::BoFree(103),
                Event::BoFree(102),
                Event::BoFree(101),
                Event::BoFree(100),
                Event::KernelClose,
                Event::DeviceClose,
            ]
        );
    }

    #[test]
    fn native_ip_submit_uses_xcl_registers_and_explicit_four_bo_group() {
        let xrt = FakeXrt::new([0, 1]);
        let mut output = [0u8; 8];

        let status =
            submit_with_ops(&xrt, &native_ip_config(20), test_request(), &mut output).unwrap();

        assert_eq!(status.stall_code, 1);
        assert_eq!(output, [0x5a; 8]);
        let events = xrt.events();
        assert!(events.contains(&Event::XclOpen(0)));
        assert!(events.contains(&Event::IpNameToIndex("ternip_big:ternip_big_1".to_string())));
        assert!(events.contains(&Event::OpenContext {
            index: 0,
            shared: false,
        }));
        assert!(!events.iter().any(|event| matches!(
            event,
            Event::KernelOpen(_) | Event::GroupId(_) | Event::KernelClose
        )));

        let allocations: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::BoAlloc { bo, size, group } => Some((*bo, *size, *group)),
                _ => None,
            })
            .collect();
        assert_eq!(
            allocations,
            vec![(100, 4, 0), (101, 4, 0), (102, 8, 0), (103, 96, 0)]
        );

        let writes: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::IpRegisterWrite { offset, value } => Some((*offset, *value)),
                _ => None,
            })
            .collect();
        assert_eq!(
            writes,
            vec![
                (0x2000, 0),
                (0x0000, 1),
                (0x0018, 0x4000),
                (0x001c, 0),
                (0x0028, 96),
                (0x1000, 1),
            ]
        );
        assert!(events.contains(&Event::IpRegisterRead(0x1000)));
        assert_eq!(
            &events[events.len() - 7..],
            &[
                Event::BoFree(103),
                Event::BoFree(102),
                Event::BoFree(101),
                Event::BoFree(100),
                Event::CloseContext(0),
                Event::XclClose,
                Event::DeviceClose,
            ]
        );
    }

    #[test]
    #[ignore = "requires HETGPU_XRT_AU250_TEST=1 and the live AU250"]
    fn au250_vector_add_runs_when_requested() {
        assert_eq!(
            std::env::var("HETGPU_XRT_AU250_TEST").as_deref(),
            Ok("1"),
            "set HETGPU_XRT_AU250_TEST=1 only inside au250-run"
        );

        let (vector_a, vector_b, expected) = au250_vector_add_fixture();
        let mut output = vec![0u8; expected.len()];
        let request = XrtTmatmulRequest {
            assembly: r#"
                ldv v0, PARAM_MATRIX
                ldv v1, PARAM_INPUT
                add v2, v0, v1
                sv v2, PARAM_OUTPUT
                stall
            "#,
            matrix_label: "PARAM_MATRIX",
            input_label: "PARAM_INPUT",
            output_label: "PARAM_OUTPUT",
            matrix: &vector_a,
            input: &vector_b,
        };

        let status = submit_xrt_tmatmul(request, &mut output).unwrap();
        if output != expected {
            let byte = output
                .iter()
                .zip(&expected)
                .position(|(actual, wanted)| actual != wanted)
                .unwrap();
            let element_offset = byte & !1;
            panic!(
                "AU250 output mismatch at element {}: got raw {}, expected raw {}",
                byte / 2,
                i16::from_le_bytes(
                    output[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap()
                ),
                i16::from_le_bytes(
                    expected[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap()
                )
            );
        }
        eprintln!(
            "AU250 XRT vector add PASS: {} elements, program={} bytes, stall=0x{:08x}",
            AU250_VECTOR_ELEMENTS, status.program_bytes, status.stall_code
        );
    }

    #[test]
    #[ignore = "requires HETGPU_XRT_AU250_TEST=1 and the live AU250"]
    fn au250_tmatmul_runs_when_requested() {
        assert_eq!(
            std::env::var("HETGPU_XRT_AU250_TEST").as_deref(),
            Ok("1"),
            "set HETGPU_XRT_AU250_TEST=1 only inside au250-run"
        );

        let (matrix, input, expected) = au250_tmatmul_fixture();
        let mut output = vec![0u8; expected.len()];
        let request = XrtTmatmulRequest {
            assembly: TEST_ASSEMBLY,
            matrix_label: "PARAM_MATRIX",
            input_label: "PARAM_INPUT",
            output_label: "PARAM_OUTPUT",
            matrix: &matrix,
            input: &input,
        };

        let status = submit_xrt_tmatmul(request, &mut output).unwrap();
        if output != expected {
            let byte = output
                .iter()
                .zip(&expected)
                .position(|(actual, wanted)| actual != wanted)
                .unwrap();
            let element_offset = byte & !1;
            panic!(
                "AU250 tmatmul mismatch at lane {}, element {}: got raw {}, expected raw {}",
                byte / 2 % AU250_BATCH_SIZE,
                byte / 2 / AU250_BATCH_SIZE,
                i16::from_le_bytes(
                    output[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap()
                ),
                i16::from_le_bytes(
                    expected[element_offset..element_offset + 2]
                        .try_into()
                        .unwrap()
                )
            );
        }
        eprintln!(
            "AU250 XRT tmatmul PASS: {}x{} ternary matrix, {} lanes, program={} bytes, stall=0x{:08x}",
            AU250_VECTOR_LENGTH,
            AU250_VECTOR_LENGTH,
            AU250_BATCH_SIZE,
            status.program_bytes,
            status.stall_code
        );
    }

    #[test]
    fn timeout_quiesces_native_ip_before_releasing_all_handles() {
        let xrt = FakeXrt::new([]);
        let mut output = [0xa5; 8];

        let error =
            submit_with_ops(&xrt, &native_ip_config(1), test_request(), &mut output).unwrap_err();

        assert_eq!(error, XrtTmatmulError::Timeout { timeout_ms: 1 });
        assert_eq!(output, [0xa5; 8]);
        let events = xrt.events();
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::BoRead { .. }))
        );
        assert_eq!(
            &events[events.len() - 11..],
            &[
                Event::IpRegisterWrite {
                    offset: RESET,
                    value: 1,
                },
                Event::IpRegisterWrite {
                    offset: MM2S_DMACR,
                    value: 1 << 2,
                },
                Event::IpRegisterRead(MM2S_DMACR),
                Event::IpRegisterRead(MM2S_DMASR),
                Event::BoFree(103),
                Event::BoFree(102),
                Event::BoFree(101),
                Event::BoFree(100),
                Event::CloseContext(0),
                Event::XclClose,
                Event::DeviceClose,
            ]
        );
    }

    #[test]
    fn register_failure_releases_bos_in_reverse_order() {
        let xrt = FakeXrt::new([1]);
        xrt.fail_register_write(MM2S_LENGTH);
        let mut output = [0u8; 8];

        let error =
            submit_with_ops(&xrt, &test_config(20), test_request(), &mut output).unwrap_err();

        assert_eq!(
            error,
            XrtTmatmulError::Xrt {
                operation: "xrtKernelWriteRegister(MM2S_LENGTH)",
                code: -5,
            }
        );
        let events = xrt.events();
        assert_eq!(
            &events[events.len() - 6..],
            &[
                Event::BoFree(103),
                Event::BoFree(102),
                Event::BoFree(101),
                Event::BoFree(100),
                Event::KernelClose,
                Event::DeviceClose,
            ]
        );
    }

    #[test]
    fn unconfirmed_quiescence_retains_live_bos_and_device_handles() {
        let xrt = FakeXrt::new([]);
        xrt.fail_register_read(MM2S_DMACR);
        let mut output = [0xa5; 8];

        let error =
            submit_with_ops(&xrt, &native_ip_config(1), test_request(), &mut output).unwrap_err();

        assert!(matches!(error, XrtTmatmulError::Quiesce { .. }));
        assert_eq!(output, [0xa5; 8]);
        let events = xrt.events();
        assert!(events.contains(&Event::IpRegisterWrite {
            offset: RESET,
            value: 1,
        }));
        assert!(events.contains(&Event::IpRegisterWrite {
            offset: MM2S_DMACR,
            value: DMACR_RESET,
        }));
        assert!(!events.iter().any(|event| matches!(
            event,
            Event::BoFree(_) | Event::CloseContext(_) | Event::XclClose | Event::DeviceClose
        )));
    }
}
