use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const AXI_INSTANCE_STRIDE: u32 = 0x4000;
const MM2S_DMACR: u32 = 0x0000;
const MM2S_SA: u32 = 0x0018;
const MM2S_LENGTH: u32 = 0x0028;
const STALL: u32 = 0x1000;
const RESET: u32 = 0x2000;
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
    reset: u32,
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

pub(crate) fn submit_xrt_tmatmul(
    request: XrtTmatmulRequest<'_>,
    output: &mut [u8],
) -> Result<XrtTmatmulStatus, XrtTmatmulError> {
    let config = XrtConfig::from_env()?;
    let xrt = RealXrt::load()?;
    submit_with_ops(&xrt, &config, request, output)
}

struct Session<'a, O: XrtOps> {
    ops: &'a O,
    device: Handle,
    kernel: Handle,
    bos: Vec<Handle>,
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
            bos: Vec::new(),
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

        let mut uuid = [0u8; 16];
        check_xrt(
            "xrtDeviceGetXclbinUUID",
            ops.get_xclbin_uuid(device, &mut uuid),
        )?;
        let kernel_name = CString::new(config.kernel_name.as_str()).map_err(|_| {
            XrtTmatmulError::Config("HETGPU_XRT_KERNEL contains a NUL byte".to_string())
        })?;
        session.kernel = ops.kernel_open_exclusive(device, &uuid, &kernel_name);
        if session.kernel.is_null() {
            return Err(XrtTmatmulError::NullHandle("xrtPLKernelOpenExclusive"));
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
}

impl<O: XrtOps> Drop for Session<'_, O> {
    fn drop(&mut self) {
        while let Some(bo) = self.bos.pop() {
            let _ = self.ops.bo_free(bo);
        }
        if !self.kernel.is_null() {
            let _ = self.ops.kernel_close(self.kernel);
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
    let group = ops.kernel_arg_group_id(session.kernel, config.memory_arg);
    if group < 0 {
        return Err(XrtTmatmulError::Xrt {
            operation: "xrtKernelArgGroupId",
            code: group,
        });
    }
    let group = group as u32;

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
    let program = super::cxl_tmatmul::assemble_tmatmul_program(request.assembly, &labels)
        .map_err(|error| XrtTmatmulError::Assemble(error.to_string()))?;
    validate_program(&program)?;
    let program_bo = session.allocate_bo(program.len(), group, "xrtBOAlloc(program)")?;
    let program_address = session.bo_address(program_bo, "program")?;

    bo_write_and_sync(ops, matrix_bo, request.matrix, "matrix")?;
    bo_write_and_sync(ops, input_bo, request.input, "input")?;
    bo_write_and_sync(ops, program_bo, &program, "program")?;

    write_register(
        ops,
        session.kernel,
        registers.reset,
        0,
        "xrtKernelWriteRegister(RESET)",
    )?;
    write_register(
        ops,
        session.kernel,
        registers.dma_control,
        1,
        "xrtKernelWriteRegister(MM2S_DMACR)",
    )?;
    write_register(
        ops,
        session.kernel,
        registers.dma_source_lo,
        program_address as u32,
        "xrtKernelWriteRegister(MM2S_SA low)",
    )?;
    write_register(
        ops,
        session.kernel,
        registers.dma_source_hi,
        (program_address >> 32) as u32,
        "xrtKernelWriteRegister(MM2S_SA high)",
    )?;
    write_register(
        ops,
        session.kernel,
        registers.dma_length,
        program.len() as u32,
        "xrtKernelWriteRegister(MM2S_LENGTH)",
    )?;

    let deadline = Instant::now() + Duration::from_millis(u64::from(config.timeout_ms));
    let stall_code = loop {
        let mut value = 0;
        check_xrt(
            "xrtKernelReadRegister(STALL)",
            ops.kernel_read_register(session.kernel, registers.stall, &mut value),
        )?;
        if value != 0 {
            break value;
        }
        if Instant::now() >= deadline {
            return Err(XrtTmatmulError::Timeout {
                timeout_ms: config.timeout_ms,
            });
        }
        std::thread::sleep(Duration::from_millis(1));
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
    write_register(
        ops,
        session.kernel,
        registers.stall,
        1,
        "xrtKernelWriteRegister(STALL)",
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

fn write_register<O: XrtOps>(
    ops: &O,
    kernel: Handle,
    offset: u32,
    value: u32,
    operation: &'static str,
) -> Result<(), XrtTmatmulError> {
    check_xrt(operation, ops.kernel_write_register(kernel, offset, value))
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

    const DEVICE_HANDLE: usize = 1;
    const KERNEL_HANDLE: usize = 2;
    const FIRST_BO_HANDLE: usize = 100;
    const AU250_VECTOR_ELEMENTS: usize = 9 * 1024;
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
        BoRead {
            bo: usize,
            size: usize,
        },
        BoFree(usize),
        KernelClose,
        DeviceClose,
    }

    struct FakeState {
        events: Vec<Event>,
        next_bo: usize,
        stall_reads: VecDeque<u32>,
        fail_register_write: Option<u32>,
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
                    fail_register_write: None,
                    output_pattern: 0x5a,
                }),
            }
        }

        fn fail_register_write(&self, offset: u32) {
            self.state.borrow_mut().fail_register_write = Some(offset);
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
            *value = state.stall_reads.pop_front().unwrap_or(0);
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
            instance: 0,
            memory_arg: 14,
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
    #[ignore = "requires an installed XRT userspace runtime"]
    fn installed_xrt_symbols_resolve() {
        let api = RealXrt::load().unwrap();
        assert!(!api.library.is_null());
    }

    #[test]
    fn submit_uses_four_bo_abi_and_embeds_real_addresses() {
        let xrt = FakeXrt::new([0, 1]);
        let mut output = [0u8; 8];

        let status = submit_with_ops(&xrt, &test_config(20), test_request(), &mut output).unwrap();

        assert_eq!(output, [0x5a; 8]);
        assert_eq!(status.program_bytes, 128);
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
            vec![(100, 4, 7), (101, 4, 7), (102, 8, 7), (103, 128, 7)]
        );

        let program = events
            .iter()
            .find_map(|event| match event {
                Event::BoWrite { bo: 103, bytes } => Some(bytes),
                _ => None,
            })
            .unwrap();
        assert_eq!(program.len(), 128);
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
                (103, XRT_BO_SYNC_TO_DEVICE, 128),
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
                (0x0028, 128),
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
    fn timeout_preserves_output_and_releases_all_handles() {
        let xrt = FakeXrt::new([]);
        let mut output = [0xa5; 8];

        let error =
            submit_with_ops(&xrt, &test_config(1), test_request(), &mut output).unwrap_err();

        assert_eq!(error, XrtTmatmulError::Timeout { timeout_ms: 1 });
        assert_eq!(output, [0xa5; 8]);
        let events = xrt.events();
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::BoRead { .. })));
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
}
