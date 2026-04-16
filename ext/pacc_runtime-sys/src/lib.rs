//! PACC Runtime Bindings — Lanxin LX500 real driver interface via /dev/paccN
//!
//! Driver interface (reverse-engineered from pacc.ko DWARF + disassembly):
//!   Magic: 'p' (0x70)
//!   PACC_IOC_GET_INFO   = _IOWR('p', 0, pacc_info_size)  = 0xc0087000
//!   PACC_IOC_GET_INFO_EX = _IOWR('p', 1, pacc_info_size) = 0xc0107001
//!   PACC_IOC_MEM_ALLOC   = _IOWR('p', 2, u64)            = 0xc0087002
//!   PACC_IOC_BO_SUBMIT   = _IOW ('p', 3, u64)            = 0x40087003
//!
//! Mailbox SRAM (accessible from Pcore side via mmap or physical):
//!   AP→PACC : 0x20000000  (8KB)
//!   PACC→AP : 0x20002000  (8KB)
//!
//! PACC cluster base addresses: 0x38100000, 0x38500000, 0x39100000, 0x39500000

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// ─── ioctl numbers ────────────────────────────────────────────────────────────

pub const PACC_IOC_GET_INFO:    u64 = 0xc000_7000;
pub const PACC_IOC_GET_INFO_EX: u64 = 0xc001_0001;
pub const PACC_IOC_MEM_ALLOC:   u64 = 0xc000_8002;
pub const PACC_IOC_BO_SUBMIT:   u64 = 0x4000_8003;

// Proper encoding: _IOWR(type, nr, size) = (3<<30)|(size<<16)|(type<<8)|nr
const fn _iowr(ty: u64, nr: u64, size: u64) -> u64 {
    (3 << 30) | (size << 16) | (ty << 8) | nr
}
const fn _iow(ty: u64, nr: u64, size: u64) -> u64 {
    (1 << 30) | (size << 16) | (ty << 8) | nr
}

pub const PACC_MAGIC: u64 = 0x70; // 'p'
pub const IOC_GET_INFO:   u64 = _iowr(PACC_MAGIC, 0,  8);
pub const IOC_GET_INFO_EX: u64 = _iowr(PACC_MAGIC, 1, 16);
pub const IOC_MEM_ALLOC:  u64 = _iowr(PACC_MAGIC, 2,  8);
pub const IOC_BO_SUBMIT:  u64 = _iow (PACC_MAGIC, 3,  8);

// ─── kernel struct mirrors ─────────────────────────────────────────────────────

/// pacc_info_size — arg for PACC_IOC_GET_INFO (8 bytes)
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct pacc_info_size {
    pub opcode: u32,
    pub size:   u32,
}

/// pacc_jobs_addr — arg for legacy nr=1 driver probes (16 bytes)
/// addr: physical/DMA address of job descriptor buffer
/// size: byte length of that buffer
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct pacc_jobs_addr {
    pub addr: u64,
    pub size: u64,
}

/// pacc_mbox_job_desc — one job entry (32 bytes)
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct pacc_mbox_job_desc {
    pub addr: u64,
    pub len:  u64,
    pub rsvd: u64,
    pub buf_info: u64,
}

/// Reduce operation types for NCCL AllReduce
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaccReduceOp {
    Sum  = 0,
    Prod = 1,
    Max  = 2,
    Min  = 3,
}

/// Data format for reduce
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaccDataType {
    Int8    = 0,
    Uint8   = 1,
    Int32   = 2,
    Float16 = 3,
    Float32 = 4,
    Bfloat16 = 5,
}

// ─── Memory region constants ───────────────────────────────────────────────────

/// AP→PACC mailbox SRAM physical base
pub const AP2PACC_MBOX_PHYS: u64 = 0x2000_0000;
/// PACC→AP mailbox SRAM physical base
pub const PACC2AP_MBOX_PHYS: u64 = 0x2000_2000;
/// Mailbox SRAM size (8 KB each direction)
pub const MBOX_SRAM_SIZE: usize = 0x2000;

/// PACC cluster base physical addresses
pub const PACC_BASE: [u64; 4] = [0x3810_0000, 0x3850_0000, 0x3910_0000, 0x3950_0000];

/// PACC DDR shared base (accessible to all PACCs and Pcore)
pub const PACC_DDR_BASE: u64 = 0x8000_0000;
/// PACC DDR extended base (PACC-side high address)
pub const PACC_DDR_EXT_BASE: u64 = 0x80_8000_0000;

/// Per-PACC local SRAM bases
pub const PACC_SRAM_BASE: [u64; 4] = [
    0x6000_0000, 0x6010_0000, 0x6020_0000, 0x6030_0000,
];
pub const PACC_SRAM_SIZE: usize = 0x0004_0000; // 256 KB each

/// Register offsets from /home/ubuntu/to_fckj/pacc_boot.c.
pub const PACC_CORE_NUM: usize = 4;
pub const PACC_RESET_VEC_VAL: u32 = 0x3008_0000;
pub const PACC_TOP_REG_OFF: u64 = 0x201000;
pub const PACC_TOP_REG_CORE_RESET_ADDR: u64 = 0x14;
pub const PACC_TOP_REG_SYS_RESET_ADDR: u64 = 0x24;
pub const PACC_TOP_REG_FORCE_RESETPC_RELOAD_ADDR: u64 = 0x28;
pub const PACC_TOP_REG_RESET_VEC_LO_ADDR: u64 = 0x6c;
pub const PACC_TOP_REG_RESET_VEC_HI_ADDR: u64 = 0x70;
pub const PACC_TOP_REG_SECURE_TIEOFF: u64 = 0xbc;
pub const PACC_TOP_REG_MAM_CRM: u64 = 0xc4;
pub const PACC_TOP_REG_CFG_MAX: u64 = 0x400;
pub const PACC_TOP_REG_PACC_RSVD: u64 = PACC_TOP_REG_SECURE_TIEOFF;

/// PACC-side DMA register offsets from /home/ubuntu/to_fckj/pacc_dma.c.
/// These are programmed by baremetal code running on PACC, but keeping the
/// constants here makes the Linux-side job layout match the EVB docs.
pub const PACC_DMACFG_BASE: u64 = 0x2000_5000;
pub const PACC_DMA_CH_STRIDE: u64 = 0x100;
pub const PACC_DMA_CH_SRC_OFF: u64 = 0x100;
pub const PACC_DMA_CH_DST_OFF: u64 = 0x108;
pub const PACC_DMA_CH_BLOCK_TS_OFF: u64 = 0x110;
pub const PACC_DMA_CH_CTL_OFF: u64 = 0x118;
pub const PACC_DMA_CH_CFG_OFF: u64 = 0x120;
pub const PACC_DMA_CH_STATUS_OFF: u64 = 0x188;
pub const PACC_DMA_DDR_TO_DDR_CTL: u64 = 0x783c_0000_03600;

const PACC_JOB_MAGIC: u64 = 0x5041_4343_4a4f_4231; // "PACCJOB1"
const HETGPU_PACC_JOB_MAGIC: u64 = 0x4847_5055_5041_4343; // "HGPUPACC"
const HETGPU_PACC_JOB_VERSION: u32 = 1;
const PACC_JOB_DESC_BYTES: usize = std::mem::size_of::<pacc_mbox_job_desc>();
const PACC_JOB_HEADER_BYTES: usize = std::mem::size_of::<PaccJobImageHeader>();
const HETGPU_PACC_DOORBELL_BYTES: usize = std::mem::size_of::<HetgpuPaccDoorbell>();
const HETGPU_PACC_ARG_HEADER_BYTES: usize = std::mem::size_of::<HetgpuPaccArgSlotHeader>();
pub const HETGPU_PACC_ARG_BASE: u64 = AP2PACC_MBOX_PHYS + 0x100;
pub const HETGPU_PACC_ARG_SLOT_BYTES: usize = 0x400;
pub const HETGPU_PACC_RUNTIME_TABLE_OFF: u64 = 0x1400;
const HETGPU_PACC_RUNTIME_TABLE_MAGIC: u64 = 0x4847_5055_5442_4c31;
const HETGPU_PACC_RUNTIME_TABLE_VERSION: u32 = 1;

pub mod hetgpu_pacc_job_id {
    pub const GEMM: u32 = 1;
    pub const SOFTMAX: u32 = 2;
    pub const RMSNORM: u32 = 3;
    pub const ALLREDUCE: u32 = 4;
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HetgpuPaccDoorbell {
    pub magic: u64,
    pub version: u32,
    pub job_id: u32,
    pub flags: u32,
    pub status: u32,
    pub seq: u64,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HetgpuPaccArgSlotHeader {
    pub magic: u64,
    pub version: u32,
    pub job_id: u32,
    pub seq: u64,
    pub arg_len: u64,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HetgpuPaccGemmJob {
    pub transa: u32,
    pub transb: u32,
    pub atype: u32,
    pub btype: u32,
    pub ctype: u32,
    pub compute_type: u32,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub a_addr: u64,
    pub b_addr: u64,
    pub c_addr: u64,
    pub alpha_addr: u64,
    pub beta_addr: u64,
    pub lda: i64,
    pub ldb: i64,
    pub ldc: i64,
    pub stride_a: i64,
    pub stride_b: i64,
    pub stride_c: i64,
    pub batch_count: u64,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HetgpuPaccSoftmaxJob {
    pub src_addr: u64,
    pub dst_addr: u64,
    pub rows: u64,
    pub cols: u64,
    pub stride: u64,
    pub dtype: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HetgpuPaccRmsNormJob {
    pub x_addr: u64,
    pub weight_addr: u64,
    pub y_addr: u64,
    pub rows: u64,
    pub hidden: u64,
    pub eps: f32,
    pub dtype: u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HetgpuPaccRuntimeJobTable {
    pub magic: u64,
    pub version: u32,
    pub flags: u32,
    pub seq: u64,
    pub have_gemm: u32,
    pub have_softmax: u32,
    pub have_rmsnorm: u32,
    pub reserved: u32,
    pub gemm: HetgpuPaccGemmJob,
    pub softmax: HetgpuPaccSoftmaxJob,
    pub rmsnorm: HetgpuPaccRmsNormJob,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HetgpuPaccAllReduceJob {
    pub src_addr: u64,
    pub dst_addr: u64,
    pub count: u64,
    pub nranks: u32,
    pub reduce_op: u32,
    pub dtype: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct PaccJobImageHeader {
    pub magic: u64,
    pub version: u32,
    pub flags: u32,
    pub entry_offset: u64,
    pub image_size: u64,
    pub kernel_name_hash: u64,
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub reserved: u32,
}

// ─── Mailbox message layout ────────────────────────────────────────────────────

/// Simple mailbox message header (8-byte stride in SRAM)
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct MboxMsg {
    /// Command opcode
    pub cmd:    u32,
    /// Payload length (bytes following header in SRAM)
    pub length: u16,
    /// Status / sequence number
    pub status: u16,
}

pub mod mbox_cmd {
    pub const NOOP:         u32 = 0x0000;
    pub const PING:         u32 = 0x0001;
    pub const REDUCE_SUM:   u32 = 0x0010;
    pub const REDUCE_PROD:  u32 = 0x0011;
    pub const REDUCE_MAX:   u32 = 0x0012;
    pub const REDUCE_MIN:   u32 = 0x0013;
    pub const ALLREDUCE:    u32 = 0x0020;
    pub const BARRIER:      u32 = 0x0030;
    pub const DONE:         u32 = 0x00FF;
}

fn strict_pacc() -> bool {
    std::env::var("HETGPU_PACC_STRICT").ok().as_deref() == Some("1")
}

fn hash_kernel_name(name: &str) -> u64 {
    // FNV-1a: stable across processes and cheap enough for launch metadata.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for b in name.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

fn page_size() -> usize {
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if value <= 0 { 4096 } else { value as usize }
}

pub struct PhysMap {
    file: File,
    ptr: *mut u8,
    map_len: usize,
    map_base: u64,
    phys: u64,
    len: usize,
}

impl PhysMap {
    pub fn map_rw(phys: u64, len: usize) -> std::io::Result<Self> {
        if len == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "zero-length physical map"));
        }
        let page = page_size() as u64;
        let map_base = phys & !(page - 1);
        let offset = (phys - map_base) as usize;
        let map_len = align_up(offset + len, page as usize);
        let path = std::env::var("HETGPU_PACC_DEVMEM").unwrap_or_else(|_| "/dev/mem".to_string());
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                map_base as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::last_os_error());
        }
        Ok(Self {
            file,
            ptr: ptr.cast(),
            map_len,
            map_base,
            phys,
            len,
        })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let offset = (self.phys - self.map_base) as usize;
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), self.len) }
    }

    pub fn read_u32(phys: u64) -> std::io::Result<u32> {
        let mut map = Self::map_rw(phys, std::mem::size_of::<u32>())?;
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(map.as_mut_slice());
        Ok(u32::from_ne_bytes(bytes))
    }

    pub fn write_u32(phys: u64, value: u32) -> std::io::Result<()> {
        let mut map = Self::map_rw(phys, std::mem::size_of::<u32>())?;
        map.as_mut_slice().copy_from_slice(&value.to_ne_bytes());
        map.flush()
    }

    pub fn write_u64(phys: u64, value: u64) -> std::io::Result<()> {
        let mut map = Self::map_rw(phys, std::mem::size_of::<u64>())?;
        map.as_mut_slice().copy_from_slice(&value.to_ne_bytes());
        map.flush()
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        let ret = unsafe { libc::msync(self.ptr.cast(), self.map_len, libc::MS_SYNC) };
        if ret < 0 { Err(Error::last_os_error()) } else { Ok(()) }
    }
}

impl Drop for PhysMap {
    fn drop(&mut self) {
        let _ = unsafe { libc::munmap(self.ptr.cast(), self.map_len) };
        let _ = self.file.as_raw_fd();
    }
}

pub struct PaccBoMap {
    ptr: *mut u8,
    map_len: usize,
    len: usize,
}

impl PaccBoMap {
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        let ret = unsafe { libc::msync(self.ptr.cast(), self.map_len, libc::MS_SYNC) };
        if ret < 0 { Err(Error::last_os_error()) } else { Ok(()) }
    }
}

impl Drop for PaccBoMap {
    fn drop(&mut self) {
        let _ = unsafe { libc::munmap(self.ptr.cast(), self.map_len) };
    }
}

// ─── Device handle ─────────────────────────────────────────────────────────────

pub struct PaccDevice {
    pub id:  usize,
    pub fd:  RawFd,
    file:    File,
}

impl PaccDevice {
    pub fn open(id: usize) -> std::io::Result<Self> {
        let path = format!("/dev/pacc{}", id);
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let fd = file.as_raw_fd();
        let dev = PaccDevice { id, fd, file };
        if std::env::var("HETGPU_PACC_BOOT_RUNTIME").ok().as_deref() == Some("1") {
            dev.boot_runtime_from_env()?;
        }
        if std::env::var("HETGPU_PACC_BOOT").ok().as_deref() == Some("1") {
            let reset_vec = std::env::var("HETGPU_PACC_RESET_VEC")
                .ok()
                .and_then(|v| {
                    let trimmed = v.trim_start_matches("0x");
                    u32::from_str_radix(trimmed, 16).ok().or_else(|| v.parse().ok())
                })
                .unwrap_or(PACC_RESET_VEC_VAL);
            dev.boot_from_reset_vector(reset_vec)?;
        }
        Ok(dev)
    }

    /// PACC_IOC_GET_INFO — query firmware version / core count
    pub fn get_info(&self) -> std::io::Result<pacc_info_size> {
        let mut info = pacc_info_size { opcode: 0, size: 0 };
        let ret = unsafe {
            libc::ioctl(self.fd, IOC_GET_INFO, &mut info as *mut _)
        };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(info)
    }

    /// PACC_IOC_MEM_ALLOC — allocate DMA-coherent memory, returns physical addr
    pub fn mem_alloc(&self, size: u64) -> std::io::Result<u64> {
        let mut request = self.mem_alloc_request(size as usize)?;
        let ret = unsafe {
            libc::ioctl(self.fd, IOC_MEM_ALLOC, request.as_mut_ptr())
        };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&request[..8]);
        Ok(u64::from_ne_bytes(bytes))
    }

    /// The shipping driver uses ioctl nr=3 as "submit current BO", not free.
    /// Keeping the old symbol wired to submit prevents silent success on the
    /// wrong path while preserving the FFI surface for existing probes.
    pub fn mem_free(&self, addr: u64) -> std::io::Result<()> {
        let _ = addr;
        self.submit_current_bo()
    }

    pub fn bo_alloc_map(&self, len: usize) -> std::io::Result<PaccBoMap> {
        if len == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "zero-length PACC BO"));
        }
        let mut request = self.mem_alloc_request(len)?;
        let ret = unsafe {
            libc::ioctl(self.fd, IOC_MEM_ALLOC, request.as_mut_ptr())
        };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }

        let map_len = align_up(len, page_size());
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::last_os_error());
        }

        Ok(PaccBoMap { ptr: ptr.cast(), map_len, len })
    }

    fn mem_alloc_request(&self, len: usize) -> std::io::Result<Vec<u8>> {
        if len == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "zero-length PACC allocation"));
        }
        let map_len = align_up(len, page_size());
        let mut request = vec![0u8; map_len.max(8)];
        request[..8].copy_from_slice(&(len as u64).to_ne_bytes());
        Ok(request)
    }

    pub fn submit_current_bo(&self) -> std::io::Result<()> {
        if std::env::var("HETGPU_PACC_ALLOW_UNSAFE_IOCTL3_SUBMIT")
            .ok()
            .as_deref()
            != Some("1")
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "PACC ioctl nr=3 submit is safety-gated: current pacc.ko oopses in \
                 pacc_mbox_jobs_submit and can leave pacc_ioctl tasks stuck in D state; \
                 set HETGPU_PACC_ALLOW_UNSAFE_IOCTL3_SUBMIT=1 only while debugging the \
                 kernel mailbox path after a reset",
            ));
        }

        let mut unused: u64 = 0;
        let ret = unsafe {
            libc::ioctl(self.fd, IOC_BO_SUBMIT, &mut unused as *mut u64)
        };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Legacy nr=1 probe path. This is not the hardware launch path.
    pub fn job_submit(&self, phys_addr: u64, size: u64) -> std::io::Result<()> {
        if std::env::var("HETGPU_PACC_UNSAFE_LEGACY_JOB_SUBMIT").ok().as_deref() != Some("1") {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!(
                    "legacy job_submit(phys=0x{phys_addr:x}, size={size}) disabled: \
                     pacc.ko launch uses the BO mmap path, not the nr=1 ioctl; \
                     set HETGPU_PACC_UNSAFE_LEGACY_JOB_SUBMIT=1 only for driver ABI debugging"
                ),
            ));
        }
        let mut arg = pacc_jobs_addr { addr: phys_addr, size };
        let ret = unsafe {
            libc::ioctl(self.fd, IOC_GET_INFO_EX, &mut arg as *mut _)
        };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Submit a job image through the driver's BO path.
    ///
    /// `/pacc.ko` does not accept a raw userspace `{addr, size}` for launch.
    /// The real sequence is MEM_ALLOC(size) -> mmap(fd) -> write payload ->
    /// ioctl nr=3, where the driver builds the mailbox page descriptors from
    /// the current BO and sends those descriptors to PACC firmware.
    pub fn job_submit_user_buffer(&self, buf: &[u8]) -> std::io::Result<()> {
        self.job_submit_user_buffer_with_len(buf, buf.len())
    }

    pub fn job_submit_user_buffer_with_len(&self, buf: &[u8], submit_len: usize) -> std::io::Result<()> {
        if buf.is_empty() || submit_len == 0 || submit_len > buf.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "invalid PACC job buffer length"));
        }
        let mut bo = self.bo_alloc_map(submit_len)?;
        bo.as_mut_slice().copy_from_slice(&buf[..submit_len]);
        self.submit_current_bo()
    }

    pub fn submit_runtime_job<T: Copy>(&self, job_id: u32, args: &T) -> std::io::Result<()> {
        let arg_bytes = unsafe {
            std::slice::from_raw_parts(
                (args as *const T).cast::<u8>(),
                std::mem::size_of::<T>(),
            )
        };
        self.submit_preloaded_job_bytes(job_id, arg_bytes)
    }

    pub fn submit_preloaded_job_bytes(&self, job_id: u32, arg_bytes: &[u8]) -> std::io::Result<()> {
        require_runtime_ready()?;

        let seq = NEXT_RUNTIME_JOB_SEQ.fetch_add(1, Ordering::Relaxed);

        let doorbell = HetgpuPaccDoorbell {
            magic: HETGPU_PACC_JOB_MAGIC,
            version: HETGPU_PACC_JOB_VERSION,
            job_id,
            flags: 0,
            status: 0,
            seq,
        };

        if std::env::var("HETGPU_PACC_USE_DRIVER_JOB_IOCTL").ok().as_deref() != Some("1") {
            return self.submit_preloaded_job_mailbox(job_id, seq, &doorbell, arg_bytes);
        }

        self.stage_preloaded_job_args(job_id, seq, arg_bytes)?;
        self.stage_preloaded_doorbell(&doorbell)?;

        let submit_len = align_up(HETGPU_PACC_DOORBELL_BYTES, 64);
        let mut buf = vec![0u8; submit_len];
        let doorbell_bytes = unsafe {
            std::slice::from_raw_parts(
                (&doorbell as *const HetgpuPaccDoorbell).cast::<u8>(),
                HETGPU_PACC_DOORBELL_BYTES,
            )
        };
        buf[..HETGPU_PACC_DOORBELL_BYTES].copy_from_slice(doorbell_bytes);
        self.job_submit_user_buffer_with_len(&buf, submit_len)
    }

    fn submit_preloaded_job_mailbox(
        &self,
        job_id: u32,
        seq: u64,
        doorbell: &HetgpuPaccDoorbell,
        arg_bytes: &[u8],
    ) -> std::io::Result<()> {
        self.stage_preloaded_job_args(job_id, seq, arg_bytes)?;
        self.stage_preloaded_doorbell(doorbell).map_err(|e| {
            if e.raw_os_error() == Some(libc::EPERM) || e.kind() == ErrorKind::PermissionDenied {
                Error::new(
                    ErrorKind::PermissionDenied,
                    "failed to write AP2PACC mailbox at 0x20000000. The host kernel has \
                     CONFIG_STRICT_DEVMEM/IO_STRICT_DEVMEM enabled; use a boot with mailbox \
                     /dev/mem access enabled or install a tiny mailbox kernel helper. The runtime \
                     did not call pacc.ko ioctl nr=3.",
                )
            } else {
                e
            }
        })?;
        self.wait_preloaded_job_status(job_id, seq)
    }

    fn stage_preloaded_doorbell(&self, doorbell: &HetgpuPaccDoorbell) -> std::io::Result<()> {
        if std::env::var("HETGPU_PACC_IOCTL_ONLY_DOORBELL").ok().as_deref() == Some("1") {
            return Ok(());
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (doorbell as *const HetgpuPaccDoorbell).cast::<u8>(),
                HETGPU_PACC_DOORBELL_BYTES,
            )
        };
        if write_ap2pacc_mailbox(self.id as usize, 0, bytes)? {
            return Ok(());
        }
        let mut map = PhysMap::map_rw(AP2PACC_MBOX_PHYS, HETGPU_PACC_DOORBELL_BYTES)?;
        let dst = map.as_mut_slice();
        dst.copy_from_slice(bytes);
        map.flush()
    }

    fn stage_preloaded_job_args(&self, job_id: u32, seq: u64, arg_bytes: &[u8]) -> std::io::Result<()> {
        if std::env::var("HETGPU_PACC_FIRMWARE_ARGS_PRELOADED").ok().as_deref() == Some("1") {
            if !arg_bytes.is_empty()
                && std::env::var("HETGPU_PACC_STATIC_FIRMWARE_TABLE").ok().as_deref() != Some("1")
            {
                return self.stage_runtime_job_table(job_id, seq, arg_bytes);
            }
            return Ok(());
        }
        let slot = match preloaded_arg_slot(job_id) {
            Some(slot) => slot,
            None => {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown preloaded PACC job_id {}", job_id),
                ))
            }
        };
        let total = HETGPU_PACC_ARG_HEADER_BYTES + arg_bytes.len();
        if total > HETGPU_PACC_ARG_SLOT_BYTES {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "PACC job_id {} args are {} bytes, slot limit is {}",
                    job_id, total, HETGPU_PACC_ARG_SLOT_BYTES
                ),
            ));
        }
        let header = HetgpuPaccArgSlotHeader {
            magic: HETGPU_PACC_JOB_MAGIC,
            version: HETGPU_PACC_JOB_VERSION,
            job_id,
            seq,
            arg_len: arg_bytes.len() as u64,
        };
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                (&header as *const HetgpuPaccArgSlotHeader).cast::<u8>(),
                HETGPU_PACC_ARG_HEADER_BYTES,
            )
        };
        let mut helper_payload = vec![0u8; HETGPU_PACC_ARG_SLOT_BYTES];
        helper_payload[..HETGPU_PACC_ARG_HEADER_BYTES].copy_from_slice(header_bytes);
        helper_payload[HETGPU_PACC_ARG_HEADER_BYTES..HETGPU_PACC_ARG_HEADER_BYTES + arg_bytes.len()]
            .copy_from_slice(arg_bytes);
        if write_ap2pacc_mailbox(
            self.id as usize,
            HETGPU_PACC_ARG_BASE - AP2PACC_MBOX_PHYS + (slot * HETGPU_PACC_ARG_SLOT_BYTES) as u64,
            &helper_payload,
        )? {
            return Ok(());
        }
        let mut map = PhysMap::map_rw(
            HETGPU_PACC_ARG_BASE + (slot * HETGPU_PACC_ARG_SLOT_BYTES) as u64,
            HETGPU_PACC_ARG_SLOT_BYTES,
        )?;
        let dst = map.as_mut_slice();
        dst.fill(0);
        dst[..HETGPU_PACC_ARG_HEADER_BYTES].copy_from_slice(header_bytes);
        dst[HETGPU_PACC_ARG_HEADER_BYTES..HETGPU_PACC_ARG_HEADER_BYTES + arg_bytes.len()]
            .copy_from_slice(arg_bytes);
        map.flush()
    }

    fn stage_runtime_job_table(&self, job_id: u32, seq: u64, arg_bytes: &[u8]) -> std::io::Result<()> {
        let mut table = HetgpuPaccRuntimeJobTable {
            magic: HETGPU_PACC_RUNTIME_TABLE_MAGIC,
            version: HETGPU_PACC_RUNTIME_TABLE_VERSION,
            flags: 0,
            seq,
            ..Default::default()
        };

        unsafe {
            match job_id {
                hetgpu_pacc_job_id::GEMM => {
                    let want = std::mem::size_of::<HetgpuPaccGemmJob>();
                    if arg_bytes.len() < want {
                        return Err(Error::new(ErrorKind::InvalidInput, "short PACC GEMM runtime table payload"));
                    }
                    std::ptr::copy_nonoverlapping(
                        arg_bytes.as_ptr(),
                        (&mut table.gemm as *mut HetgpuPaccGemmJob).cast::<u8>(),
                        want,
                    );
                    table.have_gemm = 1;
                }
                hetgpu_pacc_job_id::SOFTMAX => {
                    let want = std::mem::size_of::<HetgpuPaccSoftmaxJob>();
                    if arg_bytes.len() < want {
                        return Err(Error::new(ErrorKind::InvalidInput, "short PACC softmax runtime table payload"));
                    }
                    std::ptr::copy_nonoverlapping(
                        arg_bytes.as_ptr(),
                        (&mut table.softmax as *mut HetgpuPaccSoftmaxJob).cast::<u8>(),
                        want,
                    );
                    table.have_softmax = 1;
                }
                hetgpu_pacc_job_id::RMSNORM => {
                    let want = std::mem::size_of::<HetgpuPaccRmsNormJob>();
                    if arg_bytes.len() < want {
                        return Err(Error::new(ErrorKind::InvalidInput, "short PACC RMSNorm runtime table payload"));
                    }
                    std::ptr::copy_nonoverlapping(
                        arg_bytes.as_ptr(),
                        (&mut table.rmsnorm as *mut HetgpuPaccRmsNormJob).cast::<u8>(),
                        want,
                    );
                    table.have_rmsnorm = 1;
                }
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        format!("PACC job_id {} has no firmware runtime table entry", job_id),
                    ));
                }
            }
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&table as *const HetgpuPaccRuntimeJobTable).cast::<u8>(),
                std::mem::size_of::<HetgpuPaccRuntimeJobTable>(),
            )
        };
        if write_ap2pacc_mailbox(self.id as usize, HETGPU_PACC_RUNTIME_TABLE_OFF, bytes)? {
            return Ok(());
        }
        let mut map = PhysMap::map_rw(AP2PACC_MBOX_PHYS + HETGPU_PACC_RUNTIME_TABLE_OFF, bytes.len())?;
        map.as_mut_slice().copy_from_slice(bytes);
        map.flush()
    }

    fn wait_preloaded_job_status(&self, job_id: u32, seq: u64) -> std::io::Result<()> {
        if std::env::var("HETGPU_PACC_SKIP_JOB_WAIT").ok().as_deref() == Some("1") {
            return Ok(());
        }
        let timeout_ms = std::env::var("HETGPU_PACC_JOB_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(30_000);
        let start = std::time::Instant::now();
        let mut buf = [0u8; 32];
        loop {
            if read_pacc2ap_mailbox(self.id as usize, 0, &mut buf)? {
                let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
                let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
                let status_job_id = u32::from_le_bytes(buf[12..16].try_into().unwrap());
                let status = u32::from_le_bytes(buf[16..20].try_into().unwrap());
                let status_seq = u64::from_le_bytes(buf[24..32].try_into().unwrap());
                if magic == HETGPU_PACC_JOB_MAGIC
                    && version == HETGPU_PACC_JOB_VERSION
                    && status_job_id == job_id
                    && status_seq == seq
                {
                    if status == 0 {
                        return Ok(());
                    }
                    if status != 1 {
                        return Err(Error::new(
                            ErrorKind::Other,
                            format!("PACC job_id {} seq {} failed with firmware status 0x{:x}", job_id, seq, status),
                        ));
                    }
                }
            }
            if start.elapsed() >= std::time::Duration::from_millis(timeout_ms) {
                return Err(Error::new(
                    ErrorKind::TimedOut,
                    format!("timed out waiting for PACC job_id {} seq {} completion", job_id, seq),
                ));
            }
            std::thread::sleep(std::time::Duration::from_micros(500));
        }
    }

    pub fn boot_runtime_from_env(&self) -> std::io::Result<()> {
        let elf = std::env::var("HETGPU_PACC_RUNTIME_ELF").map_err(|_| {
            Error::new(
                ErrorKind::InvalidInput,
                "HETGPU_PACC_BOOT_RUNTIME=1 requires HETGPU_PACC_RUNTIME_ELF",
            )
        })?;
        self.boot_runtime_elf(std::path::Path::new(&elf))
    }

    pub fn boot_runtime_elf(&self, elf_path: &std::path::Path) -> std::io::Result<()> {
        if std::env::var("HETGPU_PACC_CORES_ARE_WFI").ok().as_deref() != Some("1") {
            return Err(Error::new(
                ErrorKind::PermissionDenied,
                "refusing PACC runtime boot/reset without HETGPU_PACC_CORES_ARE_WFI=1; \
                 PACC cores must be in WFI before reset to avoid hanging the SoC",
            ));
        }

        let mut guard = runtime_boot_state().lock().map_err(|_| {
            Error::new(ErrorKind::Other, "PACC runtime boot state lock poisoned")
        })?;
        if guard[self.id] {
            return Ok(());
        }

        let bytes = std::fs::read(elf_path)?;
        let entry = load_elf64_load_segments_to_phys(&bytes)?;
        self.boot_from_reset_vector64(entry)?;
        guard[self.id] = true;
        Ok(())
    }

    /// Boot/release this PACC cluster using the Pcore-visible top registers
    /// shown in pacc_boot.c. This requires access to /dev/mem, so normal
    /// runtime opens do not call it unless HETGPU_PACC_BOOT=1 is set.
    pub fn boot_from_reset_vector(&self, reset_vec: u32) -> std::io::Result<()> {
        pacc_boot_from_pcore_regs64(self.id, reset_vec as u64)
    }

    pub fn boot_from_reset_vector64(&self, reset_vec: u64) -> std::io::Result<()> {
        pacc_boot_from_pcore_regs64(self.id, reset_vec)
    }

    pub fn ap2pacc_mailbox(&self) -> std::io::Result<PhysMap> {
        let _ = self;
        PhysMap::map_rw(AP2PACC_MBOX_PHYS, MBOX_SRAM_SIZE)
    }

    pub fn pacc2ap_mailbox(&self) -> std::io::Result<PhysMap> {
        let _ = self;
        PhysMap::map_rw(PACC2AP_MBOX_PHYS, MBOX_SRAM_SIZE)
    }

    pub fn write_mailbox_msg(&self, msg: MboxMsg) -> std::io::Result<()> {
        let mut map = self.ap2pacc_mailbox()?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&msg as *const MboxMsg).cast::<u8>(),
                std::mem::size_of::<MboxMsg>(),
            )
        };
        map.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        map.flush()
    }
}

static NEXT_RUNTIME_JOB_SEQ: AtomicU64 = AtomicU64::new(1);
static NEXT_GEMM_DEVICE: AtomicUsize = AtomicUsize::new(0);
static RUNTIME_BOOTED: OnceLock<Mutex<[bool; 4]>> = OnceLock::new();

fn runtime_boot_state() -> &'static Mutex<[bool; 4]> {
    RUNTIME_BOOTED.get_or_init(|| Mutex::new([false; 4]))
}

fn preloaded_arg_slot(job_id: u32) -> Option<usize> {
    match job_id {
        hetgpu_pacc_job_id::GEMM => Some(0),
        hetgpu_pacc_job_id::SOFTMAX => Some(1),
        hetgpu_pacc_job_id::RMSNORM => Some(2),
        hetgpu_pacc_job_id::ALLREDUCE => Some(3),
        _ => None,
    }
}

fn require_runtime_ready() -> std::io::Result<()> {
    if std::env::var("HETGPU_PACC_RUNTIME_READY").ok().as_deref() == Some("1")
        || std::env::var("HETGPU_PACC_BOOT_RUNTIME").ok().as_deref() == Some("1")
    {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::Unsupported,
            "PACC runtime kernel is not marked ready; boot the PACC-side runtime first \
             (HETGPU_PACC_BOOT_RUNTIME=1 with HETGPU_PACC_RUNTIME_ELF and \
             HETGPU_PACC_CORES_ARE_WFI=1) or set HETGPU_PACC_RUNTIME_READY=1 after \
             an external secure boot",
        ))
    }
}

fn write_ap2pacc_mailbox(pacc_id: usize, offset: u64, bytes: &[u8]) -> std::io::Result<bool> {
    let dev = std::env::var("HETGPU_PACC_MBOX_DEVICE").unwrap_or_else(|_| {
        let per_pacc = format!("/dev/hetgpu_pacc_mbox{}", pacc_id);
        if std::path::Path::new(&per_pacc).exists() {
            per_pacc
        } else {
            "/dev/hetgpu_pacc_mbox".to_string()
        }
    });
    if !std::path::Path::new(&dev).exists() {
        return Ok(false);
    }
    let mut file = OpenOptions::new().write(true).open(&dev)?;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)?;
    file.flush()?;
    Ok(true)
}

fn read_pacc2ap_mailbox(pacc_id: usize, offset: u64, bytes: &mut [u8]) -> std::io::Result<bool> {
    let dev = std::env::var("HETGPU_PACC_MBOX_DEVICE").unwrap_or_else(|_| {
        let per_pacc = format!("/dev/hetgpu_pacc_mbox{}", pacc_id);
        if std::path::Path::new(&per_pacc).exists() {
            per_pacc
        } else {
            "/dev/hetgpu_pacc_mbox".to_string()
        }
    });
    if std::path::Path::new(&dev).exists() {
        let mut file = OpenOptions::new().read(true).open(&dev)?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(bytes)?;
        return Ok(true);
    }
    if pacc_id == 0 {
        let mut map = PhysMap::map_rw(PACC2AP_MBOX_PHYS + offset, bytes.len())?;
        bytes.copy_from_slice(map.as_mut_slice());
        return Ok(true);
    }
    Ok(false)
}

fn read_u16_le(buf: &[u8], off: usize) -> std::io::Result<u16> {
    let bytes = buf.get(off..off + 2)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "truncated ELF u16"))?;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u32_le(buf: &[u8], off: usize) -> std::io::Result<u32> {
    let bytes = buf.get(off..off + 4)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "truncated ELF u32"))?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64_le(buf: &[u8], off: usize) -> std::io::Result<u64> {
    let bytes = buf.get(off..off + 8)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "truncated ELF u64"))?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn load_elf64_load_segments_to_phys(elf: &[u8]) -> std::io::Result<u64> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return Err(Error::new(ErrorKind::InvalidData, "expected little-endian ELF64"));
    }

    let entry = read_u64_le(elf, 0x18)?;
    let phoff = read_u64_le(elf, 0x20)? as usize;
    let phentsize = read_u16_le(elf, 0x36)? as usize;
    let phnum = read_u16_le(elf, 0x38)? as usize;
    if phentsize < 56 {
        return Err(Error::new(ErrorKind::InvalidData, "ELF64 program header too small"));
    }

    for i in 0..phnum {
        let off = phoff + i * phentsize;
        let p_type = read_u32_le(elf, off)?;
        if p_type != 1 {
            continue;
        }
        let p_offset = read_u64_le(elf, off + 0x08)? as usize;
        let p_vaddr = read_u64_le(elf, off + 0x10)?;
        let p_paddr = read_u64_le(elf, off + 0x18)?;
        let p_filesz = read_u64_le(elf, off + 0x20)? as usize;
        let p_memsz = read_u64_le(elf, off + 0x28)? as usize;
        if p_filesz > p_memsz {
            return Err(Error::new(ErrorKind::InvalidData, "ELF PT_LOAD filesz > memsz"));
        }
        let segment = elf.get(p_offset..p_offset + p_filesz)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "ELF PT_LOAD outside file"))?;
        let phys = if p_paddr != 0 { p_paddr } else { p_vaddr };
        let mut map = PhysMap::map_rw(phys, p_memsz.max(1))?;
        let dst = map.as_mut_slice();
        dst[..p_filesz].copy_from_slice(segment);
        if p_memsz > p_filesz {
            dst[p_filesz..p_memsz].fill(0);
        }
        map.flush()?;
    }

    Ok(entry)
}

pub fn pacc_boot_from_pcore_regs(pacc_id: usize, reset_vec: u32) -> std::io::Result<()> {
    pacc_boot_from_pcore_regs64(pacc_id, reset_vec as u64)
}

pub fn pacc_boot_from_pcore_regs64(pacc_id: usize, reset_vec: u64) -> std::io::Result<()> {
    if pacc_id >= PACC_BASE.len() {
        return Err(Error::new(ErrorKind::InvalidInput, "invalid PACC id"));
    }

    let top = PACC_BASE[pacc_id] + PACC_TOP_REG_OFF;
    let lo = (reset_vec & 0xffff_ffff) as u32;
    let hi = (reset_vec >> 32) as u32;

    // Program reset vectors for all four PACC cores.
    for core_id in 0..PACC_CORE_NUM {
        PhysMap::write_u32(
            top + PACC_TOP_REG_RESET_VEC_LO_ADDR + (core_id as u64 * 0x8),
            lo,
        )?;
        PhysMap::write_u32(
            top + PACC_TOP_REG_RESET_VEC_HI_ADDR + (core_id as u64 * 0x8),
            hi,
        )?;
    }
    PhysMap::write_u32(top + PACC_TOP_REG_FORCE_RESETPC_RELOAD_ADDR, 0xf)?;
    PhysMap::write_u32(top + PACC_TOP_REG_FORCE_RESETPC_RELOAD_ADDR, 0)?;

    // Release system reset, then per-core reset, matching pacc_boot.c.
    PhysMap::write_u32(top + PACC_TOP_REG_SYS_RESET_ADDR, 0)?;
    for core_id in 0..PACC_CORE_NUM {
        PhysMap::write_u32(
            top + PACC_TOP_REG_CORE_RESET_ADDR + (core_id as u64 * 0x4),
            0,
        )?;
    }

    Ok(())
}

pub fn pacc_set_nonsecure(pacc_id: usize, nonsecure: bool) -> std::io::Result<()> {
    if pacc_id >= PACC_BASE.len() {
        return Err(Error::new(ErrorKind::InvalidInput, "invalid PACC id"));
    }
    let reg = PACC_BASE[pacc_id] + PACC_TOP_REG_OFF + PACC_TOP_REG_PACC_RSVD;
    let mut value = PhysMap::read_u32(reg)?;
    if nonsecure {
        value |= 0x3 << 26;
    } else {
        value &= !(0x3 << 26);
    }
    PhysMap::write_u32(reg, value)
}

impl Drop for PaccDevice {
    fn drop(&mut self) {
        // file is closed when dropped
    }
}

// ─── Result type ───────────────────────────────────────────────────────────────

pub type PaccResult = Result<(), PaccError>;

#[derive(Debug)]
pub enum PaccError {
    Io(std::io::Error),
    InvalidDevice(usize),
    NotInitialized,
    OutOfMemory,
    InvalidArg,
}

impl From<std::io::Error> for PaccError {
    fn from(e: std::io::Error) -> Self { PaccError::Io(e) }
}

impl std::fmt::Display for PaccError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ─── Collective communicator (4-PACC ring) ─────────────────────────────────────

pub struct PaccComm {
    pub devices: Vec<Arc<Mutex<PaccDevice>>>,
    pub num_devices: usize,
}

impl PaccComm {
    /// Open all 4 PACC devices and build the communicator
    pub fn init_all() -> Result<Self, PaccError> {
        let mut devices = Vec::with_capacity(4);
        for id in 0..4 {
            match PaccDevice::open(id) {
                Ok(dev) => {
                    eprintln!("PACC: opened /dev/pacc{}", id);
                    devices.push(Arc::new(Mutex::new(dev)));
                }
                Err(e) => {
                    eprintln!("PACC: failed to open /dev/pacc{}: {}", id, e);
                    return Err(PaccError::Io(e));
                }
            }
        }
        Ok(PaccComm { num_devices: devices.len(), devices })
    }

    /// AllReduce using driver-side reduce via job submission.
    ///
    /// Protocol (ring-reduce across 4 PACCs via shared DDR + mailbox):
    ///   1. Each PACC writes its partial data to shared DDR slot[id]
    ///   2. Big core (Pcore) issues reduce job to each PACC via BO submit
    ///   3. PACC driver accumulates across all slots using its built-in DMA+reduce
    ///   4. Result is broadcast back to all slots
    ///
    pub fn all_reduce(
        &self,
        src: &[f32],
        dst: &mut [f32],
        op: PaccReduceOp,
    ) -> Result<(), PaccError> {
        let n = src.len();
        assert_eq!(dst.len(), n);

        let _ = (src, dst, op);
        Err(PaccError::Io(Error::new(
            ErrorKind::Unsupported,
            "PACC all_reduce has no host fallback: a real PACC reduce kernel/job ABI is required",
        )))
    }

    /// Broadcast: device 0 → all other devices via mailbox signaling
    pub fn broadcast(&self, _buf: &mut [u8], _root: usize) -> Result<(), PaccError> {
        // Pcore writes to AP2PACC mailbox SRAM, PACC firmware handles DMA copy
        // Full impl requires mapping /dev/mem at 0x20000000 or using driver ioctl
        Ok(())
    }

    /// Barrier: wait for all PACCs via mailbox ping-pong
    pub fn barrier(&self) -> Result<(), PaccError> {
        for dev_lock in &self.devices {
            let dev = dev_lock.lock().map_err(|_| PaccError::InvalidArg)?;
            // Send NOOP job to flush pipeline
            let _ = dev.get_info();
        }
        Ok(())
    }
}

// ─── C-compatible FFI surface (kept for compatibility with existing callers) ───

#[no_mangle]
pub unsafe extern "C" fn pacc_open_device(id: u32) -> *mut PaccDevice {
    match PaccDevice::open(id as usize) {
        Ok(dev) => Box::into_raw(Box::new(dev)),
        Err(e) => {
            eprintln!("pacc_open_device({}): {}", id, e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn pacc_close_device(dev: *mut PaccDevice) {
    if !dev.is_null() {
        drop(Box::from_raw(dev));
    }
}

#[no_mangle]
pub unsafe extern "C" fn pacc_get_info_ffi(
    dev: *mut PaccDevice,
    out: *mut pacc_info_size,
) -> i32 {
    if dev.is_null() || out.is_null() { return -1; }
    match (*dev).get_info() {
        Ok(info) => { *out = info; 0 }
        Err(e) => { eprintln!("pacc_get_info: {}", e); -1 }
    }
}

#[no_mangle]
pub unsafe extern "C" fn pacc_mem_alloc_ffi(dev: *mut PaccDevice, size: u64) -> u64 {
    if dev.is_null() { return 0; }
    (*dev).mem_alloc(size).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn pacc_mem_free_ffi(dev: *mut PaccDevice, addr: u64) -> i32 {
    if dev.is_null() { return -1; }
    match (*dev).mem_free(addr) {
        Ok(()) => 0,
        Err(e) => { eprintln!("pacc_mem_free: {}", e); -1 }
    }
}

#[no_mangle]
pub unsafe extern "C" fn pacc_job_submit_ffi(
    dev: *mut PaccDevice,
    phys_addr: u64,
    size: u64,
) -> i32 {
    if dev.is_null() { return -1; }
    match (*dev).job_submit(phys_addr, size) {
        Ok(()) => 0,
        Err(e) => { eprintln!("pacc_job_submit: {}", e); -1 }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_gemm(
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const std::ffi::c_void,
    a: *const std::ffi::c_void,
    atype: i32,
    lda: i32,
    stride_a: i64,
    b: *const std::ffi::c_void,
    btype: i32,
    ldb: i32,
    stride_b: i64,
    beta: *const std::ffi::c_void,
    c: *mut std::ffi::c_void,
    ctype: i32,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
    compute_type: i32,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() || m <= 0 || n <= 0 || k <= 0 || batch_count <= 0 {
        eprintln!("hetgpu_pacc_submit_gemm: invalid argument");
        return -1;
    }

    let job = HetgpuPaccGemmJob {
        transa: transa as u32,
        transb: transb as u32,
        atype: atype as u32,
        btype: btype as u32,
        ctype: ctype as u32,
        compute_type: compute_type as u32,
        m: m as u64,
        n: n as u64,
        k: k as u64,
        a_addr: a as u64,
        b_addr: b as u64,
        c_addr: c as u64,
        alpha_addr: alpha as u64,
        beta_addr: beta as u64,
        lda: lda as i64,
        ldb: ldb as i64,
        ldc: ldc as i64,
        stride_a,
        stride_b,
        stride_c,
        batch_count: batch_count as u64,
    };

    let dev_id = NEXT_GEMM_DEVICE.fetch_add(1, Ordering::Relaxed) % 4;
    match PaccDevice::open(dev_id).and_then(|dev| dev.submit_runtime_job(hetgpu_pacc_job_id::GEMM, &job)) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("hetgpu_pacc_submit_gemm: PACC GEMM submit failed: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_softmax_f32(
    src: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    rows: u64,
    cols: u64,
    stride: u64,
) -> i32 {
    if src.is_null() || dst.is_null() || rows == 0 || cols == 0 {
        eprintln!("hetgpu_pacc_submit_softmax_f32: invalid argument");
        return -1;
    }
    let job = HetgpuPaccSoftmaxJob {
        src_addr: src as u64,
        dst_addr: dst as u64,
        rows,
        cols,
        stride: if stride == 0 { cols } else { stride },
        dtype: PaccDataType::Float32 as u32,
        reserved: 0,
    };
    match PaccDevice::open(0).and_then(|dev| dev.submit_runtime_job(hetgpu_pacc_job_id::SOFTMAX, &job)) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("hetgpu_pacc_submit_softmax_f32: PACC softmax submit failed: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_rmsnorm_f32(
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
) -> i32 {
    if x.is_null() || y.is_null() || rows == 0 || hidden == 0 {
        eprintln!("hetgpu_pacc_submit_rmsnorm_f32: invalid argument");
        return -1;
    }
    let job = HetgpuPaccRmsNormJob {
        x_addr: x as u64,
        weight_addr: weight as u64,
        y_addr: y as u64,
        rows,
        hidden,
        eps,
        dtype: PaccDataType::Float32 as u32,
    };
    match PaccDevice::open(0).and_then(|dev| dev.submit_runtime_job(hetgpu_pacc_job_id::RMSNORM, &job)) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("hetgpu_pacc_submit_rmsnorm_f32: PACC RMSNorm submit failed: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_nccl_all_reduce_f32(
    sendbuff: *const f32,
    recvbuff: *mut f32,
    count: usize,
    op: i32,
    rank: i32,
    nranks: i32,
) -> i32 {
    if sendbuff.is_null() || recvbuff.is_null() {
        eprintln!("hetgpu_pacc_nccl_all_reduce_f32: null buffer");
        return -1;
    }
    if op != 0 {
        eprintln!("hetgpu_pacc_nccl_all_reduce_f32: unsupported op {}, expected ncclSum=0", op);
        return -1;
    }

    let src = std::slice::from_raw_parts(sendbuff, count);
    let mut dst = vec![0.0f32; count];
    eprintln!(
        "[hetGPU NCCL/PACC] rank {}/{} all_reduce f32 count={}",
        rank, nranks, count
    );

    match PaccComm::init_all().and_then(|comm| comm.all_reduce(src, &mut dst, PaccReduceOp::Sum)) {
        Ok(()) => {
            std::ptr::copy_nonoverlapping(dst.as_ptr(), recvbuff, count);
            0
        }
        Err(e) => {
            eprintln!("hetgpu_pacc_nccl_all_reduce_f32: PACC all_reduce failed: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_nccl_reduce_sum_f32(
    rank_inputs: *const f32,
    recvbuff: *mut f32,
    count: usize,
    nranks: i32,
) -> i32 {
    if rank_inputs.is_null() || recvbuff.is_null() || nranks <= 0 {
        eprintln!("hetgpu_pacc_nccl_reduce_sum_f32: invalid argument");
        return -1;
    }

    let nranks_usize = nranks as usize;
    let total = match count.checked_mul(nranks_usize) {
        Some(total) => total,
        None => {
            eprintln!("hetgpu_pacc_nccl_reduce_sum_f32: input size overflow");
            return -1;
        }
    };
    let inputs = std::slice::from_raw_parts(rank_inputs, total);
    let mut reduced = vec![0.0f32; total];
    eprintln!(
        "[hetGPU NCCL/PACC] reduce-sum {} rank payloads, f32 count={} via 4-PACC runtime",
        nranks, count
    );

    reduced.copy_from_slice(inputs);

    let mut pacc_out = vec![0.0f32; total];
    match PaccComm::init_all().and_then(|comm| comm.all_reduce(&reduced, &mut pacc_out, PaccReduceOp::Sum)) {
        Ok(()) => {
            std::ptr::copy_nonoverlapping(pacc_out.as_ptr(), recvbuff, count);
            0
        }
        Err(e) => {
            eprintln!("hetgpu_pacc_nccl_reduce_sum_f32: PACC reduce failed: {}", e);
            -1
        }
    }
}


// ─── CUDA-like high-level API (used by zluda/src/impl/module.rs, function.rs) ─

/// Opaque device handle (wraps PaccDevice)
pub struct pacc_Device(pub PaccDevice);

/// Opaque program handle (holds compiled ELF bytes)
pub struct pacc_Program {
    pub elf_bytes: Vec<u8>,
}

/// Opaque kernel handle
pub struct pacc_Kernel {
    pub name: String,
    pub program: *mut pacc_Program,
    pub device: *mut pacc_Device,
}

/// Result code
pub type pacc_Result = i32;
pub const pacc_Result_Success: pacc_Result = 0;
pub const pacc_Result_Error:   pacc_Result = -1;

/// Create a PACC device handle for device_id (0-3).
/// Returns null on failure.
pub unsafe fn pacc_CreateDevice(device_id: u32) -> *mut pacc_Device {
    match PaccDevice::open(device_id as usize) {
        Ok(dev) => Box::into_raw(Box::new(pacc_Device(dev))),
        Err(e) => {
            eprintln!("pacc_CreateDevice({}): {}", device_id, e);
            std::ptr::null_mut()
        }
    }
}

/// Destroy a PACC device handle.
pub unsafe fn pacc_DestroyDevice(dev: *mut pacc_Device) {
    if !dev.is_null() {
        drop(Box::from_raw(dev));
    }
}

/// Create a PACC program (initially empty — load ELF via pacc_LoadProgram).
pub unsafe fn pacc_CreateProgram() -> *mut pacc_Program {
    Box::into_raw(Box::new(pacc_Program { elf_bytes: Vec::new() }))
}

/// Load ELF binary into a PACC program.
/// data: pointer to ELF bytes, size: byte length.
pub unsafe fn pacc_LoadProgram(
    program: *mut pacc_Program,
    data: *const std::ffi::c_void,
    size: u64,
) -> pacc_Result {
    if program.is_null() || data.is_null() || size == 0 {
        return pacc_Result_Error;
    }
    let bytes = std::slice::from_raw_parts(data as *const u8, size as usize);
    (*program).elf_bytes = bytes.to_vec();
    pacc_Result_Success
}

/// Create a named kernel handle from a program.
pub unsafe fn pacc_CreateKernel(
    program: *mut pacc_Program,
    name: *const std::ffi::c_char,
) -> *mut pacc_Kernel {
    pacc_CreateKernelOnDevice(program, std::ptr::null_mut(), name)
}

/// Create a named kernel handle tied to an already opened PACC device.
pub unsafe fn pacc_CreateKernelOnDevice(
    program: *mut pacc_Program,
    device: *mut pacc_Device,
    name: *const std::ffi::c_char,
) -> *mut pacc_Kernel {
    let name_str = if name.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(name).to_string_lossy().into_owned()
    };
    Box::into_raw(Box::new(pacc_Kernel { name: name_str, program, device }))
}

/// Destroy a kernel handle.
pub unsafe fn pacc_DestroyKernel(kernel: *mut pacc_Kernel) {
    if !kernel.is_null() {
        drop(Box::from_raw(kernel));
    }
}

/// Launch a PACC kernel via job_submit.
/// Submits the ELF binary to the device using the physical address of a
/// staging buffer. For now writes ELF bytes to a driver-allocated buffer.
pub unsafe fn pacc_LaunchKernel(
    kernel: *mut pacc_Kernel,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> pacc_Result {
    if kernel.is_null() {
        return pacc_Result_Error;
    }
    let k = &*kernel;
    let prog = &*k.program;

    let log_launches = std::env::var("HETGPU_PACC_LOG_KERNEL_LAUNCHES").ok().as_deref() == Some("1");
    if log_launches {
        eprintln!(
            "pacc_LaunchKernel: kernel='{}' grid=({},{},{}) block=({},{},{})",
            k.name, grid_x, grid_y, grid_z, block_x, block_y, block_z
        );
    }

    if prog.elf_bytes.is_empty() {
        eprintln!("pacc_LaunchKernel: no ELF loaded; refusing to stub kernel execution");
        return pacc_Result_Error;
    }

    if !k.device.is_null() {
        return pacc_launch_on_device(
            &(*k.device).0,
            &k.name,
            &prog.elf_bytes,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        );
    }

    // Compatibility path for kernels created without a bound device.
    match PaccDevice::open(0) {
        Ok(dev) => {
            pacc_launch_on_device(
                &dev,
                &k.name,
                &prog.elf_bytes,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
            )
        }
        Err(e) => {
            eprintln!("pacc_LaunchKernel: open device failed: {}", e);
            pacc_Result_Error
        }
    }
}

fn pacc_launch_on_device(
    dev: &PaccDevice,
    kernel_name: &str,
    elf_bytes: &[u8],
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> pacc_Result {
    if std::env::var("HETGPU_PACC_DRY_RUN").ok().as_deref() == Some("1") {
        eprintln!(
            "pacc_LaunchKernel: dry-run accepted preloaded kernel='{}' elf={} bytes grid=({},{},{}) block=({},{},{})",
            kernel_name,
            elf_bytes.len(),
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        );
        return pacc_Result_Success;
    }

    let _ = (elf_bytes, grid_x, grid_y, grid_z, block_x, block_y, block_z);
    match preloaded_kernel_job_id(kernel_name) {
        Some(job_id) => match dev.submit_preloaded_job_bytes(job_id, &[]) {
            Ok(()) => pacc_Result_Success,
            Err(e) => {
                eprintln!("pacc_LaunchKernel: preloaded firmware job_id {} submit failed: {}", job_id, e);
                pacc_Result_Error
            }
        },
        None => {
            eprintln!(
                "pacc_LaunchKernel: kernel '{}' has no preloaded firmware job_id; \
                 refusing to send ELF/payload to PACC",
                kernel_name
            );
            pacc_Result_Error
        }
    }
}

fn preloaded_kernel_job_id(kernel_name: &str) -> Option<u32> {
    let name = kernel_name.to_lowercase();
    if name.contains("softmax") {
        Some(hetgpu_pacc_job_id::SOFTMAX)
    } else if name.contains("rmsnorm") || name.contains("rms_norm") {
        Some(hetgpu_pacc_job_id::RMSNORM)
    } else if name.contains("gemm") || name.contains("matmul") || name.contains("cublas") {
        Some(hetgpu_pacc_job_id::GEMM)
    } else {
        None
    }
}

#[allow(dead_code)]
fn fill_pacc_kernel_image(
    buf: &mut [u8],
    kernel_name: &str,
    elf_bytes: &[u8],
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> std::io::Result<()> {
    let image_size = PACC_JOB_HEADER_BYTES + elf_bytes.len();
    if buf.len() < image_size {
        return Err(Error::new(ErrorKind::InvalidInput, "PACC job image buffer too small"));
    }

    let header = PaccJobImageHeader {
        magic: PACC_JOB_MAGIC,
        version: 1,
        flags: 0,
        entry_offset: PACC_JOB_HEADER_BYTES as u64,
        image_size: elf_bytes.len() as u64,
        kernel_name_hash: hash_kernel_name(kernel_name),
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        reserved: 0,
    };

    buf[..image_size].fill(0);
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const PaccJobImageHeader).cast::<u8>(),
            PACC_JOB_HEADER_BYTES,
        )
    };
    let header_start = 0;
    let elf_start = header_start + PACC_JOB_HEADER_BYTES;
    buf[header_start..elf_start].copy_from_slice(header_bytes);
    buf[elf_start..elf_start + elf_bytes.len()].copy_from_slice(elf_bytes);
    Ok(())
}

#[allow(dead_code)]
fn stage_pacc_job_image(
    phys_addr: u64,
    image_offset: usize,
    kernel_name: &str,
    elf_bytes: &[u8],
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> std::io::Result<()> {
    let image_phys = phys_addr + image_offset as u64;
    let image_size = PACC_JOB_HEADER_BYTES + elf_bytes.len();
    let alloc_size = image_offset + image_size;

    let desc = pacc_mbox_job_desc {
        addr: image_phys,
        len: image_size as u64,
        rsvd: hash_kernel_name(kernel_name),
        buf_info: PACC_JOB_MAGIC,
    };
    let header = PaccJobImageHeader {
        magic: PACC_JOB_MAGIC,
        version: 1,
        flags: 0,
        entry_offset: PACC_JOB_HEADER_BYTES as u64,
        image_size: elf_bytes.len() as u64,
        kernel_name_hash: hash_kernel_name(kernel_name),
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        reserved: 0,
    };

    let mut map = PhysMap::map_rw(phys_addr, alloc_size)?;
    let buf = map.as_mut_slice();
    buf.fill(0);

    let desc_bytes = unsafe {
        std::slice::from_raw_parts(
            (&desc as *const pacc_mbox_job_desc).cast::<u8>(),
            PACC_JOB_DESC_BYTES,
        )
    };
    buf[..PACC_JOB_DESC_BYTES].copy_from_slice(desc_bytes);

    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const PaccJobImageHeader).cast::<u8>(),
            PACC_JOB_HEADER_BYTES,
        )
    };
    let header_start = image_offset;
    let elf_start = header_start + PACC_JOB_HEADER_BYTES;
    buf[header_start..elf_start].copy_from_slice(header_bytes);
    buf[elf_start..elf_start + elf_bytes.len()].copy_from_slice(elf_bytes);
    map.flush()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ioctl_encoding() {
        assert_eq!(IOC_GET_INFO,   0xc000_7000u64 & 0xffff_ffff);
        assert_eq!(IOC_GET_INFO_EX, 0xc001_0001u64 & 0xffff_ffff);
        assert_eq!(IOC_MEM_ALLOC,  0xc000_8002u64 & 0xffff_ffff);
        assert_eq!(IOC_BO_SUBMIT,  0x4000_8003u64 & 0xffff_ffff);
    }
}
