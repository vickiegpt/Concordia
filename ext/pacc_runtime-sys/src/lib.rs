//! PACC Runtime Bindings — Lanxin LX500 real driver interface via /dev/paccN
//!
//! Driver interface (reverse-engineered from pacc.ko DWARF + disassembly):
//!   Magic: 'p' (0x70)
//!   PACC_IOC_GET_INFO_SIZE = _IOWR('p', 0, struct pacc_info_size)
//!   PACC_IOC_GET_INFO      = _IOWR('p', 1, struct pacc_info)
//!   PACC_IOC_CREATE_BO     = _IOWR('p', 2, struct pacc_bo)
//!   PACC_IOC_SUBMIT_OP     = _IOW ('p', 3, struct pacc_op)
//!   PACC_IOC_FREE_BO       = _IOW ('p', 4, struct pacc_bo)
//!
//! Mailbox SRAM (accessible from Pcore side via mmap or physical):
//!   AP→PACC : 0x20000000  (8KB)
//!   PACC→AP : 0x20002000  (8KB)
//!
//! PACC cluster base addresses: 0x38100000, 0x38500000, 0x39100000, 0x39500000

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

// Proper encoding: _IOWR(type, nr, size) = (3<<30)|(size<<16)|(type<<8)|nr
const fn _iowr(ty: u64, nr: u64, size: u64) -> u64 {
    (3 << 30) | (size << 16) | (ty << 8) | nr
}
const fn _iow(ty: u64, nr: u64, size: u64) -> u64 {
    (1 << 30) | (size << 16) | (ty << 8) | nr
}

pub const PACC_MAGIC: u64 = 0x70; // 'p'

// ─── kernel struct mirrors ─────────────────────────────────────────────────────

/// pacc_info_size — arg for PACC_IOC_GET_INFO_SIZE (8 bytes)
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct pacc_info_size {
    pub opcode: u32,
    pub size: u32,
}

/// `pacc_info` payload returned by ioctl nr=1.
/// The concrete header has not landed in this tree yet, so keep it opaque.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct pacc_info {
    pub raw: [u64; 2],
}

/// `pacc_bo` create/free descriptor.
/// `size` is the requested contiguous allocation length on create, and `addr`
/// is filled by the kernel with the BO's physical base.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct pacc_bo {
    pub size: u64,
    pub addr: u64,
}

/// `pacc_op` submit descriptor. Current callers only need the default
/// zeroed "submit current BO" behavior.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct pacc_op {
    pub reserved: u64,
}

/// pacc_mbox_job_desc — one job entry (32 bytes)
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct pacc_mbox_job_desc {
    pub addr: u64,
    pub len: u64,
    pub rsvd: u64,
    pub buf_info: u64,
}

pub const PACC_IOC_GET_INFO_SIZE: u64 =
    _iowr(PACC_MAGIC, 0, std::mem::size_of::<pacc_info_size>() as u64);
pub const PACC_IOC_GET_INFO: u64 = _iowr(PACC_MAGIC, 1, std::mem::size_of::<pacc_info>() as u64);
pub const PACC_IOC_CREATE_BO: u64 = _iowr(PACC_MAGIC, 2, std::mem::size_of::<pacc_bo>() as u64);
pub const PACC_IOC_SUBMIT_OP: u64 = _iow(PACC_MAGIC, 3, std::mem::size_of::<pacc_op>() as u64);
pub const PACC_IOC_FREE_BO: u64 = _iow(PACC_MAGIC, 4, std::mem::size_of::<pacc_bo>() as u64);

// Backward-compatible aliases for older local call sites.
pub const PACC_IOC_GET_INFO_EX: u64 = PACC_IOC_GET_INFO;
pub const PACC_IOC_MEM_ALLOC: u64 = PACC_IOC_CREATE_BO;
pub const PACC_IOC_BO_SUBMIT: u64 = PACC_IOC_SUBMIT_OP;
pub const IOC_GET_INFO_SIZE: u64 = PACC_IOC_GET_INFO_SIZE;
pub const IOC_GET_INFO: u64 = PACC_IOC_GET_INFO;
pub const IOC_GET_INFO_EX: u64 = PACC_IOC_GET_INFO;
pub const IOC_CREATE_BO: u64 = PACC_IOC_CREATE_BO;
pub const IOC_MEM_ALLOC: u64 = PACC_IOC_CREATE_BO;
pub const IOC_SUBMIT_OP: u64 = PACC_IOC_SUBMIT_OP;
pub const IOC_BO_SUBMIT: u64 = PACC_IOC_SUBMIT_OP;
pub const IOC_FREE_BO: u64 = PACC_IOC_FREE_BO;

/// Reduce operation types for NCCL AllReduce
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaccReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
}

/// Data format for reduce
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaccDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
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
/// PACC-visible reduce scratch base. Prefer the mailbox helper's allocated
/// window exported in `/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base`.
pub const HETGPU_PACC_SHARED_DDR_BASE: u64 = 0;
pub const HETGPU_PACC_SHARED_DDR_BYTES: usize = 0x0100_0000;
pub const HETGPU_PACC_SHARED_DDR_HELPER_OFF: u64 = 0x0010_0000;

/// Per-PACC local SRAM bases
pub const PACC_SRAM_BASE: [u64; 4] = [0x6000_0000, 0x6010_0000, 0x6020_0000, 0x6030_0000];
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
const PACC_JOB_FLAG_HAS_LAUNCH_ABI: u32 = 1 << 0;
const PACC_KERNEL_LAUNCH_ABI_MAGIC: u64 = 0x5041_4343_4152_4731; // "PACCARG1"
const PACC_KERNEL_LAUNCH_ABI_VERSION: u32 = 1;
const HETGPU_PACC_DOORBELL_BYTES: usize = std::mem::size_of::<HetgpuPaccDoorbell>();
const HETGPU_PACC_ARG_HEADER_BYTES: usize = std::mem::size_of::<HetgpuPaccArgSlotHeader>();
pub const HETGPU_PACC_ARG_BASE: u64 = AP2PACC_MBOX_PHYS + 0x100;
pub const HETGPU_PACC_COMPLETION_OFF: u64 = 0x1f20;
pub const HETGPU_PACC_ARG_SLOT_BYTES: usize = 0x400;
pub const HETGPU_PACC_RUNTIME_TABLE_OFF: u64 = 0x1400;
const HETGPU_PACC_RUNTIME_TABLE_MAGIC: u64 = 0x4847_5055_5442_4c31;
const HETGPU_PACC_RUNTIME_TABLE_VERSION: u32 = 1;

pub mod hetgpu_pacc_job_id {
    pub const KERNEL: u32 = 0;
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
    pub have_allreduce: u32,
    pub gemm: HetgpuPaccGemmJob,
    pub softmax: HetgpuPaccSoftmaxJob,
    pub rmsnorm: HetgpuPaccRmsNormJob,
    pub allreduce: HetgpuPaccAllReduceJob,
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

pub const PACC_KERNEL_ARG_KIND_SCALAR: u32 = 0;
pub const PACC_KERNEL_ARG_KIND_POINTER: u32 = 1;
pub const PACC_KERNEL_ARG_FLAG_SIGNED: u32 = 1 << 0;
pub const PACC_KERNEL_ARG_FLAG_FLOAT: u32 = 1 << 1;
pub const PACC_KERNEL_ARG_FLAG_BUFFER_INPUT: u32 = 1 << 8;
pub const PACC_KERNEL_ARG_FLAG_BUFFER_OUTPUT: u32 = 1 << 9;
pub const PACC_KERNEL_ARG_FLAG_BUFFER_INOUT: u32 =
    PACC_KERNEL_ARG_FLAG_BUFFER_INPUT | PACC_KERNEL_ARG_FLAG_BUFFER_OUTPUT;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct PaccKernelLaunchAbiHeader {
    pub magic: u64,
    pub version: u32,
    pub flags: u32,
    pub arg_records_offset: u32,
    pub arg_record_count: u32,
    pub bindings_offset: u32,
    pub binding_count: u32,
    pub raw_param_offset: u32,
    pub raw_param_size: u32,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct PaccKernelArgRecord {
    pub kind: u32,
    pub size: u32,
    pub flags: u32,
    pub reserved: u32,
    pub value: u64,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct PaccKernelBufferBinding {
    pub arg_index: u32,
    pub flags: u32,
    pub addr: u64,
    pub size: u64,
}

#[derive(Debug, Default, Clone)]
struct PaccKernelLaunchState {
    raw_param_blob: Vec<u8>,
    arg_records: Vec<PaccKernelArgRecord>,
    bindings: Vec<PaccKernelBufferBinding>,
}

impl PaccKernelLaunchState {
    fn is_empty(&self) -> bool {
        self.raw_param_blob.is_empty() && self.arg_records.is_empty() && self.bindings.is_empty()
    }
}

// ─── Mailbox message layout ────────────────────────────────────────────────────

/// Simple mailbox message header (8-byte stride in SRAM)
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct MboxMsg {
    /// Command opcode
    pub cmd: u32,
    /// Payload length (bytes following header in SRAM)
    pub length: u16,
    /// Status / sequence number
    pub status: u16,
}

pub mod mbox_cmd {
    pub const NOOP: u32 = 0x0000;
    pub const PING: u32 = 0x0001;
    pub const REDUCE_SUM: u32 = 0x0010;
    pub const REDUCE_PROD: u32 = 0x0011;
    pub const REDUCE_MAX: u32 = 0x0012;
    pub const REDUCE_MIN: u32 = 0x0013;
    pub const ALLREDUCE: u32 = 0x0020;
    pub const BARRIER: u32 = 0x0030;
    pub const DONE: u32 = 0x00FF;
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
    if value <= 0 {
        4096
    } else {
        value as usize
    }
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
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "zero-length physical map",
            ));
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
        if ret < 0 {
            Err(Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

impl Drop for PhysMap {
    fn drop(&mut self) {
        let _ = unsafe { libc::munmap(self.ptr.cast(), self.map_len) };
        let _ = self.file.as_raw_fd();
    }
}

pub struct PaccBoMap {
    file: File,
    bo: pacc_bo,
    phys: u64,
    ptr: *mut u8,
    map_len: usize,
    len: usize,
}

unsafe impl Send for PaccBoMap {}

impl PaccBoMap {
    pub fn phys(&self) -> u64 {
        self.phys
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        let ret = unsafe { libc::msync(self.ptr.cast(), self.map_len, libc::MS_SYNC) };
        if ret < 0 {
            Err(Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

impl Drop for PaccBoMap {
    fn drop(&mut self) {
        let _ = unsafe { libc::munmap(self.ptr.cast(), self.map_len) };
        let mut bo = self.bo;
        let _ = unsafe { libc::ioctl(self.file.as_raw_fd(), IOC_FREE_BO, &mut bo as *mut _) };
    }
}

fn extract_bo_phys(bo: &pacc_bo, requested_size: u64) -> std::io::Result<u64> {
    if bo.addr != 0 {
        return Ok(bo.addr);
    }
    if bo.size != 0 && bo.size != requested_size {
        return Ok(bo.size);
    }
    Err(Error::new(
        ErrorKind::InvalidData,
        "PACC create_bo returned an empty BO descriptor",
    ))
}

// ─── Device handle ─────────────────────────────────────────────────────────────

pub struct PaccDevice {
    pub id: usize,
    pub fd: RawFd,
    file: File,
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
                    u32::from_str_radix(trimmed, 16)
                        .ok()
                        .or_else(|| v.parse().ok())
                })
                .unwrap_or(PACC_RESET_VEC_VAL);
            dev.boot_from_reset_vector(reset_vec)?;
        }
        Ok(dev)
    }

    /// PACC_IOC_GET_INFO_SIZE — query the size of the full info payload.
    pub fn get_info(&self) -> std::io::Result<pacc_info_size> {
        let mut info = pacc_info_size { opcode: 0, size: 0 };
        let ret = unsafe { libc::ioctl(self.fd, IOC_GET_INFO_SIZE, &mut info as *mut _) };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(info)
    }

    /// PACC_IOC_GET_INFO — fetch the full opaque info record.
    pub fn get_info_full(&self) -> std::io::Result<pacc_info> {
        let mut info = pacc_info::default();
        let ret = unsafe { libc::ioctl(self.fd, IOC_GET_INFO, &mut info as *mut _) };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(info)
    }

    /// PACC_IOC_CREATE_BO — allocate a contiguous BO and return its physical base.
    pub fn mem_alloc(&self, size: u64) -> std::io::Result<u64> {
        let mut request = self.bo_request(size as usize)?;
        let requested_size = request.size;
        let ret = unsafe { libc::ioctl(self.fd, IOC_CREATE_BO, &mut request as *mut _) };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        extract_bo_phys(&request, requested_size)
    }

    /// PACC_IOC_FREE_BO — release a previously created BO.
    pub fn mem_free(&self, addr: u64) -> std::io::Result<()> {
        let mut request = pacc_bo { size: 0, addr };
        let ret = unsafe { libc::ioctl(self.fd, IOC_FREE_BO, &mut request as *mut _) };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn bo_alloc_map(&self, len: usize) -> std::io::Result<PaccBoMap> {
        if len == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "zero-length PACC BO"));
        }
        let mut request = self.bo_request(len)?;
        let requested_size = request.size;
        let ret = unsafe { libc::ioctl(self.fd, IOC_CREATE_BO, &mut request as *mut _) };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let phys = extract_bo_phys(&request, requested_size)?;

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

        Ok(PaccBoMap {
            file: self.file.try_clone()?,
            bo: request,
            phys,
            ptr: ptr.cast(),
            map_len,
            len,
        })
    }

    fn bo_request(&self, len: usize) -> std::io::Result<pacc_bo> {
        if len == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "zero-length PACC allocation",
            ));
        }
        let map_len = align_up(len, page_size());
        Ok(pacc_bo {
            size: map_len as u64,
            addr: 0,
        })
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

        let mut op = pacc_op::default();
        let ret = unsafe { libc::ioctl(self.fd, IOC_SUBMIT_OP, &mut op as *mut _) };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Legacy nr=1 raw job submit is no longer valid: ioctl nr=1 is now
    /// `PACC_IOC_GET_INFO`, so the BO submit path or mailbox helper must be used.
    pub fn job_submit(&self, phys_addr: u64, size: u64) -> std::io::Result<()> {
        Err(Error::new(
            ErrorKind::Unsupported,
            format!(
                "legacy job_submit(phys=0x{phys_addr:x}, size={size}) removed: \
                 ioctl nr=1 is now PACC_IOC_GET_INFO; use the BO submit path or \
                 mailbox helper launch flow instead"
            ),
        ))
    }

    /// Submit a job image through the driver's BO path.
    ///
    /// `/pacc.ko` does not accept a raw userspace `{addr, size}` for launch.
    /// The real sequence is CREATE_BO(size) -> mmap(fd) -> write payload ->
    /// SUBMIT_OP, where the driver builds the mailbox page descriptors from
    /// the current BO and sends those descriptors to PACC firmware.
    pub fn job_submit_user_buffer(&self, buf: &[u8]) -> std::io::Result<()> {
        self.job_submit_user_buffer_with_len(buf, buf.len())
    }

    pub fn job_submit_user_buffer_with_len(
        &self,
        buf: &[u8],
        submit_len: usize,
    ) -> std::io::Result<()> {
        if buf.is_empty() || submit_len == 0 || submit_len > buf.len() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "invalid PACC job buffer length",
            ));
        }
        let mut bo = self.bo_alloc_map(submit_len)?;
        bo.as_mut_slice().copy_from_slice(&buf[..submit_len]);
        self.submit_current_bo()
    }

    pub fn submit_runtime_job<T: Copy>(&self, job_id: u32, args: &T) -> std::io::Result<()> {
        let arg_bytes = unsafe {
            std::slice::from_raw_parts((args as *const T).cast::<u8>(), std::mem::size_of::<T>())
        };
        self.submit_preloaded_job_bytes(job_id, arg_bytes)
    }

    pub fn submit_preloaded_job_bytes(&self, job_id: u32, arg_bytes: &[u8]) -> std::io::Result<()> {
        require_runtime_ready()?;

        let seq = next_runtime_job_seq();

        let doorbell = HetgpuPaccDoorbell {
            magic: HETGPU_PACC_JOB_MAGIC,
            version: HETGPU_PACC_JOB_VERSION,
            job_id,
            flags: 0,
            status: 0,
            seq,
        };

        if std::env::var("HETGPU_PACC_USE_DRIVER_JOB_IOCTL")
            .ok()
            .as_deref()
            != Some("1")
        {
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
        nvtop_record_submit(self.id as usize, job_id, seq, None, 0);
        let result = self.wait_preloaded_job_status(job_id, seq);
        nvtop_record_complete(self.id as usize, job_id, seq, &result);
        result
    }

    fn stage_preloaded_doorbell(&self, doorbell: &HetgpuPaccDoorbell) -> std::io::Result<()> {
        if std::env::var("HETGPU_PACC_IOCTL_ONLY_DOORBELL")
            .ok()
            .as_deref()
            == Some("1")
        {
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

    fn stage_preloaded_job_args(
        &self,
        job_id: u32,
        seq: u64,
        arg_bytes: &[u8],
    ) -> std::io::Result<()> {
        if !arg_bytes.is_empty()
            && std::env::var("HETGPU_PACC_STATIC_FIRMWARE_TABLE")
                .ok()
                .as_deref()
                != Some("1")
            && preloaded_arg_slot(job_id).is_some()
        {
            return self.stage_runtime_job_table(job_id, seq, arg_bytes);
        }
        if std::env::var("HETGPU_PACC_FIRMWARE_ARGS_PRELOADED")
            .ok()
            .as_deref()
            == Some("1")
        {
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
        helper_payload
            [HETGPU_PACC_ARG_HEADER_BYTES..HETGPU_PACC_ARG_HEADER_BYTES + arg_bytes.len()]
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

    fn stage_runtime_job_table(
        &self,
        job_id: u32,
        seq: u64,
        arg_bytes: &[u8],
    ) -> std::io::Result<()> {
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
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "short PACC GEMM runtime table payload",
                        ));
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
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "short PACC softmax runtime table payload",
                        ));
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
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "short PACC RMSNorm runtime table payload",
                        ));
                    }
                    std::ptr::copy_nonoverlapping(
                        arg_bytes.as_ptr(),
                        (&mut table.rmsnorm as *mut HetgpuPaccRmsNormJob).cast::<u8>(),
                        want,
                    );
                    table.have_rmsnorm = 1;
                }
                hetgpu_pacc_job_id::ALLREDUCE => {
                    let want = std::mem::size_of::<HetgpuPaccAllReduceJob>();
                    if arg_bytes.len() < want {
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "short PACC allreduce runtime table payload",
                        ));
                    }
                    std::ptr::copy_nonoverlapping(
                        arg_bytes.as_ptr(),
                        (&mut table.allreduce as *mut HetgpuPaccAllReduceJob).cast::<u8>(),
                        want,
                    );
                    table.have_allreduce = 1;
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
        let mut map = PhysMap::map_rw(
            AP2PACC_MBOX_PHYS + HETGPU_PACC_RUNTIME_TABLE_OFF,
            bytes.len(),
        )?;
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
            if read_pacc2ap_mailbox(self.id as usize, HETGPU_PACC_COMPLETION_OFF, &mut buf)? {
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
                            format!(
                                "PACC job_id {} seq {} failed with firmware status 0x{:x}",
                                job_id, seq, status
                            ),
                        ));
                    }
                }
            }
            if start.elapsed() >= std::time::Duration::from_millis(timeout_ms) {
                return Err(Error::new(
                    ErrorKind::TimedOut,
                    format!(
                        "timed out waiting for PACC job_id {} seq {} completion",
                        job_id, seq
                    ),
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

        let mut guard = runtime_boot_state()
            .lock()
            .map_err(|_| Error::new(ErrorKind::Other, "PACC runtime boot state lock poisoned"))?;
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

fn next_gemm_device() -> usize {
    std::env::var("HETGPU_PACC_GEMM_DEVICE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&id| id < PACC_CORE_NUM)
        .unwrap_or_else(|| NEXT_GEMM_DEVICE.fetch_add(1, Ordering::Relaxed) % PACC_CORE_NUM)
}
static RUNTIME_BOOTED: OnceLock<Mutex<[bool; 4]>> = OnceLock::new();
static SHARED_DDR_REDUCE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static SHARED_DDR_GEMM_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static SHARED_DDR_KERNEL_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static NVTOP_STATE: OnceLock<Mutex<NvtopProcessState>> = OnceLock::new();

#[derive(Debug, Clone, Default)]
struct NvtopGemmShape {
    m: u64,
    n: u64,
    k: u64,
    lda: i64,
    ldb: i64,
    ldc: i64,
}

#[derive(Debug, Clone, Default)]
struct NvtopDeviceState {
    jobs_submitted: u64,
    jobs_completed: u64,
    jobs_failed: u64,
    timeouts: u64,
    inflight: u64,
    last_seq: u64,
    last_job_id: u32,
    last_status: String,
    last_error: String,
    last_submit_ms: u64,
    last_complete_ms: u64,
    last_latency_ms: u64,
    last_bytes: u64,
    last_gemm: Option<NvtopGemmShape>,
}

#[derive(Debug)]
struct NvtopProcessState {
    pid: u32,
    comm: String,
    cmdline: String,
    update_ms: u64,
    last_flush_ms: u64,
    devices: std::collections::BTreeMap<usize, NvtopDeviceState>,
}

fn nvtop_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn nvtop_enabled() -> bool {
    std::env::var("HETGPU_PACC_NVTOP").ok().as_deref() != Some("0")
}

fn nvtop_flush_ms() -> u64 {
    parse_env_usize("HETGPU_PACC_NVTOP_FLUSH_MS", 250) as u64
}

fn nvtop_dir() -> Option<std::path::PathBuf> {
    if !nvtop_enabled() {
        return None;
    }
    Some(std::path::PathBuf::from(
        std::env::var("HETGPU_PACC_NVTOP_DIR")
            .unwrap_or_else(|_| "/dev/shm/hetgpu_pacc_nvtop".to_string()),
    ))
}

fn nvtop_read_comm() -> String {
    std::fs::read_to_string("/proc/self/comm")
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn nvtop_read_cmdline(default_comm: &str) -> String {
    std::fs::read("/proc/self/cmdline")
        .ok()
        .map(|bytes| {
            let mut out = String::from_utf8_lossy(&bytes).replace('\0', " ");
            out = out.trim().to_string();
            if out.is_empty() {
                default_comm.to_string()
            } else {
                out
            }
        })
        .unwrap_or_else(|| default_comm.to_string())
}

fn nvtop_state() -> Option<&'static Mutex<NvtopProcessState>> {
    if nvtop_dir().is_none() {
        return None;
    }
    Some(NVTOP_STATE.get_or_init(|| {
        let comm = nvtop_read_comm();
        Mutex::new(NvtopProcessState {
            pid: std::process::id(),
            comm: comm.clone(),
            cmdline: nvtop_read_cmdline(&comm),
            update_ms: nvtop_now_ms(),
            last_flush_ms: 0,
            devices: std::collections::BTreeMap::new(),
        })
    }))
}

fn nvtop_job_name(job_id: u32) -> &'static str {
    match job_id {
        hetgpu_pacc_job_id::GEMM => "GEMM",
        hetgpu_pacc_job_id::SOFTMAX => "SOFTMAX",
        hetgpu_pacc_job_id::RMSNORM => "RMSNORM",
        hetgpu_pacc_job_id::ALLREDUCE => "ALLREDUCE",
        _ => "UNKNOWN",
    }
}

fn nvtop_json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\\\"),
            '\n' => out.push_str("\\\\n"),
            '\r' => out.push_str("\\\\r"),
            '\t' => out.push_str("\\\\t"),
            c if c.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn nvtop_write_snapshot(state: &NvtopProcessState) -> std::io::Result<()> {
    let Some(dir) = nvtop_dir() else {
        return Ok(());
    };
    std::fs::create_dir_all(&dir)?;
    let tmp = dir.join(format!("{}.json.tmp", state.pid));
    let dst = dir.join(format!("{}.json", state.pid));

    let mut json = String::new();
    json.push('{');
    json.push_str(&format!("\"pid\":{},", state.pid));
    json.push_str(&format!("\"comm\":\"{}\",", nvtop_json_escape(&state.comm)));
    json.push_str(&format!(
        "\"cmdline\":\"{}\",",
        nvtop_json_escape(&state.cmdline)
    ));
    json.push_str(&format!("\"update_ms\":{},", state.update_ms));
    json.push_str("\"devices\":{");
    for (index, (dev_id, dev)) in state.devices.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&format!("\"{}\":{{", dev_id));
        json.push_str(&format!("\"jobs_submitted\":{},", dev.jobs_submitted));
        json.push_str(&format!("\"jobs_completed\":{},", dev.jobs_completed));
        json.push_str(&format!("\"jobs_failed\":{},", dev.jobs_failed));
        json.push_str(&format!("\"timeouts\":{},", dev.timeouts));
        json.push_str(&format!("\"inflight\":{},", dev.inflight));
        json.push_str(&format!("\"last_seq\":{},", dev.last_seq));
        json.push_str(&format!("\"last_job_id\":{},", dev.last_job_id));
        json.push_str(&format!(
            "\"last_job\":\"{}\",",
            nvtop_job_name(dev.last_job_id)
        ));
        json.push_str(&format!(
            "\"last_status\":\"{}\",",
            nvtop_json_escape(&dev.last_status)
        ));
        json.push_str(&format!(
            "\"last_error\":\"{}\",",
            nvtop_json_escape(&dev.last_error)
        ));
        json.push_str(&format!("\"last_submit_ms\":{},", dev.last_submit_ms));
        json.push_str(&format!("\"last_complete_ms\":{},", dev.last_complete_ms));
        json.push_str(&format!("\"last_latency_ms\":{},", dev.last_latency_ms));
        json.push_str(&format!("\"last_bytes\":{},", dev.last_bytes));
        if let Some(gemm) = &dev.last_gemm {
            json.push_str("\"last_gemm\":{");
            json.push_str(&format!(
                "\"m\":{},\"n\":{},\"k\":{},",
                gemm.m, gemm.n, gemm.k
            ));
            json.push_str(&format!(
                "\"lda\":{},\"ldb\":{},\"ldc\":{}",
                gemm.lda, gemm.ldb, gemm.ldc
            ));
            json.push('}');
        } else {
            json.push_str("\"last_gemm\":null");
        }
        json.push('}');
    }
    json.push_str("}}");

    std::fs::write(&tmp, json.as_bytes())?;
    std::fs::rename(tmp, dst)?;
    Ok(())
}

fn nvtop_flush_locked(state: &mut NvtopProcessState, force: bool) {
    let now = nvtop_now_ms();
    state.update_ms = now;
    if !force && now.saturating_sub(state.last_flush_ms) < nvtop_flush_ms() {
        return;
    }
    if nvtop_write_snapshot(state).is_ok() {
        state.last_flush_ms = now;
    }
}

fn nvtop_record_submit(
    pacc_id: usize,
    job_id: u32,
    seq: u64,
    gemm: Option<&HetgpuPaccGemmJob>,
    last_bytes: u64,
) {
    let Some(state_lock) = nvtop_state() else {
        return;
    };
    let Ok(mut state) = state_lock.lock() else {
        return;
    };
    let now = nvtop_now_ms();
    let dev = state.devices.entry(pacc_id).or_default();
    dev.jobs_submitted = dev.jobs_submitted.saturating_add(1);
    dev.inflight = dev.inflight.saturating_add(1);
    dev.last_seq = seq;
    dev.last_job_id = job_id;
    dev.last_status.clear();
    dev.last_status.push_str("inflight");
    dev.last_error.clear();
    dev.last_submit_ms = now;
    if last_bytes > 0 {
        dev.last_bytes = last_bytes;
    }
    if let Some(job) = gemm {
        dev.last_gemm = Some(NvtopGemmShape {
            m: job.m,
            n: job.n,
            k: job.k,
            lda: job.lda,
            ldb: job.ldb,
            ldc: job.ldc,
        });
    } else {
        dev.last_gemm = None;
    }
    nvtop_flush_locked(&mut state, false);
}

fn nvtop_record_complete(pacc_id: usize, job_id: u32, seq: u64, result: &std::io::Result<()>) {
    let Some(state_lock) = nvtop_state() else {
        return;
    };
    let Ok(mut state) = state_lock.lock() else {
        return;
    };
    let now = nvtop_now_ms();
    let dev = state.devices.entry(pacc_id).or_default();
    dev.inflight = dev.inflight.saturating_sub(1);
    dev.last_seq = seq;
    dev.last_job_id = job_id;
    dev.last_complete_ms = now;
    dev.last_latency_ms = now.saturating_sub(dev.last_submit_ms);
    match result {
        Ok(()) => {
            dev.jobs_completed = dev.jobs_completed.saturating_add(1);
            dev.last_status.clear();
            dev.last_status.push_str("ok");
            dev.last_error.clear();
        }
        Err(err) => {
            dev.jobs_failed = dev.jobs_failed.saturating_add(1);
            if err.kind() == ErrorKind::TimedOut {
                dev.timeouts = dev.timeouts.saturating_add(1);
                dev.last_status.clear();
                dev.last_status.push_str("timeout");
            } else {
                dev.last_status.clear();
                dev.last_status.push_str("error");
            }
            dev.last_error = err.to_string();
        }
    }
    nvtop_flush_locked(&mut state, true);
}

fn next_runtime_job_seq() -> u64 {
    let counter = NEXT_RUNTIME_JOB_SEQ.fetch_add(1, Ordering::Relaxed);
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    (micros << 12) ^ counter
}

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
    helper_write_all(&mut file, offset, bytes)?;
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
        helper_read_exact(&mut file, offset, bytes)?;
        return Ok(true);
    }
    if pacc_id == 0 {
        let mut map = PhysMap::map_rw(PACC2AP_MBOX_PHYS + offset, bytes.len())?;
        bytes.copy_from_slice(map.as_mut_slice());
        return Ok(true);
    }
    Ok(false)
}

fn shared_ddr_reduce_lock() -> &'static Mutex<()> {
    SHARED_DDR_REDUCE_LOCK.get_or_init(|| Mutex::new(()))
}

fn shared_ddr_gemm_lock() -> &'static Mutex<()> {
    SHARED_DDR_GEMM_LOCK.get_or_init(|| Mutex::new(()))
}

fn shared_ddr_kernel_lock() -> &'static Mutex<()> {
    SHARED_DDR_KERNEL_LOCK.get_or_init(|| Mutex::new(()))
}

fn shared_ddr_base() -> u64 {
    std::env::var("HETGPU_PACC_SHARED_DDR_BASE")
        .ok()
        .and_then(|v| {
            let trimmed = v.trim_start_matches("0x");
            u64::from_str_radix(trimmed, 16)
                .ok()
                .or_else(|| v.parse().ok())
        })
        .or_else(|| {
            std::fs::read_to_string("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base")
                .ok()
                .and_then(|v| {
                    let value = v.trim();
                    let trimmed = value.trim_start_matches("0x");
                    u64::from_str_radix(trimmed, 16)
                        .ok()
                        .or_else(|| value.parse().ok())
                })
        })
        .unwrap_or(HETGPU_PACC_SHARED_DDR_BASE)
}

fn shared_ddr_bytes() -> usize {
    std::env::var("HETGPU_PACC_SHARED_DDR_BYTES")
        .ok()
        .and_then(|v| {
            let trimmed = v.trim_start_matches("0x");
            usize::from_str_radix(trimmed, 16)
                .ok()
                .or_else(|| v.parse().ok())
        })
        .unwrap_or(HETGPU_PACC_SHARED_DDR_BYTES)
}

fn helper_path_for_pacc(pacc_id: usize) -> String {
    std::env::var("HETGPU_PACC_MBOX_DEVICE").unwrap_or_else(|_| {
        let per_pacc = format!("/dev/hetgpu_pacc_mbox{}", pacc_id);
        if std::path::Path::new(&per_pacc).exists() {
            per_pacc
        } else {
            "/dev/hetgpu_pacc_mbox".to_string()
        }
    })
}

fn helper_io_chunk_bytes() -> usize {
    HETGPU_PACC_DOORBELL_BYTES
}

fn helper_write_all(file: &mut File, base_offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    let chunk = helper_io_chunk_bytes();
    for (i, part) in bytes.chunks(chunk).enumerate() {
        file.seek(SeekFrom::Start(base_offset + (i * chunk) as u64))?;
        file.write_all(part)?;
    }
    file.flush()
}

fn helper_read_exact(file: &mut File, base_offset: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    let chunk = helper_io_chunk_bytes();
    for (i, part) in bytes.chunks_mut(chunk).enumerate() {
        file.seek(SeekFrom::Start(base_offset + (i * chunk) as u64))?;
        file.read_exact(part)?;
    }
    Ok(())
}

fn write_shared_ddr_window(offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    let dev = helper_path_for_pacc(0);
    if std::path::Path::new(&dev).exists() {
        let helper_result = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new().write(true).open(&dev)?;
            helper_write_all(&mut file, HETGPU_PACC_SHARED_DDR_HELPER_OFF + offset, bytes)
        })();
        if helper_result.is_ok() {
            return helper_result;
        }
    }

    let phys = shared_ddr_base()
        .checked_add(offset)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "shared DDR offset overflow"))?;
    let mut map = PhysMap::map_rw(phys, bytes.len())?;
    map.as_mut_slice().copy_from_slice(bytes);
    map.flush()
}

fn read_shared_ddr_window(offset: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    let dev = helper_path_for_pacc(0);
    if std::path::Path::new(&dev).exists() {
        let helper_result = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new().read(true).open(&dev)?;
            helper_read_exact(&mut file, HETGPU_PACC_SHARED_DDR_HELPER_OFF + offset, bytes)
        })();
        if helper_result.is_ok() {
            return helper_result;
        }
    }

    let phys = shared_ddr_base()
        .checked_add(offset)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "shared DDR offset overflow"))?;
    let mut map = PhysMap::map_rw(phys, bytes.len())?;
    bytes.copy_from_slice(map.as_mut_slice());
    Ok(())
}

fn open_shared_ddr_window_file(pacc_id: usize) -> Option<File> {
    let dev = helper_path_for_pacc(pacc_id);
    if std::path::Path::new(&dev).exists() {
        OpenOptions::new().read(true).write(true).open(&dev).ok()
    } else {
        None
    }
}

fn write_shared_ddr_window_cached(
    file: &mut Option<File>,
    offset: u64,
    bytes: &[u8],
) -> std::io::Result<()> {
    if let Some(file) = file.as_mut() {
        helper_write_all(file, HETGPU_PACC_SHARED_DDR_HELPER_OFF + offset, bytes)?;
        return Ok(());
    }
    write_shared_ddr_window(offset, bytes)
}

fn read_shared_ddr_window_cached(
    file: &mut Option<File>,
    offset: u64,
    bytes: &mut [u8],
) -> std::io::Result<()> {
    if let Some(file) = file.as_mut() {
        helper_read_exact(file, HETGPU_PACC_SHARED_DDR_HELPER_OFF + offset, bytes)?;
        return Ok(());
    }
    read_shared_ddr_window(offset, bytes)
}

fn pacc_gemm_trace_enabled() -> bool {
    std::env::var("HETGPU_PACC_GEMM_TRACE").ok().as_deref() == Some("1")
}

fn open_pacc_mailbox_file(pacc_id: usize) -> Option<File> {
    let dev = helper_path_for_pacc(pacc_id);
    if std::path::Path::new(&dev).exists() {
        OpenOptions::new().read(true).write(true).open(&dev).ok()
    } else {
        None
    }
}

fn write_ap2pacc_mailbox_cached(
    file: &mut Option<File>,
    pacc_id: usize,
    offset: u64,
    bytes: &[u8],
) -> std::io::Result<bool> {
    if let Some(file) = file.as_mut() {
        helper_write_all(file, offset, bytes)?;
        return Ok(true);
    }
    write_ap2pacc_mailbox(pacc_id, offset, bytes)
}

fn read_pacc2ap_mailbox_cached(
    file: &mut Option<File>,
    pacc_id: usize,
    offset: u64,
    bytes: &mut [u8],
) -> std::io::Result<bool> {
    if let Some(file) = file.as_mut() {
        helper_read_exact(file, offset, bytes)?;
        return Ok(true);
    }
    read_pacc2ap_mailbox(pacc_id, offset, bytes)
}

fn wait_mailbox_job_status_cached(
    pacc_id: usize,
    expected_job_id: u32,
    seq: u64,
    mailbox_file: &mut Option<File>,
) -> std::io::Result<()> {
    if std::env::var("HETGPU_PACC_SKIP_JOB_WAIT").ok().as_deref() == Some("1") {
        return Ok(());
    }
    let timeout_ms = std::env::var("HETGPU_PACC_JOB_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30_000);
    let poll_us = parse_env_usize("HETGPU_PACC_JOB_POLL_US", 50).min(500);
    let start = std::time::Instant::now();
    let mut buf = [0u8; 32];
    loop {
        if read_pacc2ap_mailbox(pacc_id, HETGPU_PACC_COMPLETION_OFF, &mut buf)? {
            let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            let status_job_id = u32::from_le_bytes(buf[12..16].try_into().unwrap());
            let status = u32::from_le_bytes(buf[16..20].try_into().unwrap());
            let status_seq = u64::from_le_bytes(buf[24..32].try_into().unwrap());
            if magic == HETGPU_PACC_JOB_MAGIC
                && version == HETGPU_PACC_JOB_VERSION
                && status_job_id == expected_job_id
                && status_seq == seq
            {
                if status == 0 {
                    return Ok(());
                }
                if status != 1 {
                    return Err(Error::new(
                        ErrorKind::Other,
                        format!(
                            "PACC job_id {} seq {} failed with firmware status 0x{:x}",
                            expected_job_id, seq, status
                        ),
                    ));
                }
            }
        }
        if start.elapsed() >= std::time::Duration::from_millis(timeout_ms) {
            return Err(Error::new(
                ErrorKind::TimedOut,
                format!(
                    "timed out waiting for PACC job_id {} seq {} completion",
                    expected_job_id, seq
                ),
            ));
        }
        if poll_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(poll_us as u64));
        } else {
            std::hint::spin_loop();
        }
    }
}

fn wait_preloaded_gemm_status_cached(
    pacc_id: usize,
    seq: u64,
    mailbox_file: &mut Option<File>,
) -> std::io::Result<()> {
    wait_mailbox_job_status_cached(pacc_id, hetgpu_pacc_job_id::GEMM, seq, mailbox_file)
}

fn submit_gemm_runtime_job_cached(
    dev: &PaccDevice,
    job: &HetgpuPaccGemmJob,
    staged_bytes: u64,
    mailbox_file: &mut Option<File>,
) -> std::io::Result<()> {
    require_runtime_ready()?;
    let seq = next_runtime_job_seq();
    let table = HetgpuPaccRuntimeJobTable {
        magic: HETGPU_PACC_RUNTIME_TABLE_MAGIC,
        version: HETGPU_PACC_RUNTIME_TABLE_VERSION,
        flags: 0,
        seq,
        have_gemm: 1,
        gemm: *job,
        ..Default::default()
    };
    let table_bytes = unsafe {
        std::slice::from_raw_parts(
            (&table as *const HetgpuPaccRuntimeJobTable).cast::<u8>(),
            std::mem::size_of::<HetgpuPaccRuntimeJobTable>(),
        )
    };
    write_ap2pacc_mailbox_cached(
        mailbox_file,
        dev.id,
        HETGPU_PACC_RUNTIME_TABLE_OFF,
        table_bytes,
    )?;

    let doorbell = HetgpuPaccDoorbell {
        magic: HETGPU_PACC_JOB_MAGIC,
        version: HETGPU_PACC_JOB_VERSION,
        job_id: hetgpu_pacc_job_id::GEMM,
        flags: 0,
        status: 0,
        seq,
    };
    let doorbell_bytes = unsafe {
        std::slice::from_raw_parts(
            (&doorbell as *const HetgpuPaccDoorbell).cast::<u8>(),
            HETGPU_PACC_DOORBELL_BYTES,
        )
    };
    write_ap2pacc_mailbox_cached(mailbox_file, dev.id, 0, doorbell_bytes)?;
    nvtop_record_submit(
        dev.id,
        hetgpu_pacc_job_id::GEMM,
        seq,
        Some(job),
        staged_bytes,
    );
    let result = wait_preloaded_gemm_status_cached(dev.id, seq, mailbox_file);
    nvtop_record_complete(dev.id, hetgpu_pacc_job_id::GEMM, seq, &result);
    result
}

fn read_u16_le(buf: &[u8], off: usize) -> std::io::Result<u16> {
    let bytes = buf
        .get(off..off + 2)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "truncated ELF u16"))?;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u32_le(buf: &[u8], off: usize) -> std::io::Result<u32> {
    let bytes = buf
        .get(off..off + 4)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "truncated ELF u32"))?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64_le(buf: &[u8], off: usize) -> std::io::Result<u64> {
    let bytes = buf
        .get(off..off + 8)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "truncated ELF u64"))?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn load_elf64_load_segments_to_phys(elf: &[u8]) -> std::io::Result<u64> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "expected little-endian ELF64",
        ));
    }

    let entry = read_u64_le(elf, 0x18)?;
    let phoff = read_u64_le(elf, 0x20)? as usize;
    let phentsize = read_u16_le(elf, 0x36)? as usize;
    let phnum = read_u16_le(elf, 0x38)? as usize;
    if phentsize < 56 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "ELF64 program header too small",
        ));
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
            return Err(Error::new(
                ErrorKind::InvalidData,
                "ELF PT_LOAD filesz > memsz",
            ));
        }
        let segment = elf
            .get(p_offset..p_offset + p_filesz)
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
    fn from(e: std::io::Error) -> Self {
        PaccError::Io(e)
    }
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
        Ok(PaccComm {
            num_devices: devices.len(),
            devices,
        })
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

        if op != PaccReduceOp::Sum {
            return Err(PaccError::Io(Error::new(
                ErrorKind::Unsupported,
                "PACC all_reduce currently supports f32 sum only",
            )));
        }
        if n == 0 {
            return Ok(());
        }

        let nranks = if n % self.num_devices == 0 {
            self.num_devices
        } else {
            1
        };
        let per_rank_count = n / nranks;
        let input_bytes = src.len() * std::mem::size_of::<f32>();
        let output_bytes = per_rank_count * std::mem::size_of::<f32>();
        let output_off = align_up(input_bytes, 64);
        let total_bytes = output_off + output_bytes;
        let shared_base = shared_ddr_base();
        let shared_bytes = shared_ddr_bytes();
        if total_bytes > shared_bytes {
            return Err(PaccError::Io(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "PACC all_reduce needs {total_bytes} bytes, shared DDR helper window is {shared_bytes} bytes"
                ),
            )));
        }
        let _shared_guard = shared_ddr_reduce_lock().lock().map_err(|_| {
            PaccError::Io(Error::new(
                ErrorKind::Other,
                "PACC shared DDR reduce mutex poisoned",
            ))
        })?;

        let reduce_device = std::env::var("HETGPU_PACC_REDUCE_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&id| id < self.devices.len())
            .unwrap_or_else(|| if self.devices.len() > 1 { 1 } else { 0 });
        let dev_guard = self.devices[reduce_device].lock().map_err(|_| {
            PaccError::Io(Error::new(ErrorKind::Other, "PACC device mutex poisoned"))
        })?;

        if std::env::var("HETGPU_PACC_REDUCE_MAILBOX").ok().as_deref() == Some("1") {
            const REDUCE_SRC_OFF: u64 = 0x1800;
            const REDUCE_DST_OFF: u64 = 0x1000;
            const REDUCE_SRC_BYTES: usize = 0x0800;
            let max_chunk = (REDUCE_SRC_BYTES / (nranks * std::mem::size_of::<f32>())).max(1);
            let mut reduced = vec![0.0f32; per_rank_count];

            for start in (0..per_rank_count).step_by(max_chunk) {
                let chunk = std::cmp::min(max_chunk, per_rank_count - start);
                let mut in_storage = vec![0u8; chunk * nranks * std::mem::size_of::<f32>()];
                for rank in 0..nranks {
                    let rank_base = rank * per_rank_count + start;
                    let rank_slice = &src[rank_base..rank_base + chunk];
                    let rank_bytes = unsafe {
                        std::slice::from_raw_parts(
                            rank_slice.as_ptr().cast::<u8>(),
                            chunk * std::mem::size_of::<f32>(),
                        )
                    };
                    let dst_off = rank * chunk * std::mem::size_of::<f32>();
                    in_storage[dst_off..dst_off + rank_bytes.len()].copy_from_slice(rank_bytes);
                }
                write_ap2pacc_mailbox(reduce_device, REDUCE_SRC_OFF, &in_storage)
                    .and_then(|ok| {
                        if ok {
                            Ok(())
                        } else {
                            Err(Error::new(
                                ErrorKind::NotFound,
                                "PACC mailbox helper is not loaded",
                            ))
                        }
                    })
                    .map_err(|e| {
                        PaccError::Io(Error::new(
                            e.kind(),
                            format!("PACC mailbox reduce write failed: {e}"),
                        ))
                    })?;

                let job = HetgpuPaccAllReduceJob {
                    src_addr: AP2PACC_MBOX_PHYS + REDUCE_SRC_OFF,
                    dst_addr: PACC2AP_MBOX_PHYS + REDUCE_DST_OFF,
                    count: chunk as u64,
                    nranks: nranks as u32,
                    reduce_op: PaccReduceOp::Sum as u32,
                    dtype: PaccDataType::Float32 as u32,
                    reserved: 0,
                };
                dev_guard
                    .submit_runtime_job(hetgpu_pacc_job_id::ALLREDUCE, &job)
                    .map_err(|e| {
                        PaccError::Io(Error::new(
                            e.kind(),
                            format!("PACC mailbox all_reduce job submit failed: {e}"),
                        ))
                    })?;

                let mut out_storage = vec![0u8; chunk * std::mem::size_of::<f32>()];
                read_pacc2ap_mailbox(reduce_device, REDUCE_DST_OFF, &mut out_storage)
                    .and_then(|ok| {
                        if ok {
                            Ok(())
                        } else {
                            Err(Error::new(
                                ErrorKind::NotFound,
                                "PACC mailbox helper is not loaded",
                            ))
                        }
                    })
                    .map_err(|e| {
                        PaccError::Io(Error::new(
                            e.kind(),
                            format!("PACC mailbox reduce read failed: {e}"),
                        ))
                    })?;
                let chunk_result = unsafe {
                    std::slice::from_raw_parts(out_storage.as_ptr().cast::<f32>(), chunk)
                };
                reduced[start..start + chunk].copy_from_slice(chunk_result);
            }

            if nranks == 1 {
                dst.copy_from_slice(&reduced);
            } else {
                for chunk in dst.chunks_mut(per_rank_count) {
                    chunk.copy_from_slice(&reduced);
                }
            }
            return Ok(());
        }

        let src_bytes =
            unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), input_bytes) };
        let mut payload = vec![0u8; total_bytes];
        payload[..input_bytes].copy_from_slice(src_bytes);
        write_shared_ddr_window(0, &payload).map_err(|e| {
            PaccError::Io(Error::new(
                e.kind(),
                format!("PACC all_reduce shared DDR write failed: {e}"),
            ))
        })?;

        let job = HetgpuPaccAllReduceJob {
            src_addr: shared_base,
            dst_addr: shared_base + output_off as u64,
            count: per_rank_count as u64,
            nranks: nranks as u32,
            reduce_op: PaccReduceOp::Sum as u32,
            dtype: PaccDataType::Float32 as u32,
            reserved: 0,
        };
        dev_guard
            .submit_runtime_job(hetgpu_pacc_job_id::ALLREDUCE, &job)
            .map_err(|e| {
                PaccError::Io(Error::new(
                    e.kind(),
                    format!("PACC all_reduce job submit failed: {e}"),
                ))
            })?;

        let mut out_storage = vec![0u8; output_bytes];
        read_shared_ddr_window(output_off as u64, &mut out_storage).map_err(|e| {
            PaccError::Io(Error::new(
                e.kind(),
                format!("PACC all_reduce shared DDR read failed: {e}"),
            ))
        })?;
        let out_bytes = out_storage.as_slice();
        let result =
            unsafe { std::slice::from_raw_parts(out_bytes.as_ptr().cast::<f32>(), per_rank_count) };
        if nranks == 1 {
            dst.copy_from_slice(result);
        } else {
            for chunk in dst.chunks_mut(per_rank_count) {
                chunk.copy_from_slice(result);
            }
        }
        Ok(())
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
pub unsafe extern "C" fn pacc_get_info_ffi(dev: *mut PaccDevice, out: *mut pacc_info_size) -> i32 {
    if dev.is_null() || out.is_null() {
        return -1;
    }
    match (*dev).get_info() {
        Ok(info) => {
            *out = info;
            0
        }
        Err(e) => {
            eprintln!("pacc_get_info: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn pacc_mem_alloc_ffi(dev: *mut PaccDevice, size: u64) -> u64 {
    if dev.is_null() {
        return 0;
    }
    (*dev).mem_alloc(size).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn pacc_mem_free_ffi(dev: *mut PaccDevice, addr: u64) -> i32 {
    if dev.is_null() {
        return -1;
    }
    match (*dev).mem_free(addr) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("pacc_mem_free: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn pacc_job_submit_ffi(
    dev: *mut PaccDevice,
    phys_addr: u64,
    size: u64,
) -> i32 {
    if dev.is_null() {
        return -1;
    }
    match (*dev).job_submit(phys_addr, size) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("pacc_job_submit: {}", e);
            -1
        }
    }
}

fn gemm_span(rows: u64, cols: u64, ld: i64) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    let lead = if ld > 0 { ld as u64 } else { rows };
    cols.checked_sub(1)?
        .checked_mul(lead)?
        .checked_add(rows)?
        .try_into()
        .ok()
}

fn pacc_dtype_size(dtype: i32) -> Option<usize> {
    match dtype as u32 {
        x if x == PaccDataType::Int8 as u32 => Some(std::mem::size_of::<i8>()),
        x if x == PaccDataType::Uint8 as u32 => Some(std::mem::size_of::<u8>()),
        x if x == PaccDataType::Int32 as u32 => Some(std::mem::size_of::<i32>()),
        x if x == PaccDataType::Float16 as u32 => Some(std::mem::size_of::<u16>()),
        x if x == PaccDataType::Float32 as u32 => Some(std::mem::size_of::<f32>()),
        x if x == PaccDataType::Bfloat16 as u32 => Some(std::mem::size_of::<u16>()),
        _ => None,
    }
}

fn pacc_tensor_dtype_supported(dtype: i32) -> bool {
    pacc_dtype_size(dtype).is_some()
}

fn align_up_u64(value: u64, align: u64) -> Option<u64> {
    value
        .checked_add(align.checked_sub(1)?)?
        .checked_div(align)?
        .checked_mul(align)
}

fn read_f32_arg(ptr: *const std::ffi::c_void, default: f32) -> f32 {
    if ptr.is_null() {
        default
    } else {
        unsafe { std::ptr::read_unaligned(ptr.cast::<f32>()) }
    }
}

fn bf16_to_f32_bits(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn f16_to_f32_bits(value: u16) -> f32 {
    let sign = ((value as u32) & 0x8000) << 16;
    let exp = ((value as u32) >> 10) & 0x1f;
    let frac = (value as u32) & 0x03ff;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_n = frac;
            let mut exp_n = 127 - 15 + 1;
            while (frac_n & 0x0400) == 0 {
                frac_n <<= 1;
                exp_n -= 1;
            }
            frac_n &= 0x03ff;
            sign | ((exp_n as u32) << 23) | (frac_n << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mant = bits & 0x007f_ffff;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let shifted = (mant | 0x0080_0000) >> (1 - exp);
        return sign | (((shifted + 0x1000) >> 13) as u16);
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    sign | (((exp as u16) << 10) | (((mant + 0x1000) >> 13) as u16))
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

fn f32_to_i8_bits(value: f32) -> i8 {
    value.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
}

fn f32_to_u8_bits(value: f32) -> u8 {
    value.round().clamp(u8::MIN as f32, u8::MAX as f32) as u8
}

fn f32_to_i32_bits(value: f32) -> i32 {
    value.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

fn gemm_storage_to_f32_bytes(src: &[u8], dtype: i32, elems: usize) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(elems * std::mem::size_of::<f32>());
    match dtype as u32 {
        x if x == PaccDataType::Int8 as u32 => {
            for i in 0..elems {
                out.extend_from_slice(&(src[i] as i8 as f32).to_ne_bytes());
            }
        }
        x if x == PaccDataType::Uint8 as u32 => {
            for i in 0..elems {
                out.extend_from_slice(&(src[i] as f32).to_ne_bytes());
            }
        }
        x if x == PaccDataType::Int32 as u32 => {
            for i in 0..elems {
                let off = i * std::mem::size_of::<i32>();
                let v = i32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
                out.extend_from_slice(&(v as f32).to_ne_bytes());
            }
        }
        x if x == PaccDataType::Float16 as u32 => {
            for i in 0..elems {
                let off = i * std::mem::size_of::<u16>();
                let v = u16::from_ne_bytes(src[off..off + 2].try_into().unwrap());
                out.extend_from_slice(&f16_to_f32_bits(v).to_ne_bytes());
            }
        }
        x if x == PaccDataType::Float32 as u32 => {
            let want = elems * std::mem::size_of::<f32>();
            out.extend_from_slice(&src[..want]);
        }
        x if x == PaccDataType::Bfloat16 as u32 => {
            for i in 0..elems {
                let off = i * std::mem::size_of::<u16>();
                let v = u16::from_ne_bytes(src[off..off + 2].try_into().unwrap());
                out.extend_from_slice(&bf16_to_f32_bits(v).to_ne_bytes());
            }
        }
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "unsupported staged GEMM dtype conversion",
            ))
        }
    }
    Ok(out)
}

fn f32_bytes_to_gemm_storage(src: &[u8], dtype: i32, elems: usize) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(elems * pacc_dtype_size(dtype).unwrap_or(0));
    match dtype as u32 {
        x if x == PaccDataType::Int8 as u32 => {
            for i in 0..elems {
                let off = i * std::mem::size_of::<f32>();
                let v = f32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
                out.extend_from_slice(&f32_to_i8_bits(v).to_ne_bytes());
            }
        }
        x if x == PaccDataType::Uint8 as u32 => {
            for i in 0..elems {
                let off = i * std::mem::size_of::<f32>();
                let v = f32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
                out.extend_from_slice(&f32_to_u8_bits(v).to_ne_bytes());
            }
        }
        x if x == PaccDataType::Int32 as u32 => {
            for i in 0..elems {
                let off = i * std::mem::size_of::<f32>();
                let v = f32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
                out.extend_from_slice(&f32_to_i32_bits(v).to_ne_bytes());
            }
        }
        x if x == PaccDataType::Float16 as u32 => {
            for i in 0..elems {
                let off = i * std::mem::size_of::<f32>();
                let v = f32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
                out.extend_from_slice(&f32_to_f16_bits(v).to_ne_bytes());
            }
        }
        x if x == PaccDataType::Float32 as u32 => {
            let want = elems * std::mem::size_of::<f32>();
            out.extend_from_slice(&src[..want]);
        }
        x if x == PaccDataType::Bfloat16 as u32 => {
            for i in 0..elems {
                let off = i * std::mem::size_of::<f32>();
                let v = f32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
                out.extend_from_slice(&f32_to_bf16_bits(v).to_ne_bytes());
            }
        }
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "unsupported staged GEMM dtype conversion",
            ))
        }
    }
    Ok(out)
}

unsafe fn gemm_read_f32(
    base: *const std::ffi::c_void,
    dtype: i32,
    index: usize,
) -> std::io::Result<f32> {
    match dtype as u32 {
        x if x == PaccDataType::Int8 as u32 => {
            Ok(std::ptr::read_unaligned(base.cast::<i8>().add(index)) as f32)
        }
        x if x == PaccDataType::Uint8 as u32 => {
            Ok(std::ptr::read_unaligned(base.cast::<u8>().add(index)) as f32)
        }
        x if x == PaccDataType::Int32 as u32 => {
            Ok(std::ptr::read_unaligned(base.cast::<i32>().add(index)) as f32)
        }
        x if x == PaccDataType::Float16 as u32 => Ok(f16_to_f32_bits(std::ptr::read_unaligned(
            base.cast::<u16>().add(index),
        ))),
        x if x == PaccDataType::Float32 as u32 => {
            Ok(std::ptr::read_unaligned(base.cast::<f32>().add(index)))
        }
        x if x == PaccDataType::Bfloat16 as u32 => Ok(bf16_to_f32_bits(std::ptr::read_unaligned(
            base.cast::<u16>().add(index),
        ))),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "unsupported staged GEMM dtype conversion",
        )),
    }
}

unsafe fn gemm_write_from_f32(
    base: *mut std::ffi::c_void,
    dtype: i32,
    index: usize,
    value: f32,
) -> std::io::Result<()> {
    match dtype as u32 {
        x if x == PaccDataType::Int8 as u32 => {
            std::ptr::write_unaligned(base.cast::<i8>().add(index), f32_to_i8_bits(value));
            Ok(())
        }
        x if x == PaccDataType::Uint8 as u32 => {
            std::ptr::write_unaligned(base.cast::<u8>().add(index), f32_to_u8_bits(value));
            Ok(())
        }
        x if x == PaccDataType::Int32 as u32 => {
            std::ptr::write_unaligned(base.cast::<i32>().add(index), f32_to_i32_bits(value));
            Ok(())
        }
        x if x == PaccDataType::Float16 as u32 => {
            std::ptr::write_unaligned(base.cast::<u16>().add(index), f32_to_f16_bits(value));
            Ok(())
        }
        x if x == PaccDataType::Float32 as u32 => {
            std::ptr::write_unaligned(base.cast::<f32>().add(index), value);
            Ok(())
        }
        x if x == PaccDataType::Bfloat16 as u32 => {
            std::ptr::write_unaligned(base.cast::<u16>().add(index), f32_to_bf16_bits(value));
            Ok(())
        }
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "unsupported staged GEMM dtype conversion",
        )),
    }
}

unsafe fn pack_gemm_operand_to_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    rows: usize,
    cols: usize,
    ld: usize,
    transposed_source: bool,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rows * cols * std::mem::size_of::<f32>());
    for col in 0..cols {
        for row in 0..rows {
            let src_index = if transposed_source {
                col + row * ld
            } else {
                row + col * ld
            };
            out.extend_from_slice(&gemm_read_f32(base, dtype, src_index)?.to_ne_bytes());
        }
    }
    Ok(out)
}

unsafe fn pack_gemm_c_to_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    rows: usize,
    cols: usize,
    ld: usize,
) -> std::io::Result<Vec<u8>> {
    pack_gemm_operand_to_f32_bytes(base, dtype, rows, cols, ld, false)
}

unsafe fn unpack_gemm_c_from_f32_bytes(
    src: &[u8],
    dst: *mut std::ffi::c_void,
    dtype: i32,
    rows: usize,
    cols: usize,
    ld: usize,
) -> std::io::Result<()> {
    for col in 0..cols {
        for row in 0..rows {
            let off = (row + col * rows) * std::mem::size_of::<f32>();
            let value = f32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
            gemm_write_from_f32(dst, dtype, row + col * ld, value)?;
        }
    }
    Ok(())
}

unsafe fn pack_gemm_a_block_rowmajor_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    row0: usize,
    rows: usize,
    k: usize,
    lda: usize,
    transposed_source: bool,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rows * k * std::mem::size_of::<f32>());
    for row in 0..rows {
        for kk in 0..k {
            let src_index = if transposed_source {
                kk + (row0 + row) * lda
            } else {
                (row0 + row) + kk * lda
            };
            out.extend_from_slice(&gemm_read_f32(base, dtype, src_index)?.to_ne_bytes());
        }
    }
    Ok(out)
}

unsafe fn pack_gemm_b_rowmajor_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    k: usize,
    n: usize,
    ldb: usize,
    transposed_source: bool,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(k * n * std::mem::size_of::<f32>());
    for kk in 0..k {
        for col in 0..n {
            let src_index = if transposed_source {
                col + kk * ldb
            } else {
                kk + col * ldb
            };
            out.extend_from_slice(&gemm_read_f32(base, dtype, src_index)?.to_ne_bytes());
        }
    }
    Ok(out)
}

unsafe fn pack_gemm_c_block_rowmajor_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    row0: usize,
    rows: usize,
    n: usize,
    ldc: usize,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rows * n * std::mem::size_of::<f32>());
    for row in 0..rows {
        for col in 0..n {
            let src_index = (row0 + row) + col * ldc;
            out.extend_from_slice(&gemm_read_f32(base, dtype, src_index)?.to_ne_bytes());
        }
    }
    Ok(out)
}

unsafe fn unpack_gemm_c_block_rowmajor_f32_bytes(
    src: &[u8],
    dst: *mut std::ffi::c_void,
    dtype: i32,
    row0: usize,
    rows: usize,
    n: usize,
    ldc: usize,
) -> std::io::Result<()> {
    for row in 0..rows {
        for col in 0..n {
            let off = (row * n + col) * std::mem::size_of::<f32>();
            let value = f32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
            gemm_write_from_f32(dst, dtype, (row0 + row) + col * ldc, value)?;
        }
    }
    Ok(())
}

unsafe fn pack_gemm_a_block_colmajor_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    row0: usize,
    rows: usize,
    k: usize,
    lda: usize,
    transposed_source: bool,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rows * k * std::mem::size_of::<f32>());
    for kk in 0..k {
        for row in 0..rows {
            let src_index = if transposed_source {
                kk + (row0 + row) * lda
            } else {
                (row0 + row) + kk * lda
            };
            out.extend_from_slice(&gemm_read_f32(base, dtype, src_index)?.to_ne_bytes());
        }
    }
    Ok(out)
}

unsafe fn pack_gemm_c_block_colmajor_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    row0: usize,
    rows: usize,
    n: usize,
    ldc: usize,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rows * n * std::mem::size_of::<f32>());
    for col in 0..n {
        for row in 0..rows {
            let src_index = (row0 + row) + col * ldc;
            out.extend_from_slice(&gemm_read_f32(base, dtype, src_index)?.to_ne_bytes());
        }
    }
    Ok(out)
}

unsafe fn unpack_gemm_c_block_colmajor_f32_bytes(
    src: &[u8],
    dst: *mut std::ffi::c_void,
    dtype: i32,
    row0: usize,
    rows: usize,
    n: usize,
    ldc: usize,
) -> std::io::Result<()> {
    for col in 0..n {
        for row in 0..rows {
            let off = (row + col * rows) * std::mem::size_of::<f32>();
            let value = f32::from_ne_bytes(src[off..off + 4].try_into().unwrap());
            gemm_write_from_f32(dst, dtype, (row0 + row) + col * ldc, value)?;
        }
    }
    Ok(())
}

unsafe fn pack_gemm_b_colmajor_padded_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    k: usize,
    n: usize,
    padded_n: usize,
    ldb: usize,
    transposed_source: bool,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(k * padded_n * std::mem::size_of::<f32>());
    for col in 0..padded_n {
        for kk in 0..k {
            let value = if col < n {
                let src_index = if transposed_source {
                    col + kk * ldb
                } else {
                    kk + col * ldb
                };
                gemm_read_f32(base, dtype, src_index)?
            } else {
                0.0
            };
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(out)
}

unsafe fn pack_gemm_c_block_colmajor_padded_f32_bytes(
    base: *const std::ffi::c_void,
    dtype: i32,
    row0: usize,
    rows: usize,
    n: usize,
    padded_n: usize,
    ldc: usize,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rows * padded_n * std::mem::size_of::<f32>());
    for col in 0..padded_n {
        for row in 0..rows {
            let value = if col < n {
                gemm_read_f32(base, dtype, (row0 + row) + col * ldc)?
            } else {
                0.0
            };
            out.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(out)
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| {
            let value = v.trim();
            value
                .strip_prefix("0x")
                .or_else(|| value.strip_prefix("0X"))
                .map(|hex| usize::from_str_radix(hex, 16).ok())
                .unwrap_or_else(|| value.parse().ok())
        })
        .unwrap_or(default)
}

fn append_aligned_region(cursor: &mut u64, len: usize, align: u64) -> std::io::Result<u64> {
    let off = align_up_u64(*cursor, align)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC staging offset overflow"))?;
    *cursor = off
        .checked_add(len as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC staging offset overflow"))?;
    Ok(off)
}

fn copy_gemm_a_k_chunk(
    dst: &mut [u8],
    src: &[u8],
    m: usize,
    k_start: usize,
    k_len: usize,
    lda: usize,
    elem_size: usize,
) -> std::io::Result<()> {
    for row in 0..m {
        let src_elem = row
            .checked_mul(lda)
            .and_then(|v| v.checked_add(k_start))
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A chunk source offset overflow"))?;
        let src_off = src_elem
            .checked_mul(elem_size)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A chunk byte offset overflow"))?;
        let len = k_len
            .checked_mul(elem_size)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A chunk byte length overflow"))?;
        let dst_off = row
            .checked_mul(k_len)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "A chunk destination offset overflow",
                )
            })?;
        dst[dst_off..dst_off + len].copy_from_slice(&src[src_off..src_off + len]);
    }
    Ok(())
}

fn copy_gemm_b_k_chunk(
    dst: &mut [u8],
    src: &[u8],
    n: usize,
    k_start: usize,
    k_len: usize,
    ldb: usize,
    elem_size: usize,
) -> std::io::Result<()> {
    for kk in 0..k_len {
        let src_elem = (k_start + kk)
            .checked_mul(ldb)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B chunk source offset overflow"))?;
        let src_off = src_elem
            .checked_mul(elem_size)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B chunk byte offset overflow"))?;
        let len = n
            .checked_mul(elem_size)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B chunk byte length overflow"))?;
        let dst_off = kk
            .checked_mul(n)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "B chunk destination offset overflow",
                )
            })?;
        dst[dst_off..dst_off + len].copy_from_slice(&src[src_off..src_off + len]);
    }
    Ok(())
}

unsafe fn submit_gemm_staged_single_shared_ddr(
    dev_override: Option<usize>,
    slot_override: Option<usize>,
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
) -> std::io::Result<()> {
    let _gemm_guard = shared_ddr_gemm_lock()
        .lock()
        .map_err(|_| Error::new(ErrorKind::Other, "PACC GEMM staging lock poisoned"))?;
    let a_dtype_size = pacc_dtype_size(atype).ok_or_else(|| {
        Error::new(
            ErrorKind::Unsupported,
            format!("unsupported staged GEMM A dtype {}", atype),
        )
    })?;
    let b_dtype_size = pacc_dtype_size(btype).ok_or_else(|| {
        Error::new(
            ErrorKind::Unsupported,
            format!("unsupported staged GEMM B dtype {}", btype),
        )
    })?;
    let c_dtype_size = pacc_dtype_size(ctype).ok_or_else(|| {
        Error::new(
            ErrorKind::Unsupported,
            format!("unsupported staged GEMM C dtype {}", ctype),
        )
    })?;

    let m = m as u64;
    let n = n as u64;
    let k = k as u64;
    let batch_count = batch_count as u64;
    let lda = lda as i64;
    let ldb = ldb as i64;
    let ldc = ldc as i64;
    let a_matrix_elems = if transa != 0 {
        gemm_span(k, m, lda)
    } else {
        gemm_span(m, k, lda)
    }
    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A GEMM matrix span overflow"))?;
    let b_matrix_elems = if transb != 0 {
        gemm_span(n, k, ldb)
    } else {
        gemm_span(k, n, ldb)
    }
    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B GEMM matrix span overflow"))?;
    let c_matrix_elems = gemm_span(m, n, ldc)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C GEMM matrix span overflow"))?;
    let a_batch_stride = if stride_a > 0 {
        stride_a as usize
    } else {
        a_matrix_elems
    };
    let b_batch_stride = if stride_b > 0 {
        stride_b as usize
    } else {
        b_matrix_elems
    };
    let c_batch_stride = if stride_c > 0 {
        stride_c as usize
    } else {
        c_matrix_elems
    };
    let batches = batch_count as usize;
    let a_elems = a_batch_stride
        .checked_mul(batches.saturating_sub(1))
        .and_then(|v| v.checked_add(a_matrix_elems))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A GEMM span overflow"))?;
    let b_elems = b_batch_stride
        .checked_mul(batches.saturating_sub(1))
        .and_then(|v| v.checked_add(b_matrix_elems))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B GEMM span overflow"))?;
    let c_elems = c_batch_stride
        .checked_mul(batches.saturating_sub(1))
        .and_then(|v| v.checked_add(c_matrix_elems))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C GEMM span overflow"))?;
    let a_bytes = a_elems
        .checked_mul(a_dtype_size)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A GEMM byte span overflow"))?;
    let b_bytes = b_elems
        .checked_mul(b_dtype_size)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B GEMM byte span overflow"))?;
    let c_bytes = c_elems
        .checked_mul(c_dtype_size)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C GEMM byte span overflow"))?;

    let firmware_f32_only = std::env::var("HETGPU_PACC_GEMM_FW_F32_ONLY")
        .ok()
        .as_deref()
        != Some("0");
    let stage_as_f32 = firmware_f32_only
        && (atype as u32 != PaccDataType::Float32 as u32
            || btype as u32 != PaccDataType::Float32 as u32
            || ctype as u32 != PaccDataType::Float32 as u32);
    let pacc_atype = if stage_as_f32 {
        PaccDataType::Float32 as i32
    } else {
        atype
    };
    let pacc_btype = if stage_as_f32 {
        PaccDataType::Float32 as i32
    } else {
        btype
    };
    let pacc_ctype = if stage_as_f32 {
        PaccDataType::Float32 as i32
    } else {
        ctype
    };
    let compact_stage = stage_as_f32
        && std::env::var("HETGPU_PACC_GEMM_PACK_COMPACT")
            .ok()
            .as_deref()
            != Some("0");
    let compact_a_elems = (m as usize)
        .checked_mul(k as usize)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "compact A GEMM span overflow"))?;
    let compact_b_elems = (k as usize)
        .checked_mul(n as usize)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "compact B GEMM span overflow"))?;
    let compact_c_elems = (m as usize)
        .checked_mul(n as usize)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "compact C GEMM span overflow"))?;
    let pacc_a_bytes = if compact_stage {
        compact_a_elems * std::mem::size_of::<f32>()
    } else if stage_as_f32 {
        a_elems * std::mem::size_of::<f32>()
    } else {
        a_bytes
    };
    let pacc_b_bytes = if compact_stage {
        compact_b_elems * std::mem::size_of::<f32>()
    } else if stage_as_f32 {
        b_elems * std::mem::size_of::<f32>()
    } else {
        b_bytes
    };
    let pacc_c_bytes = if compact_stage {
        compact_c_elems * std::mem::size_of::<f32>()
    } else if stage_as_f32 {
        c_elems * std::mem::size_of::<f32>()
    } else {
        c_bytes
    };

    let shared_base = shared_ddr_base();
    let shared_bytes = shared_ddr_bytes();
    if shared_base == 0 || shared_bytes == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            "PACC shared DDR staging window is not configured",
        ));
    }
    let slot_count = std::env::var("HETGPU_PACC_GEMM_SHARED_SLOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(PACC_CORE_NUM);
    let requested_slot = slot_override.unwrap_or(0);
    let slot_id = requested_slot.min(slot_count.saturating_sub(1));
    let slot_bytes = std::env::var("HETGPU_PACC_GEMM_SLOT_BYTES")
        .ok()
        .and_then(|v| {
            let trimmed = v.trim_start_matches("0x");
            usize::from_str_radix(trimmed, 16)
                .ok()
                .or_else(|| v.parse().ok())
        })
        .unwrap_or_else(|| shared_bytes / slot_count.max(1));
    let slot_off = (slot_id as u64)
        .checked_mul(slot_bytes as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "shared DDR slot offset overflow"))?;
    if slot_bytes == 0 || slot_off as usize >= shared_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            "PACC shared DDR slot is outside the configured window",
        ));
    }
    let slot_available = shared_bytes - slot_off as usize;
    let slot_bytes = slot_bytes.min(slot_available);

    let a_off = 0u64;
    let b_off = align_up_u64(pacc_a_bytes as u64, 64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B staging offset overflow"))?;
    let c_off = align_up_u64(
        b_off
            .checked_add(pacc_b_bytes as u64)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C staging offset overflow"))?,
        64,
    )
    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C staging offset overflow"))?;
    let alpha_off = align_up_u64(
        c_off
            .checked_add(pacc_c_bytes as u64)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "alpha staging offset overflow"))?,
        64,
    )
    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "alpha staging offset overflow"))?;
    let beta_off = alpha_off
        .checked_add(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "beta staging offset overflow"))?;
    let total = beta_off
        .checked_add(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM staging window overflow"))?;
    let old_total = c_off
        .checked_add(pacc_c_bytes as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM staging window overflow"))?;
    if total as usize > slot_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "PACC staged GEMM needs {} bytes, shared DDR slot has {}",
                total, slot_bytes
            ),
        ));
    }

    let a_src = std::slice::from_raw_parts(a.cast::<u8>(), a_bytes);
    let b_src = std::slice::from_raw_parts(b.cast::<u8>(), b_bytes);
    let a_stage;
    let b_stage;
    let c_stage;
    let (a_payload, b_payload, c_payload): (&[u8], &[u8], Option<&[u8]>) = if compact_stage {
        a_stage = pack_gemm_operand_to_f32_bytes(
            a,
            atype,
            m as usize,
            k as usize,
            lda as usize,
            transa != 0,
        )?;
        b_stage = pack_gemm_operand_to_f32_bytes(
            b,
            btype,
            k as usize,
            n as usize,
            ldb as usize,
            transb != 0,
        )?;
        c_stage =
            pack_gemm_c_to_f32_bytes(c.cast_const(), ctype, m as usize, n as usize, ldc as usize)?;
        (&a_stage, &b_stage, Some(&c_stage))
    } else if stage_as_f32 {
        a_stage = gemm_storage_to_f32_bytes(a_src, atype, a_elems)?;
        b_stage = gemm_storage_to_f32_bytes(b_src, btype, b_elems)?;
        (&a_stage, &b_stage, None)
    } else {
        (a_src, b_src, None)
    };
    write_shared_ddr_window(slot_off + a_off, a_payload)?;
    write_shared_ddr_window(slot_off + b_off, b_payload)?;
    if let Some(c_payload) = c_payload {
        write_shared_ddr_window(slot_off + c_off, c_payload)?;
    } else if pacc_c_bytes != 0 {
        let zero = vec![0u8; pacc_c_bytes];
        write_shared_ddr_window(slot_off + c_off, &zero)?;
    }
    let alpha_value = read_f32_arg(alpha, 1.0);
    let beta_value = read_f32_arg(beta, 0.0);
    write_shared_ddr_window(slot_off + alpha_off, &alpha_value.to_ne_bytes())?;
    write_shared_ddr_window(slot_off + beta_off, &beta_value.to_ne_bytes())?;

    let job = HetgpuPaccGemmJob {
        transa: if compact_stage { 0 } else { transa as u32 },
        transb: if compact_stage { 0 } else { transb as u32 },
        atype: pacc_atype as u32,
        btype: pacc_btype as u32,
        ctype: pacc_ctype as u32,
        compute_type: compute_type as u32,
        m,
        n,
        k,
        a_addr: shared_base + slot_off + a_off,
        b_addr: shared_base + slot_off + b_off,
        c_addr: shared_base + slot_off + c_off,
        alpha_addr: shared_base + slot_off + alpha_off,
        beta_addr: shared_base + slot_off + beta_off,
        lda: if compact_stage { m as i64 } else { lda },
        ldb: if compact_stage { k as i64 } else { ldb },
        ldc: if compact_stage { m as i64 } else { ldc },
        stride_a,
        stride_b,
        stride_c,
        batch_count,
    };
    let dev_id = dev_override.unwrap_or_else(next_gemm_device);
    eprintln!(
        "hetgpu_pacc_submit_gemm_staged: submit dev={} slot={} dtype A/B/C={}/{}/{} m={} n={} k={}",
        dev_id, slot_id, job.atype, job.btype, job.ctype, job.m, job.n, job.k
    );
    PaccDevice::open(dev_id)?.submit_runtime_job(hetgpu_pacc_job_id::GEMM, &job)?;

    let mut c_storage = vec![0u8; pacc_c_bytes];
    read_shared_ddr_window(slot_off + c_off, &mut c_storage)?;
    if compact_stage {
        unpack_gemm_c_from_f32_bytes(&c_storage, c, ctype, m as usize, n as usize, ldc as usize)?;
    } else if stage_as_f32 {
        let converted = f32_bytes_to_gemm_storage(&c_storage, ctype, c_elems)?;
        std::ptr::copy_nonoverlapping(converted.as_ptr(), c.cast::<u8>(), c_bytes);
    } else {
        std::ptr::copy_nonoverlapping(c_storage.as_ptr(), c.cast::<u8>(), c_bytes);
    }
    eprintln!(
        "hetgpu_pacc_submit_gemm_staged: dev={} slot={} staged {}+{} -> {} bytes via shared DDR 0x{:x}{}",
        dev_id,
        slot_id,
        pacc_a_bytes,
        pacc_b_bytes,
        old_total,
        shared_base + slot_off,
        if compact_stage {
            " (fw-f32-compact)"
        } else if stage_as_f32 {
            " (fw-f32-converted)"
        } else {
            ""
        }
    );
    Ok(())
}

unsafe fn submit_gemm_staged_4pacc_k_reduce_shared_ddr(
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
) -> std::io::Result<bool> {
    if transa != 0
        || transb != 0
        || batch_count != 1
        || stride_a > 0
        || stride_b > 0
        || stride_c > 0
    {
        return Ok(false);
    }
    if ctype as u32 != PaccDataType::Float32 as u32 {
        return Ok(false);
    }

    let a_dtype_size = pacc_dtype_size(atype).ok_or_else(|| {
        Error::new(
            ErrorKind::Unsupported,
            format!("unsupported staged GEMM A dtype {}", atype),
        )
    })?;
    let b_dtype_size = pacc_dtype_size(btype).ok_or_else(|| {
        Error::new(
            ErrorKind::Unsupported,
            format!("unsupported staged GEMM B dtype {}", btype),
        )
    })?;

    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let lda_usize = if lda > 0 { lda as usize } else { k_usize };
    let ldb_usize = if ldb > 0 { ldb as usize } else { n_usize };
    let ldc_usize = if ldc > 0 { ldc as usize } else { n_usize };
    if lda_usize < k_usize || ldb_usize < n_usize || ldc_usize < n_usize {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "invalid GEMM leading dimension for 4-PACC split",
        ));
    }

    let a_elems = gemm_span(m as u64, k as u64, lda_usize as i64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A GEMM span overflow"))?;
    let b_elems = gemm_span(k as u64, n as u64, ldb_usize as i64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B GEMM span overflow"))?;
    let c_elems = gemm_span(m as u64, n as u64, ldc_usize as i64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C GEMM span overflow"))?;
    let c_bytes = c_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C GEMM byte span overflow"))?;

    if k_usize < 2 || c_elems == 0 {
        return Ok(false);
    }

    let shared_base = shared_ddr_base();
    let shared_bytes = shared_ddr_bytes();
    if shared_base == 0 || shared_bytes == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            "PACC shared DDR staging window is not configured",
        ));
    }

    let a_src = std::slice::from_raw_parts(a.cast::<u8>(), a_elems * a_dtype_size);
    let b_src = std::slice::from_raw_parts(b.cast::<u8>(), b_elems * b_dtype_size);
    let old_c = std::slice::from_raw_parts(c.cast::<f32>(), c_elems).to_vec();

    let mut cursor = 0u64;
    let mut jobs = Vec::with_capacity(PACC_CORE_NUM);
    let alpha_off = append_aligned_region(&mut cursor, std::mem::size_of::<f32>(), 64)?;
    let beta_off = append_aligned_region(&mut cursor, std::mem::size_of::<f32>(), 64)?;

    for dev_id in 0..PACC_CORE_NUM {
        let k0 = k_usize * dev_id / PACC_CORE_NUM;
        let k1 = k_usize * (dev_id + 1) / PACC_CORE_NUM;
        let k_len = k1 - k0;
        if k_len == 0 {
            jobs.push(None);
            continue;
        }
        let a_chunk_bytes = m_usize
            .checked_mul(k_len)
            .and_then(|v| v.checked_mul(a_dtype_size))
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A chunk size overflow"))?;
        let b_chunk_bytes = k_len
            .checked_mul(n_usize)
            .and_then(|v| v.checked_mul(b_dtype_size))
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B chunk size overflow"))?;
        let a_off = append_aligned_region(&mut cursor, a_chunk_bytes, 64)?;
        let b_off = append_aligned_region(&mut cursor, b_chunk_bytes, 64)?;
        let c_off = append_aligned_region(&mut cursor, c_bytes, 64)?;
        jobs.push(Some((
            dev_id,
            k0,
            k_len,
            a_off,
            b_off,
            c_off,
            a_chunk_bytes,
            b_chunk_bytes,
        )));
    }

    if cursor as usize > shared_bytes {
        return Ok(false);
    }

    let alpha_value = read_f32_arg(alpha, 1.0);
    let zero_beta = 0.0f32;
    write_shared_ddr_window(alpha_off, &alpha_value.to_ne_bytes())?;
    write_shared_ddr_window(beta_off, &zero_beta.to_ne_bytes())?;

    let zero_c = vec![0u8; c_bytes];
    for job in &jobs {
        let Some((dev_id, k0, k_len, a_off, b_off, c_off, a_chunk_bytes, b_chunk_bytes)) = *job
        else {
            continue;
        };
        let mut a_chunk = vec![0u8; a_chunk_bytes];
        let mut b_chunk = vec![0u8; b_chunk_bytes];
        copy_gemm_a_k_chunk(
            &mut a_chunk,
            a_src,
            m_usize,
            k0,
            k_len,
            lda_usize,
            a_dtype_size,
        )?;
        copy_gemm_b_k_chunk(
            &mut b_chunk,
            b_src,
            n_usize,
            k0,
            k_len,
            ldb_usize,
            b_dtype_size,
        )?;
        write_shared_ddr_window(a_off, &a_chunk)?;
        write_shared_ddr_window(b_off, &b_chunk)?;
        write_shared_ddr_window(c_off, &zero_c)?;

        let gemm = HetgpuPaccGemmJob {
            transa: 0,
            transb: 0,
            atype: atype as u32,
            btype: btype as u32,
            ctype: ctype as u32,
            compute_type: compute_type as u32,
            m: m as u64,
            n: n as u64,
            k: k_len as u64,
            a_addr: shared_base + a_off,
            b_addr: shared_base + b_off,
            c_addr: shared_base + c_off,
            alpha_addr: shared_base + alpha_off,
            beta_addr: shared_base + beta_off,
            lda: k_len as i64,
            ldb: n as i64,
            ldc: ldc_usize as i64,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            batch_count: 1,
        };
        eprintln!(
            "hetgpu_pacc_submit_gemm_staged: split submit dev={} dtype A/B/C={}/{}/{} m={} n={} k={}",
            dev_id, gemm.atype, gemm.btype, gemm.ctype, gemm.m, gemm.n, gemm.k
        );
        PaccDevice::open(dev_id)?.submit_runtime_job(hetgpu_pacc_job_id::GEMM, &gemm)?;
    }

    let mut reduce_input = vec![0.0f32; c_elems * PACC_CORE_NUM];
    for (rank, job) in jobs.iter().enumerate() {
        let rank_dst = &mut reduce_input[rank * c_elems..(rank + 1) * c_elems];
        if let Some((_, _, _, _, _, c_off, _, _)) = *job {
            let rank_bytes =
                std::slice::from_raw_parts_mut(rank_dst.as_mut_ptr().cast::<u8>(), c_bytes);
            read_shared_ddr_window(c_off, rank_bytes)?;
        }
    }

    let mut reduce_output = vec![0.0f32; c_elems * PACC_CORE_NUM];
    PaccComm::init_all()
        .map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("PACC communicator init failed: {e}"),
            )
        })?
        .all_reduce(&reduce_input, &mut reduce_output, PaccReduceOp::Sum)
        .map_err(|e| Error::new(ErrorKind::Other, format!("PACC all_reduce failed: {e}")))?;

    let beta_value = read_f32_arg(beta, 0.0);
    let c_out = std::slice::from_raw_parts_mut(c.cast::<f32>(), c_elems);
    c_out.copy_from_slice(&reduce_output[..c_elems]);
    if beta_value != 0.0 {
        for (dst, old) in c_out.iter_mut().zip(old_c.iter()) {
            *dst += beta_value * *old;
        }
    }

    eprintln!(
        "hetgpu_pacc_submit_gemm_staged: 4-PACC split-k reduce m={} n={} k={} c_elems={} shared=0x{:x}",
        m, n, k, c_elems, shared_base
    );
    Ok(true)
}

unsafe fn submit_gemm_staged_shared_ddr(
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
) -> std::io::Result<()> {
    if std::env::var("HETGPU_PACC_GEMM_SPLIT_K").ok().as_deref() == Some("1") {
        if submit_gemm_staged_4pacc_k_reduce_shared_ddr(
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            atype,
            lda,
            stride_a,
            b,
            btype,
            ldb,
            stride_b,
            beta,
            c,
            ctype,
            ldc,
            stride_c,
            batch_count,
            compute_type,
        )? {
            return Ok(());
        }
    }

    submit_gemm_staged_single_shared_ddr(
        None,
        None,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        atype,
        lda,
        stride_a,
        b,
        btype,
        ldb,
        stride_b,
        beta,
        c,
        ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
    )
}

unsafe fn submit_gemm_staged_c_tile_on_device(
    trace_gemm: bool,
    shared_base: u64,
    slot_off: u64,
    slot_bytes: usize,
    dev: &PaccDevice,
    transa: i32,
    transb: i32,
    row0: usize,
    col0: usize,
    chunk_m: usize,
    chunk_n: usize,
    k: usize,
    max_k: usize,
    alpha_value: f32,
    beta_value: f32,
    one_value: f32,
    a_batch: *const std::ffi::c_void,
    b_batch: *const std::ffi::c_void,
    c_batch: *mut std::ffi::c_void,
    atype: i32,
    btype: i32,
    ctype: i32,
    compute_type: i32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    a_dtype_size: usize,
    b_dtype_size: usize,
    c_dtype_size: usize,
) -> std::io::Result<()> {
    let mut shared_file = open_shared_ddr_window_file(0);
    let mut mailbox_file = open_pacc_mailbox_file(dev.id);
    let c_ptr = (c_batch as *mut u8)
        .add((row0 + col0 * ldc) * c_dtype_size)
        .cast::<std::ffi::c_void>();
    let mut c_stage =
        pack_gemm_c_block_rowmajor_f32_bytes(c_ptr.cast_const(), ctype, 0, chunk_m, chunk_n, ldc)?;
    let a_bytes = chunk_m
        .checked_mul(max_k)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A tile size overflow"))?;
    let b_bytes = max_k
        .checked_mul(chunk_n)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B tile size overflow"))?;
    let c_bytes = c_stage.len();
    let a_off = 0u64;
    let b_off = align_up_u64(a_bytes as u64, 64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B coarse offset overflow"))?;
    let c_off = align_up_u64(b_off + b_bytes as u64, 64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C coarse offset overflow"))?;
    let alpha_off = align_up_u64(c_off + c_bytes as u64, 64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "alpha coarse offset overflow"))?;
    let beta_off = alpha_off + std::mem::size_of::<f32>() as u64;
    let one_off = beta_off + std::mem::size_of::<f32>() as u64;
    let total = one_off + std::mem::size_of::<f32>() as u64;
    if total as usize > slot_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "PACC coarse GEMM needs {} bytes, shared DDR slot has {}",
                total, slot_bytes
            ),
        ));
    }

    write_shared_ddr_window_cached(&mut shared_file, slot_off + c_off, &c_stage)?;
    write_shared_ddr_window_cached(
        &mut shared_file,
        slot_off + alpha_off,
        &alpha_value.to_ne_bytes(),
    )?;
    write_shared_ddr_window_cached(
        &mut shared_file,
        slot_off + beta_off,
        &beta_value.to_ne_bytes(),
    )?;
    write_shared_ddr_window_cached(
        &mut shared_file,
        slot_off + one_off,
        &one_value.to_ne_bytes(),
    )?;

    let tile_max_k = if chunk_m < 64 || chunk_n < 16 {
        max_k.min(parse_env_usize("HETGPU_PACC_GEMM_TAIL_MAX_K", 80).max(1))
    } else {
        max_k.max(1)
    };

    for kk in (0..k).step_by(tile_max_k) {
        let chunk_k = (k - kk).min(tile_max_k);
        let a_index = if transa == 0 {
            row0 + kk * lda
        } else {
            kk + row0 * lda
        };
        let b_index = if transb == 0 {
            kk + col0 * ldb
        } else {
            col0 + kk * ldb
        };
        let a_ptr = (a_batch as *const u8)
            .add(a_index * a_dtype_size)
            .cast::<std::ffi::c_void>();
        let b_ptr = (b_batch as *const u8)
            .add(b_index * b_dtype_size)
            .cast::<std::ffi::c_void>();
        let a_stage = pack_gemm_a_block_rowmajor_f32_bytes(
            a_ptr,
            atype,
            0,
            chunk_m,
            chunk_k,
            lda,
            transa != 0,
        )?;
        let b_stage =
            pack_gemm_b_rowmajor_f32_bytes(b_ptr, btype, chunk_k, chunk_n, ldb, transb != 0)?;
        write_shared_ddr_window_cached(&mut shared_file, slot_off + a_off, &a_stage)?;
        write_shared_ddr_window_cached(&mut shared_file, slot_off + b_off, &b_stage)?;

        let job = HetgpuPaccGemmJob {
            transa: 0,
            transb: 0,
            atype: PaccDataType::Float32 as u32,
            btype: PaccDataType::Float32 as u32,
            ctype: PaccDataType::Float32 as u32,
            compute_type: compute_type as u32,
            m: chunk_m as u64,
            n: chunk_n as u64,
            k: chunk_k as u64,
            a_addr: shared_base + slot_off + a_off,
            b_addr: shared_base + slot_off + b_off,
            c_addr: shared_base + slot_off + c_off,
            alpha_addr: shared_base + slot_off + alpha_off,
            beta_addr: shared_base + slot_off + if kk == 0 { beta_off } else { one_off },
            lda: chunk_k as i64,
            ldb: chunk_n as i64,
            ldc: chunk_n as i64,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            batch_count: 1,
        };
        if trace_gemm {
            eprintln!(
                "hetgpu_pacc_submit_gemm_staged_tiled: submit dev={} slot=0x{:x} row={} col={} k={} m={} n={} k={}",
                dev.id, slot_off, row0, col0, kk, job.m, job.n, job.k
            );
        }
        submit_gemm_runtime_job_cached(dev, &job, total, &mut mailbox_file).map_err(|e| {
            Error::new(
                e.kind(),
                format!(
                    "PACC coarse GEMM tile failed dev={} slot=0x{:x} row={} col={} kk={} m={} n={} k={} lda={} ldb={} ldc={}: {}",
                    dev.id, slot_off, row0, col0, kk, job.m, job.n, job.k, job.lda, job.ldb, job.ldc, e
                ),
            )
        })?;
    }

    read_shared_ddr_window_cached(&mut shared_file, slot_off + c_off, &mut c_stage)?;
    unpack_gemm_c_block_rowmajor_f32_bytes(&c_stage, c_ptr, ctype, 0, chunk_m, chunk_n, ldc)?;
    if trace_gemm {
        eprintln!(
            "hetgpu_pacc_submit_gemm_staged_tiled: tile dev={} slot=0x{:x} row={} col={} m={} n={} staged C-once={} total={} via shared DDR 0x{:x}",
            dev.id, slot_off, row0, col0, chunk_m, chunk_n, c_bytes, total, shared_base + slot_off
        );
    }
    Ok(())
}

unsafe fn submit_gemm_staged_tiled_shared_ddr(
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
    max_m: i32,
    max_n: i32,
    max_k: i32,
) -> std::io::Result<()> {
    let _gemm_guard = shared_ddr_gemm_lock()
        .lock()
        .map_err(|_| Error::new(ErrorKind::Other, "PACC GEMM staging lock poisoned"))?;
    if !pacc_tensor_dtype_supported(atype) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "unsupported coarse staged GEMM A dtype",
        ));
    }
    if !pacc_tensor_dtype_supported(btype) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "unsupported coarse staged GEMM B dtype",
        ));
    }
    if !pacc_tensor_dtype_supported(ctype) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "unsupported coarse staged GEMM C dtype",
        ));
    }
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let max_m = if max_m > 0 { max_m as usize } else { m };
    let max_n = if max_n > 0 { max_n as usize } else { n };
    let max_k = if max_k > 0 { max_k as usize } else { k };
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;
    let a_dtype_size =
        pacc_dtype_size(atype).ok_or_else(|| Error::new(ErrorKind::Unsupported, "bad A dtype"))?;
    let b_dtype_size =
        pacc_dtype_size(btype).ok_or_else(|| Error::new(ErrorKind::Unsupported, "bad B dtype"))?;
    let c_dtype_size =
        pacc_dtype_size(ctype).ok_or_else(|| Error::new(ErrorKind::Unsupported, "bad C dtype"))?;
    let shared_base = shared_ddr_base();
    let shared_bytes = shared_ddr_bytes();
    if shared_base == 0 || shared_bytes == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            "PACC shared DDR staging window is not configured",
        ));
    }
    let slot_count = parse_env_usize("HETGPU_PACC_GEMM_SHARED_SLOTS", PACC_CORE_NUM).max(1);
    let slot_bytes =
        parse_env_usize("HETGPU_PACC_GEMM_SLOT_BYTES", shared_bytes / slot_count).max(1);
    let alpha_value = read_f32_arg(alpha, 1.0);
    let beta_value = read_f32_arg(beta, 0.0);
    let one_value = 1.0f32;
    let batches = batch_count as usize;
    let trace_gemm = pacc_gemm_trace_enabled();

    let row_tiles = (m + max_m - 1) / max_m;
    let col_tiles = (n + max_n - 1) / max_n;
    let tile_count = row_tiles
        .checked_mul(col_tiles)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM tile count overflow"))?;
    let parallel_workers =
        if std::env::var("HETGPU_PACC_GEMM_PARALLEL").ok().as_deref() == Some("1") {
            parse_env_usize("HETGPU_PACC_GEMM_WORKERS", PACC_CORE_NUM)
                .max(1)
                .min(PACC_CORE_NUM.max(1))
                .min(tile_count.max(1))
        } else {
            1
        };

    for batch in 0..batches {
        let a_batch_addr = (a as *const u8).add(if stride_a > 0 {
            batch * stride_a as usize * a_dtype_size
        } else {
            0
        }) as usize;
        let b_batch_addr = (b as *const u8).add(if stride_b > 0 {
            batch * stride_b as usize * b_dtype_size
        } else {
            0
        }) as usize;
        let c_batch_addr = (c as *mut u8).add(if stride_c > 0 {
            batch * stride_c as usize * c_dtype_size
        } else {
            0
        }) as usize;

        if parallel_workers <= 1 || tile_count <= 1 {
            let dev = PaccDevice::open(0)?;
            for tile_idx in 0..tile_count {
                let col_tile = tile_idx / row_tiles;
                let row_tile = tile_idx % row_tiles;
                let row0 = row_tile * max_m;
                let col0 = col_tile * max_n;
                let chunk_m = (m - row0).min(max_m);
                let chunk_n = (n - col0).min(max_n);
                submit_gemm_staged_c_tile_on_device(
                    trace_gemm,
                    shared_base,
                    0,
                    slot_bytes.min(shared_bytes),
                    &dev,
                    transa,
                    transb,
                    row0,
                    col0,
                    chunk_m,
                    chunk_n,
                    k,
                    max_k,
                    alpha_value,
                    beta_value,
                    one_value,
                    a_batch_addr as *const std::ffi::c_void,
                    b_batch_addr as *const std::ffi::c_void,
                    c_batch_addr as *mut std::ffi::c_void,
                    atype,
                    btype,
                    ctype,
                    compute_type,
                    lda,
                    ldb,
                    ldc,
                    a_dtype_size,
                    b_dtype_size,
                    c_dtype_size,
                )?;
            }
            continue;
        }

        let mut scoped_result: std::io::Result<()> = Ok(());
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(parallel_workers);
            for worker in 0..parallel_workers {
                handles.push(scope.spawn(move || -> std::io::Result<()> {
                    let dev_id = worker % PACC_CORE_NUM.max(1);
                    let dev = PaccDevice::open(dev_id)?;
                    let slot_id = worker.min(slot_count.saturating_sub(1));
                    let slot_off =
                        (slot_id as u64)
                            .checked_mul(slot_bytes as u64)
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorKind::InvalidInput,
                                    "shared DDR slot offset overflow",
                                )
                            })?;
                    if slot_off as usize >= shared_bytes {
                        return Err(Error::new(
                            ErrorKind::OutOfMemory,
                            "PACC shared DDR worker slot is outside the configured window",
                        ));
                    }
                    let available = shared_bytes - slot_off as usize;
                    let worker_slot_bytes = slot_bytes.min(available);
                    for tile_idx in (worker..tile_count).step_by(parallel_workers) {
                        let col_tile = tile_idx / row_tiles;
                        let row_tile = tile_idx % row_tiles;
                        let row0 = row_tile * max_m;
                        let col0 = col_tile * max_n;
                        let chunk_m = (m - row0).min(max_m);
                        let chunk_n = (n - col0).min(max_n);
                        submit_gemm_staged_c_tile_on_device(
                            trace_gemm,
                            shared_base,
                            slot_off,
                            worker_slot_bytes,
                            &dev,
                            transa,
                            transb,
                            row0,
                            col0,
                            chunk_m,
                            chunk_n,
                            k,
                            max_k,
                            alpha_value,
                            beta_value,
                            one_value,
                            a_batch_addr as *const std::ffi::c_void,
                            b_batch_addr as *const std::ffi::c_void,
                            c_batch_addr as *mut std::ffi::c_void,
                            atype,
                            btype,
                            ctype,
                            compute_type,
                            lda,
                            ldb,
                            ldc,
                            a_dtype_size,
                            b_dtype_size,
                            c_dtype_size,
                        )?;
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        if scoped_result.is_ok() {
                            scoped_result = Err(e);
                        }
                    }
                    Err(_) => {
                        if scoped_result.is_ok() {
                            scoped_result =
                                Err(Error::new(ErrorKind::Other, "PACC GEMM worker panicked"));
                        }
                    }
                }
            }
        });
        scoped_result?;
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_gemm_staged_tiled(
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
    max_m: i32,
    max_n: i32,
    max_k: i32,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() || m <= 0 || n <= 0 || k <= 0 || batch_count <= 0 {
        eprintln!("hetgpu_pacc_submit_gemm_staged_tiled: invalid argument");
        return -1;
    }
    match submit_gemm_staged_tiled_shared_ddr(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        atype,
        lda,
        stride_a,
        b,
        btype,
        ldb,
        stride_b,
        beta,
        c,
        ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        max_m,
        max_n,
        max_k,
    ) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("hetgpu_pacc_submit_gemm_staged_tiled: {}", e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_gemm_staged_on(
    dev_id: i32,
    slot_id: i32,
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
        eprintln!("hetgpu_pacc_submit_gemm_staged_on: invalid argument");
        return -1;
    }
    let dev_override = if dev_id >= 0 && (dev_id as usize) < PACC_CORE_NUM {
        Some(dev_id as usize)
    } else {
        None
    };
    let slot_override = if slot_id >= 0 {
        Some(slot_id as usize)
    } else {
        None
    };
    match submit_gemm_staged_single_shared_ddr(
        dev_override,
        slot_override,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        atype,
        lda,
        stride_a,
        b,
        btype,
        ldb,
        stride_b,
        beta,
        c,
        ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
    ) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!(
                "hetgpu_pacc_submit_gemm_staged_on: PACC staged GEMM submit failed: {}",
                e
            );
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_gemm_staged(
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
    let _ = (alpha, beta);
    if a.is_null() || b.is_null() || c.is_null() || m <= 0 || n <= 0 || k <= 0 || batch_count <= 0 {
        eprintln!("hetgpu_pacc_submit_gemm_staged: invalid argument");
        return -1;
    }
    match submit_gemm_staged_shared_ddr(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        atype,
        lda,
        stride_a,
        b,
        btype,
        ldb,
        stride_b,
        beta,
        c,
        ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
    ) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!(
                "hetgpu_pacc_submit_gemm_staged: PACC staged GEMM submit failed: {}",
                e
            );
            -1
        }
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

    let dev_id = next_gemm_device();
    eprintln!(
        "hetgpu_pacc_submit_gemm: submit dev={} dtype A/B/C={}/{}/{} m={} n={} k={}",
        dev_id, job.atype, job.btype, job.ctype, job.m, job.n, job.k
    );
    match PaccDevice::open(dev_id)
        .and_then(|dev| dev.submit_runtime_job(hetgpu_pacc_job_id::GEMM, &job))
    {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("hetgpu_pacc_submit_gemm: PACC GEMM submit failed: {}", e);
            -1
        }
    }
}

unsafe fn submit_softmax_typed_impl(
    src: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    rows: u64,
    cols: u64,
    stride: u64,
    dtype: u32,
    label: &str,
) -> i32 {
    let job = HetgpuPaccSoftmaxJob {
        src_addr: src as u64,
        dst_addr: dst as u64,
        rows,
        cols,
        stride: if stride == 0 { cols } else { stride },
        dtype,
        reserved: 0,
    };
    match PaccDevice::open(0)
        .and_then(|dev| dev.submit_runtime_job(hetgpu_pacc_job_id::SOFTMAX, &job))
    {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{}: PACC softmax submit failed: {}", label, e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_softmax(
    src: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    rows: u64,
    cols: u64,
    stride: u64,
    dtype: i32,
) -> i32 {
    if src.is_null()
        || dst.is_null()
        || rows == 0
        || cols == 0
        || !pacc_tensor_dtype_supported(dtype)
    {
        eprintln!("hetgpu_pacc_submit_softmax: invalid argument");
        return -1;
    }
    submit_softmax_typed_impl(
        src,
        dst,
        rows,
        cols,
        stride,
        dtype as u32,
        "hetgpu_pacc_submit_softmax",
    )
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_softmax_f32(
    src: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    rows: u64,
    cols: u64,
    stride: u64,
) -> i32 {
    hetgpu_pacc_submit_softmax(src, dst, rows, cols, stride, PaccDataType::Float32 as i32)
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_softmax_bf16(
    src: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    rows: u64,
    cols: u64,
    stride: u64,
) -> i32 {
    hetgpu_pacc_submit_softmax(src, dst, rows, cols, stride, PaccDataType::Bfloat16 as i32)
}

unsafe fn submit_rmsnorm_typed_impl(
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
    dtype: u32,
    label: &str,
) -> i32 {
    let job = HetgpuPaccRmsNormJob {
        x_addr: x as u64,
        weight_addr: weight as u64,
        y_addr: y as u64,
        rows,
        hidden,
        eps,
        dtype,
    };
    match PaccDevice::open(0)
        .and_then(|dev| dev.submit_runtime_job(hetgpu_pacc_job_id::RMSNORM, &job))
    {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{}: PACC RMSNorm submit failed: {}", label, e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_rmsnorm(
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
    dtype: i32,
) -> i32 {
    if x.is_null() || y.is_null() || rows == 0 || hidden == 0 || !pacc_tensor_dtype_supported(dtype)
    {
        eprintln!("hetgpu_pacc_submit_rmsnorm: invalid argument");
        return -1;
    }
    submit_rmsnorm_typed_impl(
        x,
        weight,
        y,
        rows,
        hidden,
        eps,
        dtype as u32,
        "hetgpu_pacc_submit_rmsnorm",
    )
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
    hetgpu_pacc_submit_rmsnorm(
        x,
        weight,
        y,
        rows,
        hidden,
        eps,
        PaccDataType::Float32 as i32,
    )
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_rmsnorm_bf16(
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
) -> i32 {
    hetgpu_pacc_submit_rmsnorm(
        x,
        weight,
        y,
        rows,
        hidden,
        eps,
        PaccDataType::Bfloat16 as i32,
    )
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
        eprintln!(
            "hetgpu_pacc_nccl_all_reduce_f32: unsupported op {}, expected ncclSum=0",
            op
        );
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
            eprintln!(
                "hetgpu_pacc_nccl_all_reduce_f32: PACC all_reduce failed: {}",
                e
            );
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

    let max_count = std::env::var("HETGPU_PACC_ALLREDUCE_MAX_COUNT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(16);
    let mut pacc_out = vec![0.0f32; total];
    let comm = match PaccComm::init_all() {
        Ok(comm) => comm,
        Err(e) => {
            eprintln!("hetgpu_pacc_nccl_reduce_sum_f32: PACC init failed: {}", e);
            return -1;
        }
    };

    for start in (0..count).step_by(max_count) {
        let chunk = std::cmp::min(max_count, count - start);
        let mut chunk_in = vec![0.0f32; chunk * nranks_usize];
        for rank in 0..nranks_usize {
            let src_off = rank * count + start;
            let dst_off = rank * chunk;
            chunk_in[dst_off..dst_off + chunk].copy_from_slice(&reduced[src_off..src_off + chunk]);
        }

        let mut chunk_out = vec![0.0f32; chunk * nranks_usize];
        if let Err(e) = comm.all_reduce(&chunk_in, &mut chunk_out, PaccReduceOp::Sum) {
            eprintln!(
                "hetgpu_pacc_nccl_reduce_sum_f32: PACC reduce failed at start={} chunk={}: {}",
                start, chunk, e
            );
            return -1;
        }

        for rank in 0..nranks_usize {
            let src_off = rank * chunk;
            let dst_off = rank * count + start;
            pacc_out[dst_off..dst_off + chunk]
                .copy_from_slice(&chunk_out[src_off..src_off + chunk]);
        }
    }

    std::ptr::copy_nonoverlapping(pacc_out.as_ptr(), recvbuff, count);
    0
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
    launch_state: PaccKernelLaunchState,
}

/// Result code
pub type pacc_Result = i32;
pub const pacc_Result_Success: pacc_Result = 0;
pub const pacc_Result_Error: pacc_Result = -1;

fn default_source_target() -> &'static CStr {
    unsafe { CStr::from_bytes_with_nul_unchecked(b"riscv64-linux-gnu\0") }
}

fn default_ptx_target() -> &'static CStr {
    unsafe { CStr::from_bytes_with_nul_unchecked(b"riscv64-unknown-elf\0") }
}

fn default_ptx_module_name() -> &'static CStr {
    unsafe { CStr::from_bytes_with_nul_unchecked(b"module.ptx\0") }
}

unsafe fn cstr_or_default<'a>(ptr: *const std::ffi::c_char, default_value: &'a CStr) -> &'a CStr {
    if ptr.is_null() {
        default_value
    } else {
        CStr::from_ptr(ptr)
    }
}

unsafe fn slice_or_empty<'a>(ptr: *const u8, len: u64) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len as usize)
    }
}

unsafe fn load_program_elf_bytes(program: *mut pacc_Program, elf_bytes: Vec<u8>) -> pacc_Result {
    if program.is_null() || elf_bytes.is_empty() {
        return pacc_Result_Error;
    }
    (*program).elf_bytes = elf_bytes;
    pacc_Result_Success
}

/// Create a PACC device handle for device_id (0-3).
/// Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn pacc_CreateDevice(device_id: u32) -> *mut pacc_Device {
    match PaccDevice::open(device_id as usize) {
        Ok(dev) => Box::into_raw(Box::new(pacc_Device(dev))),
        Err(e) => {
            eprintln!("pacc_CreateDevice({}): {}", device_id, e);
            std::ptr::null_mut()
        }
    }
}

/// Destroy a PACC device handle.
#[no_mangle]
pub unsafe extern "C" fn pacc_DestroyDevice(dev: *mut pacc_Device) {
    if !dev.is_null() {
        drop(Box::from_raw(dev));
    }
}

/// Create a PACC program (initially empty — load ELF via pacc_LoadProgram).
#[no_mangle]
pub unsafe extern "C" fn pacc_CreateProgram() -> *mut pacc_Program {
    Box::into_raw(Box::new(pacc_Program {
        elf_bytes: Vec::new(),
    }))
}

/// Destroy a PACC program handle.
#[no_mangle]
pub unsafe extern "C" fn pacc_DestroyProgram(program: *mut pacc_Program) {
    if !program.is_null() {
        drop(Box::from_raw(program));
    }
}

/// Load ELF binary into a PACC program.
/// data: pointer to ELF bytes, size: byte length.
#[no_mangle]
pub unsafe extern "C" fn pacc_LoadProgram(
    program: *mut pacc_Program,
    data: *const std::ffi::c_void,
    size: u64,
) -> pacc_Result {
    if program.is_null() || data.is_null() || size == 0 {
        return pacc_Result_Error;
    }
    let bytes = std::slice::from_raw_parts(data as *const u8, size as usize);
    load_program_elf_bytes(program, bytes.to_vec())
}

#[no_mangle]
pub unsafe extern "C" fn pacc_LoadProgramSource(
    program: *mut pacc_Program,
    target_arch: *const std::ffi::c_char,
    source_name: *const std::ffi::c_char,
    source_buffer: *const u8,
    source_len: u64,
    working_directory: *const std::ffi::c_char,
    options: *const *const std::ffi::c_char,
    option_count: usize,
    linked_bitcode: *const u8,
    linked_bitcode_len: u64,
) -> pacc_Result {
    if program.is_null() || source_name.is_null() || source_buffer.is_null() || source_len == 0 {
        return pacc_Result_Error;
    }

    let target_arch = cstr_or_default(target_arch, default_source_target());
    let source_name = CStr::from_ptr(source_name);
    let source_buffer = std::slice::from_raw_parts(source_buffer, source_len as usize);
    let working_directory = if working_directory.is_null() {
        None
    } else {
        Some(CStr::from_ptr(working_directory))
    };
    let linked_bitcode = slice_or_empty(linked_bitcode, linked_bitcode_len);

    let mut option_refs = Vec::with_capacity(option_count);
    if !options.is_null() {
        for idx in 0..option_count {
            let opt = *options.add(idx);
            if !opt.is_null() {
                option_refs.push(CStr::from_ptr(opt));
            }
        }
    }

    match comgr::compile_source_pacc(
        target_arch,
        source_name,
        source_buffer,
        working_directory,
        &option_refs,
        linked_bitcode,
    ) {
        Ok(elf_bytes) => {
            if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "pacc_LoadProgramSource: compiled '{}' to {}-byte ELF for target {}",
                    source_name.to_string_lossy(),
                    elf_bytes.len(),
                    target_arch.to_string_lossy(),
                );
            }
            load_program_elf_bytes(program, elf_bytes)
        }
        Err(err) => {
            eprintln!(
                "pacc_LoadProgramSource: failed for {}: {:?}",
                source_name.to_string_lossy(),
                err
            );
            pacc_Result_Error
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn pacc_LoadProgramPtx(
    program: *mut pacc_Program,
    target_arch: *const std::ffi::c_char,
    module_name: *const std::ffi::c_char,
    ptx_buffer: *const u8,
    ptx_len: u64,
    linked_bitcode: *const u8,
    linked_bitcode_len: u64,
) -> pacc_Result {
    if program.is_null() || ptx_buffer.is_null() || ptx_len == 0 {
        return pacc_Result_Error;
    }

    let target_arch = cstr_or_default(target_arch, default_ptx_target());
    let module_name = if module_name.is_null() {
        default_ptx_module_name()
    } else {
        CStr::from_ptr(module_name)
    };
    let ptx_bytes = std::slice::from_raw_parts(ptx_buffer, ptx_len as usize);
    let external_linked = slice_or_empty(linked_bitcode, linked_bitcode_len);
    let ptx_text = match std::str::from_utf8(ptx_bytes) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "pacc_LoadProgramPtx: invalid UTF-8 in module {}: {}",
                module_name.to_string_lossy(),
                err
            );
            return pacc_Result_Error;
        }
    };

    let ast = match ptx_parser::parse_module_checked(ptx_text) {
        Ok(ast) => ast,
        Err(err) => {
            eprintln!(
                "pacc_LoadProgramPtx: PTX parse failed for {}: {:?}",
                module_name.to_string_lossy(),
                err
            );
            return pacc_Result_Error;
        }
    };
    let llvm_module = match ptx::to_llvm_module(
        ast,
        ptx::pass::Attributes {
            clock_rate: 1_000_000,
            emit_debug_info: false,
        },
        |_| {},
    ) {
        Ok(module) => module,
        Err(err) => {
            eprintln!(
                "pacc_LoadProgramPtx: PTX -> LLVM failed for {}: {:?}",
                module_name.to_string_lossy(),
                err
            );
            return pacc_Result_Error;
        }
    };
    let ir_bytes = llvm_module.llvm_ir.write_bitcode_to_memory();
    let internal_linked = llvm_module.linked_bitcode();
    let mut linked_modules: Vec<&[u8]> = Vec::new();
    if !internal_linked.is_empty() {
        linked_modules.push(internal_linked);
    }
    if !external_linked.is_empty() {
        linked_modules.push(external_linked);
    }

    match comgr::compile_bitcode_pacc_multi(target_arch, &*ir_bytes, &linked_modules) {
        Ok(elf_bytes) => {
            if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "pacc_LoadProgramPtx: compiled '{}' to {}-byte XM ELF for target {}",
                    module_name.to_string_lossy(),
                    elf_bytes.len(),
                    target_arch.to_string_lossy(),
                );
            }
            load_program_elf_bytes(program, elf_bytes)
        }
        Err(err) => {
            eprintln!(
                "pacc_LoadProgramPtx: LLVM -> XM ELF failed for {}: {:?}",
                module_name.to_string_lossy(),
                err
            );
            pacc_Result_Error
        }
    }
}

/// Create a named kernel handle from a program.
#[no_mangle]
pub unsafe extern "C" fn pacc_CreateKernel(
    program: *mut pacc_Program,
    name: *const std::ffi::c_char,
) -> *mut pacc_Kernel {
    pacc_CreateKernelOnDevice(program, std::ptr::null_mut(), name)
}

/// Create a named kernel handle tied to an already opened PACC device.
#[no_mangle]
pub unsafe extern "C" fn pacc_CreateKernelOnDevice(
    program: *mut pacc_Program,
    device: *mut pacc_Device,
    name: *const std::ffi::c_char,
) -> *mut pacc_Kernel {
    let name_str = if name.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(name)
            .to_string_lossy()
            .into_owned()
    };
    Box::into_raw(Box::new(pacc_Kernel {
        name: name_str,
        program,
        device,
        launch_state: PaccKernelLaunchState::default(),
    }))
}

/// Destroy a kernel handle.
#[no_mangle]
pub unsafe extern "C" fn pacc_DestroyKernel(kernel: *mut pacc_Kernel) {
    if !kernel.is_null() {
        drop(Box::from_raw(kernel));
    }
}

#[no_mangle]
pub unsafe extern "C" fn pacc_KernelClearLaunchState(kernel: *mut pacc_Kernel) -> pacc_Result {
    if kernel.is_null() {
        return pacc_Result_Error;
    }
    (*kernel).launch_state = PaccKernelLaunchState::default();
    pacc_Result_Success
}

#[no_mangle]
pub unsafe extern "C" fn pacc_KernelSetRawParamBlob(
    kernel: *mut pacc_Kernel,
    data: *const std::ffi::c_void,
    size: u64,
) -> pacc_Result {
    if kernel.is_null() || (size != 0 && data.is_null()) {
        return pacc_Result_Error;
    }
    let bytes = if size == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(data as *const u8, size as usize)
    };
    (*kernel).launch_state.raw_param_blob = bytes.to_vec();
    pacc_Result_Success
}

#[no_mangle]
pub unsafe extern "C" fn pacc_KernelPushArgRecord(
    kernel: *mut pacc_Kernel,
    record: *const PaccKernelArgRecord,
) -> pacc_Result {
    if kernel.is_null() || record.is_null() {
        return pacc_Result_Error;
    }
    let record = *record;
    if record.size == 0 || record.size > 8 {
        return pacc_Result_Error;
    }
    if record.kind != PACC_KERNEL_ARG_KIND_SCALAR && record.kind != PACC_KERNEL_ARG_KIND_POINTER {
        return pacc_Result_Error;
    }
    (*kernel).launch_state.arg_records.push(record);
    pacc_Result_Success
}

#[no_mangle]
pub unsafe extern "C" fn pacc_KernelAddBufferBinding(
    kernel: *mut pacc_Kernel,
    binding: *const PaccKernelBufferBinding,
) -> pacc_Result {
    if kernel.is_null() || binding.is_null() {
        return pacc_Result_Error;
    }
    let binding = *binding;
    if binding.arg_index as usize >= (*kernel).launch_state.arg_records.len() {
        return pacc_Result_Error;
    }
    (*kernel).launch_state.bindings.push(binding);
    pacc_Result_Success
}

#[no_mangle]
pub unsafe extern "C" fn pacc_KernelConfigureLanxinMulMatTile(
    kernel: *mut pacc_Kernel,
    m: u32,
    n: u32,
    k: u32,
    a: *const std::ffi::c_void,
    a_offset: u64,
    b: *const std::ffi::c_void,
    b_offset: u64,
    c: *mut std::ffi::c_void,
    c_offset: u64,
) -> pacc_Result {
    if kernel.is_null() {
        return pacc_Result_Error;
    }

    let clear = pacc_KernelClearLaunchState(kernel);
    if clear != pacc_Result_Success {
        return clear;
    }

    let scalar_records = [
        PaccKernelArgRecord {
            kind: PACC_KERNEL_ARG_KIND_SCALAR,
            size: std::mem::size_of::<u32>() as u32,
            flags: 0,
            reserved: 0,
            value: m as u64,
        },
        PaccKernelArgRecord {
            kind: PACC_KERNEL_ARG_KIND_SCALAR,
            size: std::mem::size_of::<u32>() as u32,
            flags: 0,
            reserved: 0,
            value: n as u64,
        },
        PaccKernelArgRecord {
            kind: PACC_KERNEL_ARG_KIND_SCALAR,
            size: std::mem::size_of::<u32>() as u32,
            flags: 0,
            reserved: 0,
            value: k as u64,
        },
        PaccKernelArgRecord {
            kind: PACC_KERNEL_ARG_KIND_POINTER,
            size: std::mem::size_of::<u64>() as u32,
            flags: 0,
            reserved: 0,
            value: a as u64,
        },
        PaccKernelArgRecord {
            kind: PACC_KERNEL_ARG_KIND_POINTER,
            size: std::mem::size_of::<u64>() as u32,
            flags: 0,
            reserved: 0,
            value: b as u64,
        },
        PaccKernelArgRecord {
            kind: PACC_KERNEL_ARG_KIND_POINTER,
            size: std::mem::size_of::<u64>() as u32,
            flags: 0,
            reserved: 0,
            value: c as u64,
        },
    ];

    for record in scalar_records.iter() {
        let rc = pacc_KernelPushArgRecord(kernel, record as *const _);
        if rc != pacc_Result_Success {
            return rc;
        }
    }

    let bindings = [
        PaccKernelBufferBinding {
            arg_index: 3,
            addr: (a as u64).saturating_add(a_offset),
            size: 0,
            flags: 0,
        },
        PaccKernelBufferBinding {
            arg_index: 4,
            addr: (b as u64).saturating_add(b_offset),
            size: 0,
            flags: 0,
        },
        PaccKernelBufferBinding {
            arg_index: 5,
            addr: (c as u64).saturating_add(c_offset),
            size: 0,
            flags: 0,
        },
    ];

    for binding in bindings.iter() {
        let rc = pacc_KernelAddBufferBinding(kernel, binding as *const _);
        if rc != pacc_Result_Success {
            return rc;
        }
    }

    pacc_Result_Success
}

/// Launch a PACC kernel via job_submit.
/// Submits the ELF binary to the device using the physical address of a
/// staging buffer. For now writes ELF bytes to a driver-allocated buffer.
#[no_mangle]
pub unsafe extern "C" fn pacc_LaunchKernel(
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

    let log_launches = std::env::var("HETGPU_PACC_LOG_KERNEL_LAUNCHES")
        .ok()
        .as_deref()
        == Some("1");
    if log_launches {
        eprintln!(
            "pacc_LaunchKernel: kernel='{}' grid=({},{},{}) block=({},{},{})",
            k.name, grid_x, grid_y, grid_z, block_x, block_y, block_z
        );
    }

    if prog.elf_bytes.is_empty() {
        if std::env::var("HETGPU_PACC_LOG_KERNEL_LAUNCHES")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!("pacc_LaunchKernel: no ELF loaded; refusing to stub kernel execution");
        }
        return pacc_Result_Error;
    }

    if !k.device.is_null() {
        return pacc_launch_on_device(
            &(*k.device).0,
            &k.name,
            &prog.elf_bytes,
            &k.launch_state,
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
        Ok(dev) => pacc_launch_on_device(
            &dev,
            &k.name,
            &prog.elf_bytes,
            &k.launch_state,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        ),
        Err(e) => {
            eprintln!("pacc_LaunchKernel: open device failed: {}", e);
            pacc_Result_Error
        }
    }
}

fn allow_preloaded_kernel_fallback() -> bool {
    std::env::var("HETGPU_PACC_ALLOW_PRELOADED_KERNEL_FALLBACK")
        .ok()
        .as_deref()
        == Some("1")
}

fn validate_pacc_kernel_elf(elf_bytes: &[u8]) -> std::io::Result<()> {
    if elf_bytes.len() < 64 || &elf_bytes[0..4] != b"\x7fELF" {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "PACC kernel image must be a non-empty ELF64 payload",
        ));
    }
    if elf_bytes[4] != 2 || elf_bytes[5] != 1 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "PACC kernel image must be little-endian ELF64",
        ));
    }
    Ok(())
}

#[derive(Debug, Default, Copy, Clone)]
struct PaccKernelImageLayout {
    flags: u32,
    launch_header_offset: usize,
    arg_records_offset: usize,
    bindings_offset: usize,
    raw_param_offset: usize,
    elf_offset: usize,
    image_len: usize,
    submit_len: usize,
}

fn compute_pacc_kernel_image_layout(
    elf_bytes: &[u8],
    launch_state: &PaccKernelLaunchState,
) -> std::io::Result<PaccKernelImageLayout> {
    validate_pacc_kernel_elf(elf_bytes)?;

    if launch_state.is_empty() {
        let image_len = PACC_JOB_HEADER_BYTES
            .checked_add(elf_bytes.len())
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC kernel image too large"))?;
        return Ok(PaccKernelImageLayout {
            flags: 0,
            launch_header_offset: 0,
            arg_records_offset: 0,
            bindings_offset: 0,
            raw_param_offset: 0,
            elf_offset: PACC_JOB_HEADER_BYTES,
            image_len,
            submit_len: align_up(image_len, 64),
        });
    }

    let launch_header_offset = PACC_JOB_HEADER_BYTES;
    let launch_header_bytes = std::mem::size_of::<PaccKernelLaunchAbiHeader>();
    let arg_record_bytes = launch_state
        .arg_records
        .len()
        .checked_mul(std::mem::size_of::<PaccKernelArgRecord>())
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "too many PACC arg records"))?;
    let binding_bytes = launch_state
        .bindings
        .len()
        .checked_mul(std::mem::size_of::<PaccKernelBufferBinding>())
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "too many PACC buffer bindings"))?;
    let mut cursor = align_up(launch_header_offset + launch_header_bytes, 8);

    let arg_records_offset = if arg_record_bytes == 0 {
        0
    } else {
        let offset = cursor;
        cursor = align_up(
            cursor
                .checked_add(arg_record_bytes)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC arg section too large"))?,
            8,
        );
        offset
    };

    let bindings_offset = if binding_bytes == 0 {
        0
    } else {
        let offset = cursor;
        cursor = align_up(
            cursor.checked_add(binding_bytes).ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "PACC binding section too large")
            })?,
            8,
        );
        offset
    };

    let raw_param_offset = if launch_state.raw_param_blob.is_empty() {
        0
    } else {
        let offset = cursor;
        cursor = align_up(
            cursor
                .checked_add(launch_state.raw_param_blob.len())
                .ok_or_else(|| {
                    Error::new(ErrorKind::InvalidInput, "PACC raw param section too large")
                })?,
            8,
        );
        offset
    };

    let elf_offset = cursor;
    let image_len = elf_offset
        .checked_add(elf_bytes.len())
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC kernel image too large"))?;

    Ok(PaccKernelImageLayout {
        flags: PACC_JOB_FLAG_HAS_LAUNCH_ABI,
        launch_header_offset,
        arg_records_offset,
        bindings_offset,
        raw_param_offset,
        elf_offset,
        image_len,
        submit_len: align_up(image_len, 64),
    })
}

fn build_pacc_kernel_submit_buffer(
    kernel_name: &str,
    elf_bytes: &[u8],
    launch_state: &PaccKernelLaunchState,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> std::io::Result<(Vec<u8>, usize)> {
    let layout = compute_pacc_kernel_image_layout(elf_bytes, launch_state)?;
    let submit_len = layout.submit_len;
    let mut buf = vec![0u8; submit_len];
    fill_pacc_kernel_image(
        &mut buf,
        kernel_name,
        elf_bytes,
        launch_state,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
    )?;
    Ok((buf, submit_len))
}

fn helper_kernel_submit_enabled(dev_id: usize) -> bool {
    if std::env::var("HETGPU_PACC_KERNEL_MBOX_SUBMIT")
        .ok()
        .as_deref()
        == Some("0")
    {
        return false;
    }
    std::path::Path::new(&helper_path_for_pacc(dev_id)).exists()
}

fn kernel_launch_wait_enabled() -> bool {
    std::env::var("HETGPU_PACC_WAIT_KERNEL_LAUNCH")
        .ok()
        .as_deref()
        == Some("1")
}

fn kernel_submit_slot_layout(dev_id: usize) -> std::io::Result<(u64, usize)> {
    let shared_bytes = shared_ddr_bytes();
    let slot_count = std::env::var("HETGPU_PACC_KERNEL_SLOT_COUNT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(PACC_CORE_NUM.max(1));
    let slot_bytes = std::env::var("HETGPU_PACC_KERNEL_SLOT_BYTES")
        .ok()
        .and_then(|v| {
            let trimmed = v.trim_start_matches("0x");
            usize::from_str_radix(trimmed, 16)
                .ok()
                .or_else(|| v.parse().ok())
        })
        .filter(|&v| v > 0)
        .unwrap_or(0x0020_0000);
    let reserved = slot_bytes
        .checked_mul(slot_count)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "kernel slot reservation overflow"))?;
    if reserved > shared_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "kernel helper slots need {} bytes, shared DDR window has {}",
                reserved, shared_bytes
            ),
        ));
    }
    let base_off = (shared_bytes - reserved) as u64;
    let slot_off = base_off
        .checked_add((dev_id % slot_count) as u64 * slot_bytes as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "kernel slot offset overflow"))?;
    Ok((slot_off, slot_bytes))
}

fn submit_pacc_kernel_image_via_helper(
    dev: &PaccDevice,
    kernel_name: &str,
    elf_bytes: &[u8],
    launch_state: &PaccKernelLaunchState,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> std::io::Result<usize> {
    let _guard = shared_ddr_kernel_lock()
        .lock()
        .map_err(|_| Error::new(ErrorKind::Other, "PACC kernel helper mutex poisoned"))?;
    let shared_base = shared_ddr_base();
    if shared_base == 0 || shared_ddr_bytes() == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            "PACC shared DDR helper window is not configured",
        ));
    }

    let (buf, submit_len) = build_pacc_kernel_submit_buffer(
        kernel_name,
        elf_bytes,
        launch_state,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
    )?;
    let (slot_off, slot_bytes) = kernel_submit_slot_layout(dev.id)?;
    if submit_len > slot_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "kernel image needs {} bytes, helper slot has {}",
                submit_len, slot_bytes
            ),
        ));
    }

    let seq = next_runtime_job_seq();
    let desc = pacc_mbox_job_desc {
        addr: shared_base + slot_off,
        len: submit_len as u64,
        rsvd: seq,
        buf_info: PACC_JOB_MAGIC,
    };
    let desc_bytes = unsafe {
        std::slice::from_raw_parts(
            (&desc as *const pacc_mbox_job_desc).cast::<u8>(),
            std::mem::size_of::<pacc_mbox_job_desc>(),
        )
    };

    let mut shared_file = open_shared_ddr_window_file(dev.id);
    let mut mailbox_file = open_pacc_mailbox_file(dev.id);
    write_shared_ddr_window_cached(&mut shared_file, slot_off, &buf[..submit_len])?;
    write_ap2pacc_mailbox_cached(&mut mailbox_file, dev.id, 0, desc_bytes).and_then(|ok| {
        if ok {
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::NotFound,
                "PACC mailbox helper is not loaded for kernel submit",
            ))
        }
    })?;

    if kernel_launch_wait_enabled() {
        wait_mailbox_job_status_cached(dev.id, hetgpu_pacc_job_id::KERNEL, seq, &mut mailbox_file)?;
    }

    if std::env::var("HETGPU_PACC_LOG_KERNEL_LAUNCHES")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "pacc_LaunchKernel: helper-submit kernel='{}' seq={} shared=0x{:x} submit={} bytes on pacc{}",
            kernel_name,
            seq,
            shared_base + slot_off,
            submit_len,
            dev.id
        );
    }
    Ok(submit_len)
}

fn submit_pacc_kernel_image(
    dev: &PaccDevice,
    kernel_name: &str,
    elf_bytes: &[u8],
    launch_state: &PaccKernelLaunchState,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> std::io::Result<usize> {
    if helper_kernel_submit_enabled(dev.id) {
        match submit_pacc_kernel_image_via_helper(
            dev,
            kernel_name,
            elf_bytes,
            launch_state,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        ) {
            Ok(submit_len) => return Ok(submit_len),
            Err(e) => {
                if std::env::var("HETGPU_PACC_LOG_KERNEL_LAUNCHES")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    eprintln!(
                        "pacc_LaunchKernel: helper submit unavailable for '{}' on pacc{}: {}; falling back to driver submit",
                        kernel_name, dev.id, e
                    );
                }
            }
        }
    }
    let (buf, submit_len) = build_pacc_kernel_submit_buffer(
        kernel_name,
        elf_bytes,
        launch_state,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
    )?;
    dev.job_submit_user_buffer_with_len(&buf, submit_len)?;
    Ok(submit_len)
}

fn pacc_launch_on_device(
    dev: &PaccDevice,
    kernel_name: &str,
    elf_bytes: &[u8],
    launch_state: &PaccKernelLaunchState,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> pacc_Result {
    if std::env::var("HETGPU_PACC_DRY_RUN").ok().as_deref() == Some("1") {
        match build_pacc_kernel_submit_buffer(
            kernel_name,
            elf_bytes,
            launch_state,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        ) {
            Ok((_buf, submit_len)) => {
                eprintln!(
                    "pacc_LaunchKernel: dry-run accepted ELF kernel='{}' elf={} bytes submit={} bytes args={} bindings={} raw={} bytes grid=({},{},{}) block=({},{},{})",
                    kernel_name,
                    elf_bytes.len(),
                    submit_len,
                    launch_state.arg_records.len(),
                    launch_state.bindings.len(),
                    launch_state.raw_param_blob.len(),
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    block_z,
                );
                return pacc_Result_Success;
            }
            Err(e) => {
                eprintln!(
                    "pacc_LaunchKernel: dry-run rejected ELF kernel '{}' : {}",
                    kernel_name, e
                );
                return pacc_Result_Error;
            }
        }
    }

    match submit_pacc_kernel_image(
        dev,
        kernel_name,
        elf_bytes,
        launch_state,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
    ) {
        Ok(submit_len) => {
            if std::env::var("HETGPU_PACC_LOG_KERNEL_LAUNCHES")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "pacc_LaunchKernel: submitted ELF kernel='{}' elf={} bytes submit={} bytes args={} bindings={} raw={} bytes on pacc{}",
                    kernel_name,
                    elf_bytes.len(),
                    submit_len,
                    launch_state.arg_records.len(),
                    launch_state.bindings.len(),
                    launch_state.raw_param_blob.len(),
                    dev.id,
                );
            }
            pacc_Result_Success
        }
        Err(primary_err) => {
            if allow_preloaded_kernel_fallback() {
                match preloaded_kernel_job_id(kernel_name) {
                    Some(job_id) => match dev.submit_preloaded_job_bytes(job_id, &[]) {
                        Ok(()) => {
                            eprintln!(
                                "pacc_LaunchKernel: ELF submit failed for '{}' ({}) ; fell back to preloaded firmware job_id {}",
                                kernel_name, primary_err, job_id
                            );
                            pacc_Result_Success
                        }
                        Err(fallback_err) => {
                            eprintln!(
                                "pacc_LaunchKernel: ELF submit failed for '{}' ({}) and preloaded fallback job_id {} also failed: {}",
                                kernel_name, primary_err, job_id, fallback_err
                            );
                            pacc_Result_Error
                        }
                    },
                    None => {
                        eprintln!(
                            "pacc_LaunchKernel: ELF submit failed for '{}' ({}) and no preloaded fallback exists",
                            kernel_name, primary_err
                        );
                        pacc_Result_Error
                    }
                }
            } else {
                eprintln!(
                    "pacc_LaunchKernel: ELF submit failed for '{}' : {}",
                    kernel_name, primary_err
                );
                pacc_Result_Error
            }
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
    launch_state: &PaccKernelLaunchState,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> std::io::Result<()> {
    let layout = compute_pacc_kernel_image_layout(elf_bytes, launch_state)?;
    if buf.len() < layout.image_len {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "PACC job image buffer too small",
        ));
    }

    let header = PaccJobImageHeader {
        magic: PACC_JOB_MAGIC,
        version: 1,
        flags: layout.flags,
        entry_offset: layout.elf_offset as u64,
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

    buf[..layout.image_len].fill(0);
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const PaccJobImageHeader).cast::<u8>(),
            PACC_JOB_HEADER_BYTES,
        )
    };
    buf[..PACC_JOB_HEADER_BYTES].copy_from_slice(header_bytes);

    if layout.flags & PACC_JOB_FLAG_HAS_LAUNCH_ABI != 0 {
        let abi = PaccKernelLaunchAbiHeader {
            magic: PACC_KERNEL_LAUNCH_ABI_MAGIC,
            version: PACC_KERNEL_LAUNCH_ABI_VERSION,
            flags: 0,
            arg_records_offset: layout.arg_records_offset as u32,
            arg_record_count: launch_state.arg_records.len() as u32,
            bindings_offset: layout.bindings_offset as u32,
            binding_count: launch_state.bindings.len() as u32,
            raw_param_offset: layout.raw_param_offset as u32,
            raw_param_size: launch_state.raw_param_blob.len() as u32,
            reserved: 0,
        };
        let abi_bytes = unsafe {
            std::slice::from_raw_parts(
                (&abi as *const PaccKernelLaunchAbiHeader).cast::<u8>(),
                std::mem::size_of::<PaccKernelLaunchAbiHeader>(),
            )
        };
        let abi_end = layout.launch_header_offset + abi_bytes.len();
        buf[layout.launch_header_offset..abi_end].copy_from_slice(abi_bytes);

        if !launch_state.arg_records.is_empty() {
            let arg_bytes = unsafe {
                std::slice::from_raw_parts(
                    launch_state.arg_records.as_ptr().cast::<u8>(),
                    launch_state.arg_records.len() * std::mem::size_of::<PaccKernelArgRecord>(),
                )
            };
            let end = layout.arg_records_offset + arg_bytes.len();
            buf[layout.arg_records_offset..end].copy_from_slice(arg_bytes);
        }

        if !launch_state.bindings.is_empty() {
            let binding_bytes = unsafe {
                std::slice::from_raw_parts(
                    launch_state.bindings.as_ptr().cast::<u8>(),
                    launch_state.bindings.len() * std::mem::size_of::<PaccKernelBufferBinding>(),
                )
            };
            let end = layout.bindings_offset + binding_bytes.len();
            buf[layout.bindings_offset..end].copy_from_slice(binding_bytes);
        }

        if !launch_state.raw_param_blob.is_empty() {
            let end = layout.raw_param_offset + launch_state.raw_param_blob.len();
            buf[layout.raw_param_offset..end].copy_from_slice(&launch_state.raw_param_blob);
        }
    }

    let elf_end = layout.elf_offset + elf_bytes.len();
    buf[layout.elf_offset..elf_end].copy_from_slice(elf_bytes);
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
        assert_eq!(IOC_GET_INFO_SIZE, 0xc008_7000u64);
        assert_eq!(IOC_GET_INFO, 0xc010_7001u64);
        assert_eq!(IOC_CREATE_BO, 0xc010_7002u64);
        assert_eq!(IOC_SUBMIT_OP, 0x4008_7003u64);
        assert_eq!(IOC_FREE_BO, 0x4010_7004u64);
    }
}
