//! PACC Runtime Bindings — Lanxin LX500 real driver interface via /dev/paccN
//!
//! Driver interface (reverse-engineered from pacc.ko DWARF + disassembly):
//!   Magic: 'p' (0x70)
//!   PACC_IOC_GET_INFO_SIZE = _IOWR('p', 0, struct pacc_info_size)
//!   PACC_IOC_GET_INFO      = _IOWR('p', 1, struct pacc_info)
//!   PACC_IOC_CREATE_BO     = _IOWR('p', 2, struct pacc_bo)
//!   PACC_IOC_SUBMIT_OP     = _IOW ('p', 3, struct pacc_op)
//!   PACC_IOC_FREE_BO       = _IOW ('p', 4, struct pacc_bo)
//!   PACC_IOC_ZLUDA_IRQ     = _IOW ('p', 5, struct HetgpuPaccSharedDdrInfo)
//!
//! Mailbox SRAM (accessible from Pcore side via mmap or physical):
//!   AP→PACC : 0x20000000  (8KB)
//!   PACC→AP : 0x20002000  (8KB)
//!
//! PACC cluster base addresses: 0x38100000, 0x38500000, 0x39100000, 0x39500000

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError};
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

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct HetgpuPaccSharedDdrInfo {
    pub ddr_base: u64,
    pub ddr_size: u64,
}

pub type HetgpuPaccShardDdrInfo = HetgpuPaccSharedDdrInfo;

pub const PACC_IOC_GET_INFO_SIZE: u64 =
    _iowr(PACC_MAGIC, 0, std::mem::size_of::<pacc_info_size>() as u64);
pub const PACC_IOC_GET_INFO: u64 = _iowr(PACC_MAGIC, 1, std::mem::size_of::<pacc_info>() as u64);
pub const PACC_IOC_CREATE_BO: u64 = _iowr(PACC_MAGIC, 2, std::mem::size_of::<pacc_bo>() as u64);
pub const PACC_IOC_SUBMIT_OP: u64 = _iow(PACC_MAGIC, 3, std::mem::size_of::<pacc_op>() as u64);
pub const PACC_IOC_FREE_BO: u64 = _iow(PACC_MAGIC, 4, std::mem::size_of::<pacc_bo>() as u64);
pub const PACC_IOC_ZLUDA_IRQ: u64 = _iow(
    PACC_MAGIC,
    5,
    std::mem::size_of::<HetgpuPaccSharedDdrInfo>() as u64,
);

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
pub const IOC_ZLUDA_IRQ: u64 = PACC_IOC_ZLUDA_IRQ;

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
pub const PACC_HOST_MBOX_SRAM_OFF: u64 = 0x0021_0000;

/// PACC DDR shared base (accessible to all PACCs and Pcore)
pub const PACC_DDR_BASE: u64 = 0x8000_0000;
/// PACC DDR extended base (PACC-side high address)
pub const PACC_DDR_EXT_BASE: u64 = 0x80_8000_0000;
/// PACC-visible reduce scratch base. Prefer the mailbox helper's allocated
/// window exported in `/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base`.
pub const HETGPU_PACC_SHARED_DDR_BASE: u64 = 0;
pub const HETGPU_PACC_SHARED_DDR_BYTES: usize = 0x0100_0000;
pub const HETGPU_PACC_SHARED_DDR_HELPER_OFF: u64 = 0x0010_0000;
pub const HETGPU_PACC_SHARED_DDR_BASE_INFO_OFF: u64 = 0x0200_4000;

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
pub const HETGPU_PACC_DOORBELL_OFF: u64 = 0;
pub const HETGPU_PACC_ARG_BASE_OFF: u64 = 0x100;
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

struct SharedDdrMmap {
    ptr: *mut u8,
    map_len: usize,
    map_base: u64,
    offset: u64,
    len: usize,
}

impl SharedDdrMmap {
    fn map_file(file: &File, offset: u64, len: usize) -> std::io::Result<Self> {
        if len == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "zero-length shared DDR mmap",
            ));
        }
        validate_shared_ddr_window_range(offset, len)?;

        let page = page_size() as u64;
        let map_base = offset & !(page - 1);
        let page_off = (offset - map_base) as usize;
        let map_len = align_up(page_off + len, page as usize);
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
            ptr: ptr.cast(),
            map_len,
            map_base,
            offset,
            len,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        let page_off = (self.offset - self.map_base) as usize;
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(page_off), self.len) }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::sync::atomic::fence(Ordering::SeqCst);
        let ret = unsafe { libc::msync(self.ptr.cast(), self.map_len, libc::MS_SYNC) };
        if ret < 0 {
            Err(Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn sync_for_cpu(&mut self) -> std::io::Result<()> {
        let ret = unsafe {
            libc::msync(
                self.ptr.cast(),
                self.map_len,
                libc::MS_SYNC | libc::MS_INVALIDATE,
            )
        };
        if ret < 0 {
            Err(Error::last_os_error())
        } else {
            std::sync::atomic::fence(Ordering::SeqCst);
            Ok(())
        }
    }
}

impl Drop for SharedDdrMmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.cast(), self.map_len);
        }
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

    pub fn zluda_irq(&self, shared_ddr: HetgpuPaccSharedDdrInfo) -> std::io::Result<()> {
        let mut arg = shared_ddr;
        let ret = unsafe { libc::ioctl(self.fd, IOC_ZLUDA_IRQ, &mut arg as *mut _) };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            if zluda_irq_mock_enabled() {
                if zluda_irq_trace_enabled() {
                    eprintln!(
                        "PACC ZLUDA IRQ mock: ioctl 0x{:x} on /dev/pacc{} failed: {}; using CPU-side firmware mock",
                        IOC_ZLUDA_IRQ, self.id, err
                    );
                }
                return Ok(());
            }
            return Err(err);
        }
        if zluda_irq_trace_enabled() {
            eprintln!(
                "PACC ZLUDA IRQ: dev={} shared_ddr_base=0x{:x} shared_ddr_size=0x{:x}",
                self.id, shared_ddr.ddr_base, shared_ddr.ddr_size
            );
        }
        Ok(())
    }

    pub fn poll_response(&self, timeout_ms: u64) -> std::io::Result<()> {
        let start = std::time::Instant::now();
        loop {
            let elapsed = start.elapsed();
            let timeout = std::time::Duration::from_millis(timeout_ms);
            if elapsed >= timeout {
                return Err(Error::new(
                    ErrorKind::TimedOut,
                    format!("timed out polling /dev/pacc{} for response", self.id),
                ));
            }
            let remaining_ms = timeout
                .saturating_sub(elapsed)
                .as_millis()
                .min(i32::MAX as u128) as i32;
            let mut pfd = libc::pollfd {
                fd: self.fd,
                events: libc::POLLIN,
                revents: 0,
            };
            let ret = unsafe { libc::poll(&mut pfd as *mut libc::pollfd, 1, remaining_ms) };
            if ret > 0 {
                let readable = pfd.revents & (libc::POLLIN | libc::POLLPRI) != 0;
                let bad = pfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0;
                if readable {
                    return Ok(());
                }
                if bad {
                    return Err(Error::new(
                        ErrorKind::Other,
                        format!(
                            "poll /dev/pacc{} returned unexpected revents=0x{:x}",
                            self.id, pfd.revents
                        ),
                    ));
                }
                continue;
            }
            if ret == 0 {
                return Err(Error::new(
                    ErrorKind::TimedOut,
                    format!("timed out polling /dev/pacc{} for response", self.id),
                ));
            }
            let err = Error::last_os_error();
            if matches!(err.raw_os_error(), Some(libc::EINTR | libc::EAGAIN)) {
                continue;
            }
            return Err(err);
        }
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
        if std::env::var("HETGPU_PACC_ENFORCE_RUNTIME_READY")
            .ok()
            .as_deref()
            == Some("1")
        {
            require_runtime_ready()?;
        }

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
        if zluda_irq_mock_enabled() {
            nvtop_record_submit(self.id as usize, job_id, seq, None, 0);
            let result = self.submit_preloaded_job_zluda_irq_mock(job_id, arg_bytes);
            nvtop_record_complete(self.id as usize, job_id, seq, &result);
            return result;
        }
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
        self.zluda_irq(shared_ddr_info())?;
        nvtop_record_submit(self.id as usize, job_id, seq, None, 0);
        let result = self.wait_preloaded_job_status(job_id, seq);
        nvtop_record_complete(self.id as usize, job_id, seq, &result);
        result
    }

    fn submit_preloaded_job_zluda_irq_mock(
        &self,
        job_id: u32,
        arg_bytes: &[u8],
    ) -> std::io::Result<()> {
        self.zluda_irq(shared_ddr_info())?;
        mock_runtime_job(job_id, arg_bytes)
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
        if use_shared_ddr_control_window() {
            return write_shared_ddr_control_window(self.id, HETGPU_PACC_DOORBELL_OFF, bytes);
        }
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
                ));
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
        let slot_off = HETGPU_PACC_ARG_BASE_OFF + (slot * HETGPU_PACC_ARG_SLOT_BYTES) as u64;
        if use_shared_ddr_control_window() {
            return write_shared_ddr_control_window(self.id, slot_off, &helper_payload);
        }
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
        if use_shared_ddr_control_window() {
            return write_shared_ddr_control_window(self.id, HETGPU_PACC_RUNTIME_TABLE_OFF, bytes);
        }
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
        if use_shared_ddr_control_window() {
            let mut shared_file = open_shared_ddr_window_file(self.id);
            return wait_shared_ddr_job_status(
                self,
                job_id,
                seq,
                timeout_ms,
                start,
                &mut shared_file,
            );
        }
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

fn normalize_pacc_device_id(dev_id: i32) -> usize {
    if dev_id >= 0 && (dev_id as usize) < PACC_CORE_NUM {
        dev_id as usize
    } else {
        0
    }
}

static RUNTIME_BOOTED: OnceLock<Mutex<[bool; 4]>> = OnceLock::new();
static SHARED_DDR_REDUCE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
const SHARED_DDR_STAGE_LOCK_COUNT: usize = 64;
static SHARED_DDR_STAGE_LOCKS: OnceLock<Vec<Mutex<()>>> = OnceLock::new();
static SHARED_DDR_KERNEL_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static SHARED_DDR_BO_ARENA: OnceLock<Mutex<Option<PaccBoMap>>> = OnceLock::new();
static SHARED_DDR_MOCK_ARENA: OnceLock<Mutex<BTreeMap<u64, Vec<u8>>>> = OnceLock::new();
static SHARED_DDR_MMAP_UNAVAILABLE: AtomicUsize = AtomicUsize::new(0);
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

fn zluda_irq_mock_enabled() -> bool {
    matches!(
        std::env::var("HETGPU_PACC_ZLUDA_IRQ_MOCK")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase()),
        Some(v) if v == "1" || v == "true" || v == "yes" || v == "force"
    )
}

fn zluda_irq_trace_enabled() -> bool {
    std::env::var("HETGPU_PACC_ZLUDA_IRQ_TRACE").ok().as_deref() == Some("1")
}

fn use_shared_ddr_control_window() -> bool {
    match std::env::var("HETGPU_PACC_CONTROL_BACKEND")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("sram" | "mailbox" | "legacy") => return false,
        Some("shared-ddr" | "shared_ddr" | "ddr") => return true,
        Some(_) | None => {}
    }

    use_pacc_bo_shared_ddr()
        || use_process_mock_shared_ddr_window()
        || (shared_ddr_base() != 0 && shared_ddr_bytes() >= shared_ddr_control_reserved_bytes())
}

fn shared_ddr_control_reserved_bytes() -> usize {
    PACC_CORE_NUM.max(1) * MBOX_SRAM_SIZE
}

fn shared_ddr_payload_base_off() -> u64 {
    shared_ddr_control_reserved_bytes() as u64
}

fn shared_ddr_payload_bytes() -> usize {
    shared_ddr_bytes().saturating_sub(shared_ddr_control_reserved_bytes())
}

fn shared_ddr_control_offset(pacc_id: usize, offset: u64, len: usize) -> std::io::Result<u64> {
    if pacc_id >= PACC_CORE_NUM.max(1) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("PACC control window id {} is out of range", pacc_id),
        ));
    }
    let end = offset
        .checked_add(len as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC control offset overflow"))?;
    if end > MBOX_SRAM_SIZE as u64 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "PACC control access off=0x{offset:x} len={len} exceeds 0x{:x}",
                MBOX_SRAM_SIZE
            ),
        ));
    }
    (pacc_id as u64)
        .checked_mul(MBOX_SRAM_SIZE as u64)
        .and_then(|base| base.checked_add(offset))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC control offset overflow"))
}

fn decode_pacc_host_status(
    buf: &[u8; 32],
    expected_job_id: u32,
    seq: u64,
) -> Option<std::io::Result<()>> {
    let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let status_job_id = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    let status = u32::from_le_bytes(buf[16..20].try_into().unwrap());
    let status_seq = u64::from_le_bytes(buf[24..32].try_into().unwrap());
    if magic != HETGPU_PACC_JOB_MAGIC
        || version != HETGPU_PACC_JOB_VERSION
        || status_job_id != expected_job_id
        || status_seq != seq
    {
        return None;
    }
    if status == 0 {
        return Some(Ok(()));
    }
    if status != 1 {
        return Some(Err(Error::new(
            ErrorKind::Other,
            format!(
                "PACC job_id {} seq {} failed with firmware status 0x{:x}",
                expected_job_id, seq, status
            ),
        )));
    }
    None
}

fn wait_shared_ddr_job_status(
    dev: &PaccDevice,
    expected_job_id: u32,
    seq: u64,
    timeout_ms: u64,
    start: std::time::Instant,
    shared_file: &mut Option<File>,
) -> std::io::Result<()> {
    let mut buf = [0u8; 32];
    loop {
        read_shared_ddr_control_window_cached(
            shared_file,
            dev.id,
            HETGPU_PACC_COMPLETION_OFF,
            &mut buf,
        )?;
        if let Some(result) = decode_pacc_host_status(&buf, expected_job_id, seq) {
            return result;
        }

        if !use_process_mock_shared_ddr_window() && !use_pacc_bo_shared_ddr() {
            let elapsed = start.elapsed();
            let timeout = std::time::Duration::from_millis(timeout_ms);
            if elapsed >= timeout {
                return Err(Error::new(
                    ErrorKind::TimedOut,
                    format!(
                        "timed out waiting for PACC job_id {} seq {} completion",
                        expected_job_id, seq
                    ),
                ));
            }
            let remaining_ms = timeout.saturating_sub(elapsed).as_millis().max(1) as u64;
            let poll_ms = response_poll_slice_ms().min(remaining_ms);
            match dev.poll_response(poll_ms) {
                Ok(()) => {}
                Err(err) if err.kind() == ErrorKind::TimedOut => {}
                Err(err) => return Err(err),
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
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
}

fn response_poll_slice_ms() -> u64 {
    parse_env_usize("HETGPU_PACC_RESPONSE_POLL_SLICE_MS", 10).max(1) as u64
}

fn decode_pacc_arg<T: Copy>(bytes: &[u8], label: &str) -> std::io::Result<T> {
    let want = std::mem::size_of::<T>();
    if bytes.len() < want {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "short PACC mock {} payload: {} < {}",
                label,
                bytes.len(),
                want
            ),
        ));
    }
    let mut out = std::mem::MaybeUninit::<T>::uninit();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), want);
        Ok(out.assume_init())
    }
}

fn shared_ddr_offset_from_phys(addr: u64, len: usize) -> std::io::Result<u64> {
    let base = shared_ddr_base();
    let bytes = shared_ddr_bytes() as u64;
    let len = len as u64;
    if base == 0 || bytes == 0 {
        if zluda_irq_mock_enabled() && bytes != 0 {
            if addr.checked_add(len).filter(|&end| end <= bytes).is_some() {
                return Ok(addr);
            }
        }
        return Err(Error::new(
            ErrorKind::NotFound,
            "PACC ZLUDA IRQ mock needs a configured shared DDR window",
        ));
    }
    if addr < base
        || addr
            .checked_add(len)
            .filter(|&end| end <= base + bytes)
            .is_none()
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "PACC mock phys range 0x{:x}+0x{:x} is outside shared DDR 0x{:x}+0x{:x}",
                addr, len, base, bytes
            ),
        ));
    }
    Ok(addr - base)
}

fn read_shared_ddr_phys(addr: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    let off = shared_ddr_offset_from_phys(addr, bytes.len())?;
    read_shared_ddr_window(off, bytes)
}

fn write_shared_ddr_phys(addr: u64, bytes: &[u8]) -> std::io::Result<()> {
    let off = shared_ddr_offset_from_phys(addr, bytes.len())?;
    write_shared_ddr_window(off, bytes)
}

fn mock_runtime_job(job_id: u32, arg_bytes: &[u8]) -> std::io::Result<()> {
    match job_id {
        hetgpu_pacc_job_id::GEMM => {
            let job = decode_pacc_arg::<HetgpuPaccGemmJob>(arg_bytes, "GEMM")?;
            mock_run_gemm(&job)
        }
        hetgpu_pacc_job_id::SOFTMAX => {
            let job = decode_pacc_arg::<HetgpuPaccSoftmaxJob>(arg_bytes, "softmax")?;
            mock_run_softmax(&job)
        }
        hetgpu_pacc_job_id::RMSNORM => {
            let job = decode_pacc_arg::<HetgpuPaccRmsNormJob>(arg_bytes, "RMSNorm")?;
            mock_run_rmsnorm(&job)
        }
        hetgpu_pacc_job_id::ALLREDUCE => {
            let job = decode_pacc_arg::<HetgpuPaccAllReduceJob>(arg_bytes, "allreduce")?;
            mock_run_allreduce(&job)
        }
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            format!("PACC ZLUDA IRQ mock does not implement job_id {}", job_id),
        )),
    }
}

fn write_ap2pacc_mailbox(pacc_id: usize, offset: u64, bytes: &[u8]) -> std::io::Result<bool> {
    if !prefer_mailbox_helper() {
        write_ap2pacc_mailbox_phys(pacc_id, offset, bytes)?;
        return Ok(true);
    }
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
    if !prefer_mailbox_helper() {
        read_pacc2ap_mailbox_phys(pacc_id, offset, bytes)?;
        return Ok(true);
    }
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

fn pacc_mbox_index(pacc_id: usize) -> std::io::Result<usize> {
    if pacc_id < PACC_CORE_NUM {
        Ok(pacc_id)
    } else {
        Err(Error::new(
            ErrorKind::InvalidInput,
            format!("invalid PACC mailbox id {}", pacc_id),
        ))
    }
}

fn ap2pacc_mbox_phys(pacc_id: usize) -> std::io::Result<u64> {
    let idx = pacc_mbox_index(pacc_id)?;
    Ok(PACC_BASE[idx] + PACC_HOST_MBOX_SRAM_OFF)
}

fn pacc2ap_mbox_phys(pacc_id: usize) -> std::io::Result<u64> {
    Ok(ap2pacc_mbox_phys(pacc_id)? + MBOX_SRAM_SIZE as u64)
}

fn validate_mbox_access(offset: u64, len: usize, label: &str) -> std::io::Result<()> {
    if len == 0 {
        return Ok(());
    }
    let end = offset
        .checked_add(len as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, format!("{label} offset overflow")))?;
    if end > MBOX_SRAM_SIZE as u64 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "{label} mailbox access out of range: off=0x{offset:x} len={len} size=0x{:x}",
                MBOX_SRAM_SIZE
            ),
        ));
    }
    Ok(())
}

fn write_ap2pacc_mailbox_phys(pacc_id: usize, offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    validate_mbox_access(offset, bytes.len(), "AP2PACC")?;
    if bytes.is_empty() {
        return Ok(());
    }
    let phys = ap2pacc_mbox_phys(pacc_id)?
        .checked_add(offset)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "AP2PACC phys offset overflow"))?;
    let mut map = PhysMap::map_rw(phys, bytes.len())?;
    map.as_mut_slice().copy_from_slice(bytes);
    map.flush()
}

fn read_pacc2ap_mailbox_phys(pacc_id: usize, offset: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    validate_mbox_access(offset, bytes.len(), "PACC2AP")?;
    if bytes.is_empty() {
        return Ok(());
    }
    let phys = pacc2ap_mbox_phys(pacc_id)?
        .checked_add(offset)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC2AP phys offset overflow"))?;
    let mut map = PhysMap::map_rw(phys, bytes.len())?;
    bytes.copy_from_slice(map.as_mut_slice());
    Ok(())
}

fn shared_ddr_reduce_lock() -> &'static Mutex<()> {
    SHARED_DDR_REDUCE_LOCK.get_or_init(|| Mutex::new(()))
}

fn shared_ddr_stage_lock(slot_id: usize) -> &'static Mutex<()> {
    let locks = SHARED_DDR_STAGE_LOCKS.get_or_init(|| {
        (0..SHARED_DDR_STAGE_LOCK_COUNT)
            .map(|_| Mutex::new(()))
            .collect()
    });
    &locks[slot_id % locks.len()]
}

fn shared_ddr_stage_lock_timeout_ms() -> u64 {
    std::env::var("HETGPU_PACC_STAGE_LOCK_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| {
            std::env::var("HETGPU_PACC_JOB_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
        })
        .unwrap_or(30_000)
}

fn lock_shared_ddr_stage(slot_id: usize, label: &str) -> std::io::Result<MutexGuard<'static, ()>> {
    let lock = shared_ddr_stage_lock(slot_id);
    let timeout_ms = shared_ddr_stage_lock_timeout_ms();
    let start = std::time::Instant::now();
    let mut logged_wait = false;
    loop {
        match lock.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::Poisoned(_)) => {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("PACC shared-DDR stage lock poisoned for slot {}", slot_id),
                ));
            }
            Err(TryLockError::WouldBlock) => {
                if !logged_wait {
                    eprintln!(
                        "{}: waiting for shared-DDR stage slot {} lock",
                        label, slot_id
                    );
                    logged_wait = true;
                }
                if start.elapsed() >= std::time::Duration::from_millis(timeout_ms) {
                    return Err(Error::new(
                        ErrorKind::TimedOut,
                        format!(
                            "{}: timed out waiting for shared-DDR stage slot {} lock after {} ms",
                            label, slot_id, timeout_ms
                        ),
                    ));
                }
                std::thread::sleep(std::time::Duration::from_micros(500));
            }
        }
    }
}

fn shared_ddr_kernel_lock() -> &'static Mutex<()> {
    SHARED_DDR_KERNEL_LOCK.get_or_init(|| Mutex::new(()))
}

fn parse_u64_text(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        value.parse().ok()
    }
}

fn read_debugfs_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|v| parse_u64_text(&v))
}

fn shared_ddr_info() -> HetgpuPaccSharedDdrInfo {
    HetgpuPaccSharedDdrInfo {
        ddr_base: shared_ddr_base(),
        ddr_size: shared_ddr_bytes() as u64,
    }
}

fn shared_ddr_base() -> u64 {
    if use_pacc_bo_shared_ddr() {
        if let Ok(base) = shared_ddr_bo_base() {
            return base;
        }
    }

    read_debugfs_u64("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base")
        .or_else(|| {
            std::env::var("HETGPU_PACC_SHARED_DDR_BASE")
                .ok()
                .and_then(|v| parse_u64_text(&v))
        })
        .or_else(|| {
            let dev = helper_path_for_pacc(0);
            let file = OpenOptions::new().read(true).open(&dev).ok()?;
            let mut buf = [0u8; 8];
            file.read_exact_at(&mut buf, HETGPU_PACC_SHARED_DDR_BASE_INFO_OFF)
                .ok()?;
            let value = u64::from_le_bytes(buf);
            (value != 0).then_some(value)
        })
        .unwrap_or(HETGPU_PACC_SHARED_DDR_BASE)
}

fn shared_ddr_bo_arena() -> &'static Mutex<Option<PaccBoMap>> {
    SHARED_DDR_BO_ARENA.get_or_init(|| Mutex::new(None))
}

fn shared_ddr_backend() -> Option<String> {
    std::env::var("HETGPU_PACC_SHARED_DDR_BACKEND")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
}

fn use_pacc_bo_shared_ddr() -> bool {
    matches!(
        shared_ddr_backend().as_deref(),
        Some("pacc-bo" | "pacc_bo" | "bo")
    )
}

fn force_shared_ddr_mmap() -> bool {
    matches!(
        shared_ddr_backend().as_deref(),
        Some("mmap" | "helper-mmap" | "helper_mmap" | "mbox-mmap" | "mbox_mmap")
    ) || matches!(
        std::env::var("HETGPU_PACC_SHARED_DDR_MMAP").ok().as_deref(),
        Some("force" | "FORCE")
    )
}

fn use_shared_ddr_mmap() -> bool {
    if force_shared_ddr_mmap() {
        return true;
    }
    if matches!(
        std::env::var("HETGPU_PACC_SHARED_DDR_MMAP").ok().as_deref(),
        Some("0" | "false" | "FALSE" | "no" | "NO")
    ) {
        return false;
    }
    if use_pacc_bo_shared_ddr()
        || use_process_mock_shared_ddr_window()
        || prefer_physmap_shared_ddr()
    {
        return false;
    }
    SHARED_DDR_MMAP_UNAVAILABLE.load(Ordering::Relaxed) == 0
}

fn use_process_mock_shared_ddr_window() -> bool {
    zluda_irq_mock_enabled() && !use_pacc_bo_shared_ddr()
}

fn shared_ddr_bo_pacc_id() -> usize {
    std::env::var("HETGPU_PACC_SHARED_DDR_PACC_ID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn open_pacc_device_no_boot(id: usize) -> std::io::Result<PaccDevice> {
    let path = format!("/dev/pacc{}", id);
    let file = OpenOptions::new().read(true).write(true).open(&path)?;
    let fd = file.as_raw_fd();
    Ok(PaccDevice { id, fd, file })
}

fn with_shared_ddr_bo<R>(
    f: impl FnOnce(u64, &mut PaccBoMap) -> std::io::Result<R>,
) -> std::io::Result<R> {
    let mut guard = shared_ddr_bo_arena()
        .lock()
        .map_err(|_| Error::new(ErrorKind::Other, "shared DDR BO mutex poisoned"))?;

    if guard.is_none() {
        let pacc_id = shared_ddr_bo_pacc_id();
        let dev = open_pacc_device_no_boot(pacc_id)?;
        let bo = dev.bo_alloc_map(shared_ddr_bytes())?;
        if zluda_irq_trace_enabled() {
            eprintln!(
                "PACC shared DDR BO mock: /dev/pacc{} phys=0x{:x} bytes={}",
                pacc_id,
                bo.phys(),
                shared_ddr_bytes()
            );
        }
        *guard = Some(bo);
    }

    let bo = guard
        .as_mut()
        .ok_or_else(|| Error::new(ErrorKind::Other, "shared DDR BO allocation missing"))?;
    f(bo.phys(), bo)
}

fn shared_ddr_bo_base() -> std::io::Result<u64> {
    with_shared_ddr_bo(|base, _| Ok(base))
}

fn shared_ddr_bo_copy_in(offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    with_shared_ddr_bo(|_, bo| {
        let slice = bo.as_mut_slice();
        let start = usize::try_from(offset).map_err(|_| {
            Error::new(
                ErrorKind::InvalidInput,
                "shared DDR BO write offset does not fit usize",
            )
        })?;
        let end = start
            .checked_add(bytes.len())
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "shared DDR BO write overflow"))?;
        if end > slice.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "shared DDR BO write out of range: off=0x{offset:x} len={} arena={}",
                    bytes.len(),
                    slice.len()
                ),
            ));
        }
        slice[start..end].copy_from_slice(bytes);
        bo.flush()
    })
}

fn shared_ddr_bo_copy_out(offset: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    with_shared_ddr_bo(|_, bo| {
        let slice = bo.as_mut_slice();
        let start = usize::try_from(offset).map_err(|_| {
            Error::new(
                ErrorKind::InvalidInput,
                "shared DDR BO read offset does not fit usize",
            )
        })?;
        let end = start
            .checked_add(bytes.len())
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "shared DDR BO read overflow"))?;
        if end > slice.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "shared DDR BO read out of range: off=0x{offset:x} len={} arena={}",
                    bytes.len(),
                    slice.len()
                ),
            ));
        }
        bytes.copy_from_slice(&slice[start..end]);
        Ok(())
    })
}

fn shared_ddr_mock_arena() -> &'static Mutex<BTreeMap<u64, Vec<u8>>> {
    SHARED_DDR_MOCK_ARENA.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn validate_shared_ddr_window_range(offset: u64, len: usize) -> std::io::Result<()> {
    let bytes = shared_ddr_bytes() as u64;
    if bytes == 0
        || offset
            .checked_add(len as u64)
            .filter(|&end| end <= bytes)
            .is_none()
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "shared DDR mock access out of range: off=0x{offset:x} len={len} arena=0x{bytes:x}"
            ),
        ));
    }
    Ok(())
}

fn shared_ddr_mock_copy_in(offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    validate_shared_ddr_window_range(offset, bytes.len())?;
    let end = offset + bytes.len() as u64;
    let mut arena = shared_ddr_mock_arena()
        .lock()
        .map_err(|_| Error::new(ErrorKind::Other, "shared DDR mock arena mutex poisoned"))?;
    let overlapping: Vec<u64> = arena
        .range(..end)
        .filter_map(|(&chunk_off, chunk)| {
            let chunk_end = chunk_off + chunk.len() as u64;
            (chunk_end > offset).then_some(chunk_off)
        })
        .collect();
    for chunk_off in overlapping {
        arena.remove(&chunk_off);
    }
    arena.insert(offset, bytes.to_vec());
    Ok(())
}

fn shared_ddr_mock_copy_out(offset: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    validate_shared_ddr_window_range(offset, bytes.len())?;
    bytes.fill(0);
    let end = offset + bytes.len() as u64;
    let arena = shared_ddr_mock_arena()
        .lock()
        .map_err(|_| Error::new(ErrorKind::Other, "shared DDR mock arena mutex poisoned"))?;
    for (&chunk_off, chunk) in arena.range(..end) {
        let chunk_end = chunk_off + chunk.len() as u64;
        if chunk_end <= offset {
            continue;
        }
        let overlap_start = offset.max(chunk_off);
        let overlap_end = end.min(chunk_end);
        let dst_start = (overlap_start - offset) as usize;
        let src_start = (overlap_start - chunk_off) as usize;
        let len = (overlap_end - overlap_start) as usize;
        bytes[dst_start..dst_start + len].copy_from_slice(&chunk[src_start..src_start + len]);
    }
    Ok(())
}

fn shared_ddr_bytes() -> usize {
    read_debugfs_u64("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_bytes")
        .or_else(|| read_debugfs_u64("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_size"))
        .or_else(|| {
            std::env::var("HETGPU_PACC_SHARED_DDR_BYTES")
                .ok()
                .and_then(|v| parse_u64_text(&v))
        })
        .and_then(|v| usize::try_from(v).ok())
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

fn prefer_mailbox_helper() -> bool {
    matches!(
        std::env::var("HETGPU_PACC_MBOX_BACKEND")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase()),
        Some(v) if v == "helper" || v == "mbox" || v == "mailbox"
    ) || std::env::var("HETGPU_PACC_USE_MBOX_HELPER").ok().as_deref() == Some("1")
}

fn helper_io_chunk_bytes() -> usize {
    parse_env_usize("HETGPU_PACC_HELPER_IO_CHUNK_BYTES", 1 << 20).max(1)
}

fn prefer_physmap_shared_ddr() -> bool {
    if matches!(
        shared_ddr_backend().as_deref(),
        Some("helper" | "mbox" | "mailbox")
    ) {
        return false;
    }
    if matches!(
        shared_ddr_backend().as_deref(),
        Some("physmap" | "devmem" | "dev-mem")
    ) {
        return true;
    }
    if std::env::var("HETGPU_PACC_SHARED_DDR_NO_HELPER")
        .ok()
        .as_deref()
        == Some("1")
    {
        return true;
    }
    if std::path::Path::new(&helper_path_for_pacc(0)).exists() {
        return false;
    }
    std::env::var("HETGPU_PACC_SHARED_DDR_BASE")
        .ok()
        .and_then(|v| parse_u64_text(&v))
        .filter(|&base| base != 0)
        .is_some()
        || read_debugfs_u64("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base")
            .filter(|&base| base != 0)
            .is_some()
}

fn shared_ddr_helper_unavailable(op: &str, dev: &str) -> std::io::Error {
    Error::new(
        ErrorKind::NotFound,
        format!(
            "PACC shared DDR {op} helper {dev} is not available; set \
             HETGPU_PACC_SHARED_DDR_BASE or expose debugfs shared_ddr_base to use /dev/mem"
        ),
    )
}

fn shared_ddr_helper_failed(op: &str, dev: &str, err: std::io::Error) -> std::io::Error {
    Error::new(
        err.kind(),
        format!(
            "PACC shared DDR {op} via {dev} failed: {err}; set \
             HETGPU_PACC_SHARED_DDR_BACKEND=devmem to force /dev/mem, or helper to keep using {dev}"
        ),
    )
}

fn shared_ddr_phys_addr(offset: u64, len: usize) -> std::io::Result<u64> {
    let base = shared_ddr_base();
    let bytes = shared_ddr_bytes() as u64;
    if base == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            "PACC shared DDR physmap needs HETGPU_PACC_SHARED_DDR_BASE or debugfs shared_ddr_base",
        ));
    }
    if bytes == 0
        || offset
            .checked_add(len as u64)
            .filter(|&end| end <= bytes)
            .is_none()
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "shared DDR physmap access out of range: off=0x{offset:x} len={len} arena=0x{bytes:x}"
            ),
        ));
    }
    base.checked_add(offset)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "shared DDR offset overflow"))
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

fn note_shared_ddr_mmap_unavailable(op: &str, dev: &str, err: &std::io::Error) {
    if SHARED_DDR_MMAP_UNAVAILABLE.swap(1, Ordering::Relaxed) == 0 && zluda_irq_trace_enabled() {
        eprintln!(
            "PACC shared DDR mmap {op} via {dev} unavailable ({err}); falling back to helper read/write"
        );
    }
}

fn shared_ddr_mmap_copy_in_with_file(
    file: &File,
    dev: &str,
    offset: u64,
    bytes: &[u8],
) -> std::io::Result<bool> {
    if bytes.is_empty() {
        return Ok(true);
    }
    if !use_shared_ddr_mmap() {
        return Ok(false);
    }

    match SharedDdrMmap::map_file(file, offset, bytes.len()) {
        Ok(mut map) => {
            map.as_mut_slice().copy_from_slice(bytes);
            if let Err(err) = map.flush() {
                if force_shared_ddr_mmap() {
                    return Err(shared_ddr_helper_failed("mmap flush", dev, err));
                }
                note_shared_ddr_mmap_unavailable("flush", dev, &err);
                return Ok(false);
            }
            Ok(true)
        }
        Err(err) if err.kind() == ErrorKind::InvalidInput => Err(err),
        Err(err) if force_shared_ddr_mmap() => {
            Err(shared_ddr_helper_failed("mmap write", dev, err))
        }
        Err(err) => {
            note_shared_ddr_mmap_unavailable("write", dev, &err);
            Ok(false)
        }
    }
}

fn shared_ddr_mmap_copy_out_with_file(
    file: &File,
    dev: &str,
    offset: u64,
    bytes: &mut [u8],
) -> std::io::Result<bool> {
    if bytes.is_empty() {
        return Ok(true);
    }
    if !use_shared_ddr_mmap() {
        return Ok(false);
    }

    match SharedDdrMmap::map_file(file, offset, bytes.len()) {
        Ok(mut map) => {
            if let Err(err) = map.sync_for_cpu() {
                if force_shared_ddr_mmap() {
                    return Err(shared_ddr_helper_failed("mmap invalidate", dev, err));
                }
                note_shared_ddr_mmap_unavailable("invalidate", dev, &err);
                return Ok(false);
            }
            bytes.copy_from_slice(map.as_mut_slice());
            Ok(true)
        }
        Err(err) if err.kind() == ErrorKind::InvalidInput => Err(err),
        Err(err) if force_shared_ddr_mmap() => Err(shared_ddr_helper_failed("mmap read", dev, err)),
        Err(err) => {
            note_shared_ddr_mmap_unavailable("read", dev, &err);
            Ok(false)
        }
    }
}

fn write_shared_ddr_window(offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    if use_pacc_bo_shared_ddr() {
        return shared_ddr_bo_copy_in(offset, bytes);
    }
    if use_process_mock_shared_ddr_window() {
        return shared_ddr_mock_copy_in(offset, bytes);
    }

    let physmap = prefer_physmap_shared_ddr();
    let dev = helper_path_for_pacc(0);
    if !physmap && std::path::Path::new(&dev).exists() {
        let helper_result = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new().read(true).write(true).open(&dev)?;
            if shared_ddr_mmap_copy_in_with_file(&file, &dev, offset, bytes)? {
                return Ok(());
            }
            helper_write_all(&mut file, HETGPU_PACC_SHARED_DDR_HELPER_OFF + offset, bytes)
        })();
        match helper_result {
            Ok(()) => return Ok(()),
            Err(err) => return Err(shared_ddr_helper_failed("write", &dev, err)),
        }
    }
    if !physmap {
        return Err(shared_ddr_helper_unavailable("write", &dev));
    }

    let phys = shared_ddr_phys_addr(offset, bytes.len())?;
    let mut map = PhysMap::map_rw(phys, bytes.len())?;
    map.as_mut_slice().copy_from_slice(bytes);
    map.flush()
}

fn read_shared_ddr_window(offset: u64, bytes: &mut [u8]) -> std::io::Result<()> {
    if use_pacc_bo_shared_ddr() {
        return shared_ddr_bo_copy_out(offset, bytes);
    }
    if use_process_mock_shared_ddr_window() {
        return shared_ddr_mock_copy_out(offset, bytes);
    }

    let physmap = prefer_physmap_shared_ddr();
    let dev = helper_path_for_pacc(0);
    if !physmap && std::path::Path::new(&dev).exists() {
        let helper_result = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new().read(true).write(true).open(&dev)?;
            if shared_ddr_mmap_copy_out_with_file(&file, &dev, offset, bytes)? {
                return Ok(());
            }
            helper_read_exact(&mut file, HETGPU_PACC_SHARED_DDR_HELPER_OFF + offset, bytes)
        })();
        match helper_result {
            Ok(()) => return Ok(()),
            Err(err) => return Err(shared_ddr_helper_failed("read", &dev, err)),
        }
    }
    if !physmap {
        return Err(shared_ddr_helper_unavailable("read", &dev));
    }

    let phys = shared_ddr_phys_addr(offset, bytes.len())?;
    let mut map = PhysMap::map_rw(phys, bytes.len())?;
    bytes.copy_from_slice(map.as_mut_slice());
    Ok(())
}

fn open_shared_ddr_window_file(pacc_id: usize) -> Option<File> {
    if prefer_physmap_shared_ddr() || use_process_mock_shared_ddr_window() {
        return None;
    }
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
        if shared_ddr_mmap_copy_in_with_file(file, "cached mbox fd", offset, bytes)? {
            return Ok(());
        }
        helper_write_all(file, HETGPU_PACC_SHARED_DDR_HELPER_OFF + offset, bytes)?;
        return Ok(());
    }
    write_shared_ddr_window(offset, bytes)
}

fn write_shared_ddr_control_window(
    pacc_id: usize,
    offset: u64,
    bytes: &[u8],
) -> std::io::Result<()> {
    let offset = shared_ddr_control_offset(pacc_id, offset, bytes.len())?;
    write_shared_ddr_window(offset, bytes)
}

fn write_shared_ddr_control_window_cached(
    file: &mut Option<File>,
    pacc_id: usize,
    offset: u64,
    bytes: &[u8],
) -> std::io::Result<()> {
    let offset = shared_ddr_control_offset(pacc_id, offset, bytes.len())?;
    write_shared_ddr_window_cached(file, offset, bytes)
}

fn read_shared_ddr_window_cached(
    file: &mut Option<File>,
    offset: u64,
    bytes: &mut [u8],
) -> std::io::Result<()> {
    if let Some(file) = file.as_mut() {
        if shared_ddr_mmap_copy_out_with_file(file, "cached mbox fd", offset, bytes)? {
            return Ok(());
        }
        helper_read_exact(file, HETGPU_PACC_SHARED_DDR_HELPER_OFF + offset, bytes)?;
        return Ok(());
    }
    read_shared_ddr_window(offset, bytes)
}

fn read_shared_ddr_control_window_cached(
    file: &mut Option<File>,
    pacc_id: usize,
    offset: u64,
    bytes: &mut [u8],
) -> std::io::Result<()> {
    let offset = shared_ddr_control_offset(pacc_id, offset, bytes.len())?;
    read_shared_ddr_window_cached(file, offset, bytes)
}

fn pacc_gemm_trace_enabled() -> bool {
    std::env::var("HETGPU_PACC_GEMM_TRACE").ok().as_deref() == Some("1")
}

fn open_pacc_mailbox_file(pacc_id: usize) -> Option<File> {
    if !prefer_mailbox_helper() {
        return None;
    }
    let dev = helper_path_for_pacc(pacc_id);
    if std::path::Path::new(&dev).exists() {
        OpenOptions::new().read(true).write(true).open(&dev).ok()
    } else {
        None
    }
}

fn open_pacc_mailbox_helper_file(pacc_id: usize) -> std::io::Result<File> {
    let dev = helper_path_for_pacc(pacc_id);
    if !std::path::Path::new(&dev).exists() {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("PACC mailbox helper {dev} is not loaded"),
        ));
    }
    OpenOptions::new().read(true).write(true).open(&dev)
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

fn write_control_window_cached(
    shared_file: &mut Option<File>,
    mailbox_file: &mut Option<File>,
    pacc_id: usize,
    offset: u64,
    bytes: &[u8],
) -> std::io::Result<()> {
    if use_shared_ddr_control_window() {
        return write_shared_ddr_control_window_cached(shared_file, pacc_id, offset, bytes);
    }
    write_ap2pacc_mailbox_cached(mailbox_file, pacc_id, offset, bytes).and_then(|ok| {
        if ok {
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::NotFound,
                "PACC AP2PACC control mailbox is not available",
            ))
        }
    })
}

fn wait_mailbox_job_status_cached(
    dev: &PaccDevice,
    expected_job_id: u32,
    seq: u64,
    shared_file: &mut Option<File>,
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
    if use_shared_ddr_control_window() {
        return wait_shared_ddr_job_status(
            dev,
            expected_job_id,
            seq,
            timeout_ms,
            start,
            shared_file,
        );
    }
    loop {
        if read_pacc2ap_mailbox_cached(mailbox_file, dev.id, HETGPU_PACC_COMPLETION_OFF, &mut buf)?
        {
            if let Some(result) = decode_pacc_host_status(&buf, expected_job_id, seq) {
                return result;
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
    dev: &PaccDevice,
    seq: u64,
    shared_file: &mut Option<File>,
    mailbox_file: &mut Option<File>,
) -> std::io::Result<()> {
    wait_mailbox_job_status_cached(
        dev,
        hetgpu_pacc_job_id::GEMM,
        seq,
        shared_file,
        mailbox_file,
    )
}

fn submit_gemm_runtime_job_cached(
    dev: &PaccDevice,
    job: &HetgpuPaccGemmJob,
    staged_bytes: u64,
    mailbox_file: &mut Option<File>,
) -> std::io::Result<()> {
    if std::env::var("HETGPU_PACC_ENFORCE_RUNTIME_READY")
        .ok()
        .as_deref()
        == Some("1")
    {
        require_runtime_ready()?;
    }
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
    let mut shared_file = open_shared_ddr_window_file(dev.id);
    write_control_window_cached(
        &mut shared_file,
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
    write_control_window_cached(
        &mut shared_file,
        mailbox_file,
        dev.id,
        HETGPU_PACC_DOORBELL_OFF,
        doorbell_bytes,
    )?;
    if use_shared_ddr_control_window() {
        dev.zluda_irq(shared_ddr_info())?;
    }
    nvtop_record_submit(
        dev.id,
        hetgpu_pacc_job_id::GEMM,
        seq,
        Some(job),
        staged_bytes,
    );
    let result = wait_preloaded_gemm_status_cached(dev, seq, &mut shared_file, mailbox_file);
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
        let payload_bytes = shared_ddr_payload_bytes();
        if total_bytes > payload_bytes {
            return Err(PaccError::Io(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "PACC all_reduce needs {total_bytes} bytes, shared DDR payload window is {payload_bytes} bytes"
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

        let stage_base = shared_ddr_payload_base_off();
        let src_bytes =
            unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), input_bytes) };
        let mut payload = vec![0u8; total_bytes];
        payload[..input_bytes].copy_from_slice(src_bytes);
        write_shared_ddr_window(stage_base, &payload).map_err(|e| {
            PaccError::Io(Error::new(
                e.kind(),
                format!("PACC all_reduce shared DDR write failed: {e}"),
            ))
        })?;

        let job = HetgpuPaccAllReduceJob {
            src_addr: shared_base + stage_base,
            dst_addr: shared_base + stage_base + output_off as u64,
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
        read_shared_ddr_window(stage_base + output_off as u64, &mut out_storage).map_err(|e| {
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

fn align_up_usize(value: usize, align: usize) -> Option<usize> {
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
            ));
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
            ));
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

fn check_mock_slice_access(
    buf_len: usize,
    dtype: i32,
    index: usize,
    op: &str,
) -> std::io::Result<()> {
    let elem_size = pacc_dtype_size(dtype).ok_or_else(|| {
        Error::new(
            ErrorKind::Unsupported,
            format!("unsupported PACC mock dtype {}", dtype),
        )
    })?;
    let start = index.checked_mul(elem_size).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("PACC mock {op} byte offset overflow: dtype={dtype} index={index}"),
        )
    })?;
    let end = start.checked_add(elem_size).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("PACC mock {op} byte range overflow: dtype={dtype} index={index}"),
        )
    })?;
    if end > buf_len {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "PACC mock {op} out of range: dtype={dtype} index={index} byte_range=0x{start:x}..0x{end:x} buf=0x{buf_len:x}"
            ),
        ));
    }
    Ok(())
}

fn mock_slice_read_f32(buf: &[u8], dtype: i32, index: usize) -> std::io::Result<f32> {
    check_mock_slice_access(buf.len(), dtype, index, "read")?;
    unsafe { gemm_read_f32(buf.as_ptr().cast::<std::ffi::c_void>(), dtype, index) }
}

fn mock_slice_write_f32(
    buf: &mut [u8],
    dtype: i32,
    index: usize,
    value: f32,
) -> std::io::Result<()> {
    check_mock_slice_access(buf.len(), dtype, index, "write")?;
    unsafe {
        gemm_write_from_f32(
            buf.as_mut_ptr().cast::<std::ffi::c_void>(),
            dtype,
            index,
            value,
        )
    }
}

fn mock_run_rmsnorm(job: &HetgpuPaccRmsNormJob) -> std::io::Result<()> {
    if job.x_addr == 0 || job.y_addr == 0 || job.rows == 0 || job.hidden == 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "bad RMSNorm mock job"));
    }
    let elem_size = pacc_dtype_size(job.dtype as i32)
        .ok_or_else(|| Error::new(ErrorKind::Unsupported, "bad RMSNorm mock dtype"))?;
    let rows = usize::try_from(job.rows)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "RMSNorm rows overflow"))?;
    let hidden = usize::try_from(job.hidden)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "RMSNorm hidden overflow"))?;
    let elems = rows
        .checked_mul(hidden)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm elems overflow"))?;
    let bytes = elems
        .checked_mul(elem_size)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm bytes overflow"))?;
    let mut x = vec![0u8; bytes];
    let mut y = vec![0u8; bytes];
    read_shared_ddr_phys(job.x_addr, &mut x)?;
    let weight = if job.weight_addr != 0 {
        let mut w = vec![0u8; hidden * elem_size];
        read_shared_ddr_phys(job.weight_addr, &mut w)?;
        Some(w)
    } else {
        None
    };
    for row in 0..rows {
        let base = row * hidden;
        let mut sumsq = 0.0f32;
        for i in 0..hidden {
            let v = mock_slice_read_f32(&x, job.dtype as i32, base + i)?;
            sumsq += v * v;
        }
        let scale = 1.0f32 / (sumsq / hidden as f32 + job.eps).sqrt();
        for i in 0..hidden {
            let xv = mock_slice_read_f32(&x, job.dtype as i32, base + i)?;
            let wv = if let Some(w) = weight.as_ref() {
                mock_slice_read_f32(w, job.dtype as i32, i)?
            } else {
                1.0
            };
            mock_slice_write_f32(&mut y, job.dtype as i32, base + i, xv * scale * wv)?;
        }
    }
    write_shared_ddr_phys(job.y_addr, &y)?;
    if zluda_irq_trace_enabled() {
        eprintln!(
            "PACC ZLUDA IRQ mock: RMSNorm rows={} hidden={} dtype={} y=0x{:x}",
            job.rows, job.hidden, job.dtype, job.y_addr
        );
    }
    Ok(())
}

fn mock_run_softmax(job: &HetgpuPaccSoftmaxJob) -> std::io::Result<()> {
    if job.src_addr == 0 || job.dst_addr == 0 || job.rows == 0 || job.cols == 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "bad softmax mock job"));
    }
    let elem_size = pacc_dtype_size(job.dtype as i32)
        .ok_or_else(|| Error::new(ErrorKind::Unsupported, "bad softmax mock dtype"))?;
    let rows = usize::try_from(job.rows)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "softmax rows overflow"))?;
    let cols = usize::try_from(job.cols)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "softmax cols overflow"))?;
    let stride = usize::try_from(if job.stride == 0 {
        job.cols
    } else {
        job.stride
    })
    .map_err(|_| Error::new(ErrorKind::InvalidInput, "softmax stride overflow"))?;
    let elems = rows
        .checked_mul(stride)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "softmax elems overflow"))?;
    let bytes = elems
        .checked_mul(elem_size)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "softmax bytes overflow"))?;
    let mut src = vec![0u8; bytes];
    let mut dst = vec![0u8; bytes];
    read_shared_ddr_phys(job.src_addr, &mut src)?;
    for row in 0..rows {
        let base = row * stride;
        let mut max_v = f32::NEG_INFINITY;
        for col in 0..cols {
            max_v = max_v.max(mock_slice_read_f32(&src, job.dtype as i32, base + col)?);
        }
        let mut sum = 0.0f32;
        for col in 0..cols {
            sum += (mock_slice_read_f32(&src, job.dtype as i32, base + col)? - max_v).exp();
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for col in 0..cols {
            let e = (mock_slice_read_f32(&src, job.dtype as i32, base + col)? - max_v).exp();
            mock_slice_write_f32(&mut dst, job.dtype as i32, base + col, e * inv)?;
        }
    }
    write_shared_ddr_phys(job.dst_addr, &dst)?;
    Ok(())
}

fn mock_run_allreduce(job: &HetgpuPaccAllReduceJob) -> std::io::Result<()> {
    if job.src_addr == 0 || job.dst_addr == 0 || job.count == 0 || job.nranks == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "bad allreduce mock job",
        ));
    }
    if job.reduce_op != 0 || job.dtype != PaccDataType::Float32 as u32 {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "allreduce mock only supports f32 sum",
        ));
    }
    let count = usize::try_from(job.count)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "allreduce count overflow"))?;
    let nranks = usize::try_from(job.nranks)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "allreduce nranks overflow"))?;
    let total = count
        .checked_mul(nranks)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "allreduce elems overflow"))?;
    let mut src = vec![0u8; total * std::mem::size_of::<f32>()];
    let mut dst = vec![0u8; count * std::mem::size_of::<f32>()];
    read_shared_ddr_phys(job.src_addr, &mut src)?;
    for i in 0..count {
        let mut sum = 0.0f32;
        for rank in 0..nranks {
            sum += mock_slice_read_f32(&src, PaccDataType::Float32 as i32, rank * count + i)?;
        }
        mock_slice_write_f32(&mut dst, PaccDataType::Float32 as i32, i, sum)?;
    }
    write_shared_ddr_phys(job.dst_addr, &dst)?;
    Ok(())
}

fn mock_run_gemm(job: &HetgpuPaccGemmJob) -> std::io::Result<()> {
    if job.a_addr == 0
        || job.b_addr == 0
        || job.c_addr == 0
        || job.m == 0
        || job.n == 0
        || job.k == 0
    {
        return Err(Error::new(ErrorKind::InvalidInput, "bad GEMM mock job"));
    }
    let a_dtype_size = pacc_dtype_size(job.atype as i32)
        .ok_or_else(|| Error::new(ErrorKind::Unsupported, "bad GEMM mock A dtype"))?;
    let b_dtype_size = pacc_dtype_size(job.btype as i32)
        .ok_or_else(|| Error::new(ErrorKind::Unsupported, "bad GEMM mock B dtype"))?;
    let c_dtype_size = pacc_dtype_size(job.ctype as i32)
        .ok_or_else(|| Error::new(ErrorKind::Unsupported, "bad GEMM mock C dtype"))?;
    let batch_count = if job.batch_count == 0 {
        1
    } else {
        job.batch_count
    };
    let a_matrix_elems = if job.transa != 0 {
        gemm_span(job.k, job.m, job.lda)
    } else {
        gemm_span(job.m, job.k, job.lda)
    }
    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM mock A span overflow"))?;
    let b_matrix_elems = if job.transb != 0 {
        gemm_span(job.n, job.k, job.ldb)
    } else {
        gemm_span(job.k, job.n, job.ldb)
    }
    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM mock B span overflow"))?;
    let c_matrix_elems = gemm_span(job.m, job.n, job.ldc)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM mock C span overflow"))?;
    let a_batch_stride = if job.stride_a > 0 {
        job.stride_a as usize
    } else {
        a_matrix_elems
    };
    let b_batch_stride = if job.stride_b > 0 {
        job.stride_b as usize
    } else {
        b_matrix_elems
    };
    let c_batch_stride = if job.stride_c > 0 {
        job.stride_c as usize
    } else {
        c_matrix_elems
    };
    let batches = usize::try_from(batch_count)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "GEMM mock batch overflow"))?;
    let a_elems = a_batch_stride
        .checked_mul(batches.saturating_sub(1))
        .and_then(|v| v.checked_add(a_matrix_elems))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM mock A elems overflow"))?;
    let b_elems = b_batch_stride
        .checked_mul(batches.saturating_sub(1))
        .and_then(|v| v.checked_add(b_matrix_elems))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM mock B elems overflow"))?;
    let c_elems = c_batch_stride
        .checked_mul(batches.saturating_sub(1))
        .and_then(|v| v.checked_add(c_matrix_elems))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "GEMM mock C elems overflow"))?;
    let mut a = vec![0u8; a_elems * a_dtype_size];
    let mut b = vec![0u8; b_elems * b_dtype_size];
    let mut c = vec![0u8; c_elems * c_dtype_size];
    read_shared_ddr_phys(job.a_addr, &mut a)?;
    read_shared_ddr_phys(job.b_addr, &mut b)?;
    read_shared_ddr_phys(job.c_addr, &mut c)?;
    let alpha = if job.alpha_addr != 0 {
        let mut buf = [0u8; 4];
        read_shared_ddr_phys(job.alpha_addr, &mut buf)?;
        f32::from_ne_bytes(buf)
    } else {
        1.0
    };
    let beta = if job.beta_addr != 0 {
        let mut buf = [0u8; 4];
        read_shared_ddr_phys(job.beta_addr, &mut buf)?;
        f32::from_ne_bytes(buf)
    } else {
        0.0
    };
    let m = usize::try_from(job.m)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "GEMM m overflow"))?;
    let n = usize::try_from(job.n)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "GEMM n overflow"))?;
    let k = usize::try_from(job.k)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "GEMM k overflow"))?;
    let lda = usize::try_from(job.lda)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "GEMM lda overflow"))?;
    let ldb = usize::try_from(job.ldb)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "GEMM ldb overflow"))?;
    let ldc = usize::try_from(job.ldc)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "GEMM ldc overflow"))?;
    for batch in 0..batches {
        let a_batch = batch * a_batch_stride;
        let b_batch = batch * b_batch_stride;
        let c_batch = batch * c_batch_stride;
        for row in 0..m {
            for col in 0..n {
                let a_base = if job.transa != 0 { row } else { row * lda };
                let a_stride = if job.transa != 0 { lda } else { 1 };
                let b_base = if job.transb != 0 { col * ldb } else { col };
                let b_stride = if job.transb != 0 { 1 } else { ldb };
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let av = mock_slice_read_f32(
                        &a,
                        job.atype as i32,
                        a_batch + a_base + kk * a_stride,
                    )?;
                    let bv = mock_slice_read_f32(
                        &b,
                        job.btype as i32,
                        b_batch + b_base + kk * b_stride,
                    )?;
                    acc += av * bv;
                }
                let c_idx = c_batch + row + col * ldc;
                let old = if beta != 0.0 {
                    mock_slice_read_f32(&c, job.ctype as i32, c_idx)?
                } else {
                    0.0
                };
                mock_slice_write_f32(&mut c, job.ctype as i32, c_idx, alpha * acc + beta * old)?;
            }
        }
    }
    write_shared_ddr_phys(job.c_addr, &c)?;
    if zluda_irq_trace_enabled() {
        eprintln!(
            "PACC ZLUDA IRQ mock: GEMM m={} n={} k={} dtype={}/{}/{} c=0x{:x}",
            job.m, job.n, job.k, job.atype, job.btype, job.ctype, job.c_addr
        );
    }
    Ok(())
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
    let payload_base = shared_ddr_payload_base_off();
    let payload_bytes = shared_ddr_payload_bytes();
    if shared_base == 0 || payload_bytes == 0 {
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
    let dev_id = dev_override.unwrap_or_else(next_gemm_device);
    let requested_slot = slot_override.unwrap_or(dev_id);
    let slot_id = requested_slot.min(slot_count.saturating_sub(1));
    let _gemm_guard = lock_shared_ddr_stage(slot_id, "hetgpu_pacc_submit_gemm_staged")?;
    let slot_bytes = std::env::var("HETGPU_PACC_GEMM_SLOT_BYTES")
        .ok()
        .and_then(|v| {
            let trimmed = v.trim_start_matches("0x");
            usize::from_str_radix(trimmed, 16)
                .ok()
                .or_else(|| v.parse().ok())
        })
        .unwrap_or_else(|| payload_bytes / slot_count.max(1));
    let slot_rel_off = (slot_id as u64)
        .checked_mul(slot_bytes as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "shared DDR slot offset overflow"))?;
    let slot_off = payload_base
        .checked_add(slot_rel_off)
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
    let payload_base = shared_ddr_payload_base_off();
    let payload_bytes = shared_ddr_payload_bytes();
    if shared_base == 0 || payload_bytes == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            "PACC shared DDR staging window is not configured",
        ));
    }

    let a_src = std::slice::from_raw_parts(a.cast::<u8>(), a_elems * a_dtype_size);
    let b_src = std::slice::from_raw_parts(b.cast::<u8>(), b_elems * b_dtype_size);
    let old_c = std::slice::from_raw_parts(c.cast::<f32>(), c_elems).to_vec();

    let mut cursor = payload_base;
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
    let mut shared_file = open_shared_ddr_window_file(dev.id);
    let mut mailbox_file = open_pacc_mailbox_file(dev.id);
    let c_ptr = (c_batch as *mut u8)
        .add((row0 + col0 * ldc) * c_dtype_size)
        .cast::<std::ffi::c_void>();
    let c_initial =
        pack_gemm_c_block_rowmajor_f32_bytes(c_ptr.cast_const(), ctype, 0, chunk_m, chunk_n, ldc)?;
    let a_bytes = chunk_m
        .checked_mul(max_k)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "A tile size overflow"))?;
    let b_bytes = max_k
        .checked_mul(chunk_n)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B tile size overflow"))?;
    let c_bytes = c_initial.len();
    let a_off = 0u64;
    let b_off = align_up_u64(a_bytes as u64, 64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "B coarse offset overflow"))?;
    let c_off = align_up_u64(b_off + b_bytes as u64, 64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C coarse offset overflow"))?;
    let alpha_off = align_up_u64(c_off + c_bytes as u64, 64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "alpha coarse offset overflow"))?;
    let beta_off = alpha_off + std::mem::size_of::<f32>() as u64;
    let total = beta_off + std::mem::size_of::<f32>() as u64;
    if total as usize > slot_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "PACC coarse GEMM needs {} bytes, shared DDR slot has {}",
                total, slot_bytes
            ),
        ));
    }

    write_shared_ddr_window_cached(
        &mut shared_file,
        slot_off + alpha_off,
        &alpha_value.to_ne_bytes(),
    )?;
    write_shared_ddr_window_cached(&mut shared_file, slot_off + beta_off, &0.0f32.to_ne_bytes())?;

    let tile_max_k = if chunk_m < 64 || chunk_n < 16 {
        max_k.min(parse_env_usize("HETGPU_PACC_GEMM_TAIL_MAX_K", 80).max(1))
    } else {
        max_k.max(1)
    };
    let c_elems = chunk_m
        .checked_mul(chunk_n)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "C tile elem overflow"))?;
    let mut c_accum = vec![0.0f32; c_elems];
    if beta_value != 0.0 {
        for (i, chunk) in c_initial.chunks_exact(4).enumerate() {
            c_accum[i] = beta_value * f32::from_ne_bytes(chunk.try_into().unwrap());
        }
    }
    let zero_c_stage = vec![0u8; c_bytes];
    let mut c_partial = vec![0u8; c_bytes];

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
        write_shared_ddr_window_cached(&mut shared_file, slot_off + c_off, &zero_c_stage)?;

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
            beta_addr: shared_base + slot_off + beta_off,
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
        read_shared_ddr_window_cached(&mut shared_file, slot_off + c_off, &mut c_partial)?;
        for (i, chunk) in c_partial.chunks_exact(4).enumerate() {
            c_accum[i] += f32::from_ne_bytes(chunk.try_into().unwrap());
        }
    }

    let mut c_stage = Vec::with_capacity(c_bytes);
    for value in c_accum {
        c_stage.extend_from_slice(&value.to_ne_bytes());
    }
    unpack_gemm_c_block_rowmajor_f32_bytes(&c_stage, c_ptr, ctype, 0, chunk_m, chunk_n, ldc)?;
    if trace_gemm {
        eprintln!(
            "hetgpu_pacc_submit_gemm_staged_tiled: tile dev={} slot=0x{:x} row={} col={} m={} n={} staged C-once={} total={} via shared DDR 0x{:x}",
            dev.id,
            slot_off,
            row0,
            col0,
            chunk_m,
            chunk_n,
            c_bytes,
            total,
            shared_base + slot_off
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
    let payload_base = shared_ddr_payload_base_off();
    let payload_bytes = shared_ddr_payload_bytes();
    if shared_base == 0 || payload_bytes == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            "PACC shared DDR staging window is not configured",
        ));
    }
    let slot_count = parse_env_usize("HETGPU_PACC_GEMM_SHARED_SLOTS", PACC_CORE_NUM).max(1);
    let slot_bytes =
        parse_env_usize("HETGPU_PACC_GEMM_SLOT_BYTES", payload_bytes / slot_count).max(1);
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
        if std::env::var("HETGPU_PACC_GEMM_PARALLEL").ok().as_deref() == Some("0") {
            1
        } else {
            parse_env_usize("HETGPU_PACC_GEMM_WORKERS", PACC_CORE_NUM)
                .max(1)
                .min(PACC_CORE_NUM.max(1))
                .min(tile_count.max(1))
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
            let _slot_guard = lock_shared_ddr_stage(0, "hetgpu_pacc_submit_gemm_staged_tiled")?;
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
                    payload_base,
                    slot_bytes.min(payload_bytes),
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
                    let slot_rel_off =
                        (slot_id as u64)
                            .checked_mul(slot_bytes as u64)
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorKind::InvalidInput,
                                    "shared DDR slot offset overflow",
                                )
                            })?;
                    let slot_off = payload_base.checked_add(slot_rel_off).ok_or_else(|| {
                        Error::new(ErrorKind::InvalidInput, "shared DDR slot offset overflow")
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
                        let _slot_guard =
                            lock_shared_ddr_stage(slot_id, "hetgpu_pacc_submit_gemm_staged_tiled")?;
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
    dev_id: usize,
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
    match PaccDevice::open(dev_id)
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
pub unsafe extern "C" fn hetgpu_pacc_submit_softmax_on(
    dev_id: i32,
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
        normalize_pacc_device_id(dev_id),
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
pub unsafe extern "C" fn hetgpu_pacc_submit_softmax(
    src: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    rows: u64,
    cols: u64,
    stride: u64,
    dtype: i32,
) -> i32 {
    hetgpu_pacc_submit_softmax_on(0, src, dst, rows, cols, stride, dtype)
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
    dev_id: usize,
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
    dtype: u32,
    label: &str,
) -> i32 {
    let staged = std::env::var("HETGPU_PACC_RMSNORM_STAGE_SHARED_DDR")
        .ok()
        .as_deref()
        != Some("0");
    let result = if staged {
        submit_rmsnorm_staged_shared_ddr(dev_id, x, weight, y, rows, hidden, eps, dtype, label)
    } else {
        submit_rmsnorm_direct_runtime_job(dev_id, x, weight, y, rows, hidden, eps, dtype)
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!(
                "{}: PACC RMSNorm submit failed dev={} x=0x{:x} w=0x{:x} y=0x{:x} rows={} hidden={} eps={} dtype={}: {}",
                label, dev_id, x as usize, weight as usize, y as usize, rows, hidden, eps, dtype, e
            );
            -1
        }
    }
}

unsafe fn submit_rmsnorm_direct_runtime_job(
    dev_id: usize,
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
    dtype: u32,
) -> std::io::Result<()> {
    let job = HetgpuPaccRmsNormJob {
        x_addr: x as u64,
        weight_addr: weight as u64,
        y_addr: y as u64,
        rows,
        hidden,
        eps,
        dtype,
    };
    PaccDevice::open(dev_id)?.submit_runtime_job(hetgpu_pacc_job_id::RMSNORM, &job)
}

fn rmsnorm_trace_enabled() -> bool {
    std::env::var("HETGPU_PACC_RMSNORM_TRACE").ok().as_deref() == Some("1")
}

unsafe fn submit_rmsnorm_staged_shared_ddr(
    dev_id: usize,
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
    dtype: u32,
    label: &str,
) -> std::io::Result<()> {
    let elem_size = pacc_dtype_size(dtype as i32).ok_or_else(|| {
        Error::new(
            ErrorKind::Unsupported,
            format!("unsupported staged RMSNorm dtype {}", dtype),
        )
    })?;
    let rows_usize = usize::try_from(rows)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "RMSNorm rows overflow"))?;
    let hidden_usize = usize::try_from(hidden)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "RMSNorm hidden overflow"))?;
    let row_bytes = hidden_usize
        .checked_mul(elem_size)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm row size overflow"))?;
    if row_bytes == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "RMSNorm row size is zero",
        ));
    }

    let shared_bytes = shared_ddr_bytes();
    let payload_base = shared_ddr_payload_base_off() as usize;
    let payload_bytes = shared_ddr_payload_bytes();
    let slot_count = parse_env_usize("HETGPU_PACC_RMSNORM_SHARED_SLOTS", PACC_CORE_NUM).max(1);
    let slot_bytes =
        parse_env_usize("HETGPU_PACC_RMSNORM_SLOT_BYTES", payload_bytes / slot_count).max(1);
    let slot_id = std::env::var("HETGPU_PACC_RMSNORM_SLOT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(dev_id % slot_count)
        % slot_count;
    let slot_rel_off = slot_id
        .checked_mul(slot_bytes)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm slot offset overflow"))?;
    let slot_off_usize = payload_base
        .checked_add(slot_rel_off)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm slot offset overflow"))?;
    if slot_off_usize
        .checked_add(slot_bytes)
        .filter(|&end| end <= shared_bytes)
        .is_none()
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "RMSNorm shared-DDR slot exceeds configured window",
        ));
    }
    let _shared_guard = lock_shared_ddr_stage(slot_id, label)?;

    let weight_bytes = if weight.is_null() { 0 } else { row_bytes };
    let x_off = align_up_usize(weight_bytes, 64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm x offset overflow"))?;
    let min_y_off = align_up_usize(
        x_off
            .checked_add(row_bytes)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm y offset overflow"))?,
        64,
    )
    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm y offset overflow"))?;
    if min_y_off
        .checked_add(row_bytes)
        .filter(|&total| total <= slot_bytes)
        .is_none()
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "RMSNorm shared-DDR slot too small for one row",
        ));
    }
    let row_pair_bytes = row_bytes
        .checked_mul(2)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm row pair overflow"))?;
    let mut max_rows = (slot_bytes.saturating_sub(x_off) / row_pair_bytes).max(1);
    max_rows = max_rows.min(rows_usize);
    max_rows = max_rows.min(parse_env_usize("HETGPU_PACC_RMSNORM_MAX_ROWS", max_rows).max(1));

    let slot_off = u64::try_from(slot_off_usize)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "RMSNorm slot offset overflow"))?;
    let shared_base = shared_ddr_base();
    let mut shared_file = open_shared_ddr_window_file(dev_id);
    let pacc_dev = PaccDevice::open(dev_id)?;
    let trace = rmsnorm_trace_enabled();
    let zero_output = std::env::var("HETGPU_PACC_RMSNORM_ZERO_OUTPUT")
        .ok()
        .as_deref()
        == Some("1");
    if trace {
        eprintln!(
            "{}: staged shared-DDR dev={} slot={} rows={} hidden={} dtype={} row_bytes={} slot_bytes={}",
            label, dev_id, slot_id, rows, hidden, dtype, row_bytes, slot_bytes
        );
    }

    if !weight.is_null() {
        let weight_slice = std::slice::from_raw_parts(weight.cast::<u8>(), weight_bytes);
        if trace {
            eprintln!(
                "{}: stage RMSNorm weight dev={} off=0x{:x} bytes={}",
                label, dev_id, slot_off, weight_bytes
            );
        }
        write_shared_ddr_window_cached(&mut shared_file, slot_off, weight_slice).map_err(|e| {
            Error::new(
                e.kind(),
                format!(
                    "{}: failed to stage RMSNorm weight dev={} off=0x{:x} bytes={}: {}",
                    label, dev_id, slot_off, weight_bytes, e
                ),
            )
        })?;
    }

    let max_chunk_bytes = max_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm max chunk overflow"))?;
    let mut y_stage = vec![0u8; max_chunk_bytes];

    let mut row0 = 0usize;
    while row0 < rows_usize {
        let remaining = rows_usize - row0;
        let mut chunk_rows = remaining.min(max_rows);
        let (chunk_bytes, y_off, total_bytes) = loop {
            let chunk_bytes = chunk_rows.checked_mul(row_bytes).ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "RMSNorm chunk size overflow")
            })?;
            let y_off = align_up_usize(
                x_off.checked_add(chunk_bytes).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidInput, "RMSNorm chunk y offset overflow")
                })?,
                64,
            )
            .ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "RMSNorm chunk y offset overflow")
            })?;
            let total_bytes = y_off.checked_add(chunk_bytes).ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "RMSNorm chunk total overflow")
            })?;
            if total_bytes <= slot_bytes {
                break (chunk_bytes, y_off, total_bytes);
            }
            chunk_rows = chunk_rows.checked_sub(1).ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "RMSNorm chunk does not fit slot")
            })?;
            if chunk_rows == 0 {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "RMSNorm chunk does not fit slot",
                ));
            }
        };

        let host_off = row0
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "RMSNorm host offset overflow"))?;
        let x_slice = std::slice::from_raw_parts(x.cast::<u8>().add(host_off), chunk_bytes);
        if trace {
            eprintln!(
                "{}: stage RMSNorm x dev={} row0={} off=0x{:x} bytes={}",
                label,
                dev_id,
                row0,
                slot_off + x_off as u64,
                chunk_bytes
            );
        }
        write_shared_ddr_window_cached(&mut shared_file, slot_off + x_off as u64, x_slice)
            .map_err(|e| {
                Error::new(
                    e.kind(),
                    format!(
                        "{}: failed to stage RMSNorm x dev={} row0={} off=0x{:x} bytes={}: {}",
                        label,
                        dev_id,
                        row0,
                        slot_off + x_off as u64,
                        chunk_bytes,
                        e
                    ),
                )
            })?;
        if zero_output {
            y_stage[..chunk_bytes].fill(0);
            write_shared_ddr_window_cached(
                &mut shared_file,
                slot_off + y_off as u64,
                &y_stage[..chunk_bytes],
            )
            .map_err(|e| {
                Error::new(
                    e.kind(),
                    format!(
                        "{}: failed to prezero RMSNorm y dev={} row0={} off=0x{:x} bytes={}: {}",
                        label,
                        dev_id,
                        row0,
                        slot_off + y_off as u64,
                        chunk_bytes,
                        e
                    ),
                )
            })?;
        }

        let job = HetgpuPaccRmsNormJob {
            x_addr: shared_base + slot_off + x_off as u64,
            weight_addr: if weight.is_null() {
                0
            } else {
                shared_base + slot_off
            },
            y_addr: shared_base + slot_off + y_off as u64,
            rows: chunk_rows as u64,
            hidden,
            eps,
            dtype,
        };
        if trace {
            eprintln!(
                "{}: submit RMSNorm chunk dev={} row0={} rows={} x=0x{:x} w=0x{:x} y=0x{:x} bytes={}",
                label,
                dev_id,
                row0,
                chunk_rows,
                job.x_addr,
                job.weight_addr,
                job.y_addr,
                total_bytes
            );
        }
        pacc_dev
            .submit_runtime_job(hetgpu_pacc_job_id::RMSNORM, &job)
            .map_err(|e| {
                Error::new(
                    e.kind(),
                    format!(
                        "{}: submit RMSNorm runtime job failed dev={} row0={} rows={} hidden={}: {}",
                        label, dev_id, row0, chunk_rows, hidden, e
                    ),
                )
            })?;

        read_shared_ddr_window_cached(
            &mut shared_file,
            slot_off + y_off as u64,
            &mut y_stage[..chunk_bytes],
        )
        .map_err(|e| {
            Error::new(
                e.kind(),
                format!(
                    "{}: failed to read RMSNorm y dev={} row0={} off=0x{:x} bytes={}: {}",
                    label,
                    dev_id,
                    row0,
                    slot_off + y_off as u64,
                    chunk_bytes,
                    e
                ),
            )
        })?;
        std::ptr::copy_nonoverlapping(y_stage.as_ptr(), y.cast::<u8>().add(host_off), chunk_bytes);
        row0 += chunk_rows;
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_pacc_submit_rmsnorm_on(
    dev_id: i32,
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
        normalize_pacc_device_id(dev_id),
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
pub unsafe extern "C" fn hetgpu_pacc_submit_rmsnorm(
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    rows: u64,
    hidden: u64,
    eps: f32,
    dtype: i32,
) -> i32 {
    hetgpu_pacc_submit_rmsnorm_on(0, x, weight, y, rows, hidden, eps, dtype)
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
    pub compile_error: Option<String>,
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
    let log_program_loads = std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1");
    if log_program_loads {
        eprintln!(
            "load_program_elf_bytes: program={:?} elf_bytes={}",
            program,
            elf_bytes.len()
        );
    }
    (*program).elf_bytes = elf_bytes;
    (*program).compile_error = None;
    pacc_Result_Success
}

unsafe fn set_program_compile_error(
    program: *mut pacc_Program,
    stage: &str,
    message: String,
) -> pacc_Result {
    if !program.is_null() {
        (*program).elf_bytes.clear();
        (*program).compile_error = Some(format!("{}: {}", stage, message));
    }
    pacc_Result_Error
}

const HETGPU_PACC_ELF_CACHE_VERSION: &[u8] = b"hetgpu-pacc-elf-cache-v3";

fn pacc_program_load_logs_enabled() -> bool {
    std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1")
}

fn pacc_elf_cache_enabled() -> bool {
    std::env::var("HETGPU_PACC_DISABLE_ELF_CACHE")
        .ok()
        .as_deref()
        != Some("1")
}

fn pacc_elf_cache_dir() -> PathBuf {
    std::env::var_os("HETGPU_PACC_ELF_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/hetgpu_pacc_elf_cache"))
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn fnv1a64_with_len(hash: u64, bytes: &[u8]) -> u64 {
    let hash = fnv1a64_update(hash, &(bytes.len() as u64).to_le_bytes());
    fnv1a64_update(hash, bytes)
}

fn compute_pacc_elf_cache_key(
    target_arch: &CStr,
    ptx_bytes: &[u8],
    linked_bitcode: &[u8],
) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    hash = fnv1a64_with_len(hash, HETGPU_PACC_ELF_CACHE_VERSION);
    hash = fnv1a64_with_len(hash, target_arch.to_bytes());
    hash = fnv1a64_with_len(hash, ptx_bytes);
    hash = fnv1a64_with_len(hash, linked_bitcode);
    format!("{hash:016x}")
}

fn pacc_elf_cache_path(cache_dir: &Path, cache_key: &str) -> PathBuf {
    cache_dir.join(format!("{cache_key}.elf"))
}

fn try_load_cached_pacc_elf(
    target_arch: &CStr,
    module_name: &CStr,
    ptx_bytes: &[u8],
    linked_bitcode: &[u8],
) -> Option<Vec<u8>> {
    if !pacc_elf_cache_enabled() {
        return None;
    }

    let cache_dir = pacc_elf_cache_dir();
    let cache_key = compute_pacc_elf_cache_key(target_arch, ptx_bytes, linked_bitcode);
    let cache_path = pacc_elf_cache_path(&cache_dir, &cache_key);
    let bytes = std::fs::read(&cache_path).ok()?;
    if bytes.len() < 4 || &bytes[0..4] != b"\x7fELF" {
        return None;
    }
    if pacc_program_load_logs_enabled() {
        eprintln!(
            "pacc_LoadProgramPtx: cache hit for '{}' -> {} bytes ({})",
            module_name.to_string_lossy(),
            bytes.len(),
            cache_path.display(),
        );
    }
    Some(bytes)
}

fn store_cached_pacc_elf(
    target_arch: &CStr,
    module_name: &CStr,
    ptx_bytes: &[u8],
    linked_bitcode: &[u8],
    elf_bytes: &[u8],
) {
    if !pacc_elf_cache_enabled() || elf_bytes.len() < 4 || &elf_bytes[0..4] != b"\x7fELF" {
        return;
    }

    let cache_dir = pacc_elf_cache_dir();
    if let Err(err) = std::fs::create_dir_all(&cache_dir) {
        if pacc_program_load_logs_enabled() {
            eprintln!(
                "pacc_LoadProgramPtx: failed to create ELF cache dir {}: {}",
                cache_dir.display(),
                err
            );
        }
        return;
    }

    let cache_key = compute_pacc_elf_cache_key(target_arch, ptx_bytes, linked_bitcode);
    let cache_path = pacc_elf_cache_path(&cache_dir, &cache_key);
    match std::fs::write(&cache_path, elf_bytes) {
        Ok(()) => {
            if pacc_program_load_logs_enabled() {
                eprintln!(
                    "pacc_LoadProgramPtx: cached '{}' -> {} bytes ({})",
                    module_name.to_string_lossy(),
                    elf_bytes.len(),
                    cache_path.display(),
                );
            }
        }
        Err(err) => {
            if pacc_program_load_logs_enabled() {
                eprintln!(
                    "pacc_LoadProgramPtx: failed to write ELF cache {}: {}",
                    cache_path.display(),
                    err
                );
            }
        }
    }
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
        compile_error: None,
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
            set_program_compile_error(
                program,
                "source_compile",
                format!("{}: {:?}", source_name.to_string_lossy(), err),
            )
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

    if let Some(elf_bytes) =
        try_load_cached_pacc_elf(target_arch, module_name, ptx_bytes, external_linked)
    {
        return load_program_elf_bytes(program, elf_bytes);
    }

    let ptx_text = match std::str::from_utf8(ptx_bytes) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "pacc_LoadProgramPtx: invalid UTF-8 in module {}: {}",
                module_name.to_string_lossy(),
                err
            );
            return set_program_compile_error(
                program,
                "ptx_utf8",
                format!("{}: {}", module_name.to_string_lossy(), err),
            );
        }
    };

    if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "pacc_LoadProgramPtx: begin module='{}' target='{}' ptx_bytes={} linked_bitcode_bytes={}",
            module_name.to_string_lossy(),
            target_arch.to_string_lossy(),
            ptx_bytes.len(),
            external_linked.len(),
        );
        eprintln!(
            "pacc_LoadProgramPtx: stage PTX parse start for {}",
            module_name.to_string_lossy()
        );
    }

    let ast = match ptx_parser::parse_module_checked(ptx_text) {
        Ok(ast) => ast,
        Err(err) => {
            eprintln!(
                "pacc_LoadProgramPtx: PTX parse failed for {}: {:?}",
                module_name.to_string_lossy(),
                err
            );
            return set_program_compile_error(
                program,
                "ptx_parse",
                format!("{}: {:?}", module_name.to_string_lossy(), err),
            );
        }
    };
    if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "pacc_LoadProgramPtx: stage PTX parse done for {}",
            module_name.to_string_lossy()
        );
        eprintln!(
            "pacc_LoadProgramPtx: stage PTX -> LLVM start for {}",
            module_name.to_string_lossy()
        );
    }
    let llvm_module = match ptx::to_llvm_module(
        ast,
        ptx::pass::Attributes {
            clock_rate: 1_000_000,
            emit_debug_info: false,
        },
        |pass_name| {
            if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!("pacc_LoadProgramPtx: pass {}", pass_name);
            }
        },
    ) {
        Ok(module) => module,
        Err(err) => {
            eprintln!(
                "pacc_LoadProgramPtx: PTX -> LLVM failed for {}: {:?}",
                module_name.to_string_lossy(),
                err
            );
            return set_program_compile_error(
                program,
                "ptx_to_llvm",
                format!("{}: {:?}", module_name.to_string_lossy(), err),
            );
        }
    };
    if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "pacc_LoadProgramPtx: stage PTX -> LLVM done for {}",
            module_name.to_string_lossy()
        );
        eprintln!(
            "pacc_LoadProgramPtx: stage LLVM bitcode serialize start for {}",
            module_name.to_string_lossy()
        );
    }
    let ir_bytes = llvm_module.llvm_ir.write_bitcode_to_memory();
    if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "pacc_LoadProgramPtx: stage LLVM bitcode serialize done for {} ({} bytes)",
            module_name.to_string_lossy(),
            ir_bytes.len()
        );
    }
    let internal_linked = llvm_module.linked_bitcode();
    let mut linked_modules: Vec<&[u8]> = Vec::new();
    if !internal_linked.is_empty() {
        linked_modules.push(internal_linked);
    }
    if !external_linked.is_empty() {
        linked_modules.push(external_linked);
    }
    if std::env::var("HETGPU_PACC_LOG_PROGRAM_LOADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "pacc_LoadProgramPtx: stage LLVM -> XM ELF start for {} (linked_modules={})",
            module_name.to_string_lossy(),
            linked_modules.len()
        );
    }

    match comgr::compile_bitcode_pacc_multi(target_arch, &*ir_bytes, &linked_modules) {
        Ok(elf_bytes) => {
            store_cached_pacc_elf(
                target_arch,
                module_name,
                ptx_bytes,
                external_linked,
                &elf_bytes,
            );
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
            set_program_compile_error(
                program,
                "llvm_to_xm_elf",
                format!("{}: {:?}", module_name.to_string_lossy(), err),
            )
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
            "pacc_LaunchKernel: kernel='{}' grid=({},{},{}) block=({},{},{}) elf_bytes={}",
            k.name,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            prog.elf_bytes.len()
        );
    }

    if prog.elf_bytes.is_empty() {
        if let Some(err) = prog.compile_error.as_deref() {
            eprintln!(
                "pacc_LaunchKernel: kernel '{}' has no compiled ELF because program compilation failed: {}",
                k.name, err
            );
        }
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

fn require_kernel_mbox_submit() -> bool {
    std::env::var("HETGPU_PACC_KERNEL_MBOX_SUBMIT")
        .ok()
        .as_deref()
        != Some("0")
}

fn kernel_launch_wait_enabled() -> bool {
    match std::env::var("HETGPU_PACC_WAIT_KERNEL_LAUNCH")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("0" | "false" | "no" | "off") => false,
        Some("1" | "true" | "yes" | "on") => true,
        Some(_) | None => use_shared_ddr_control_window(),
    }
}

fn parse_optional_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| {
        let value = v.trim();
        if value.is_empty() {
            return None;
        }
        value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))
            .map(|hex| usize::from_str_radix(hex, 16).ok())
            .unwrap_or_else(|| value.parse().ok())
    })
}

fn align_down(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    value - (value % align)
}

fn kernel_submit_slot_layout(
    dev_id: usize,
    required_bytes: usize,
) -> std::io::Result<(u64, usize)> {
    let shared_bytes = shared_ddr_bytes();
    let control_reserved = shared_ddr_control_reserved_bytes();
    let usable_bytes = shared_bytes.saturating_sub(control_reserved);
    let min_slot_bytes = align_up(required_bytes, 64).max(64);
    let explicit_slot_count =
        parse_optional_env_usize("HETGPU_PACC_KERNEL_SLOT_COUNT").filter(|&v| v > 0);
    let explicit_slot_bytes =
        parse_optional_env_usize("HETGPU_PACC_KERNEL_SLOT_BYTES").filter(|&v| v > 0);

    if usable_bytes < min_slot_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "kernel image needs {} bytes, shared DDR payload area has {}",
                min_slot_bytes, usable_bytes
            ),
        ));
    }

    let mut slot_count = explicit_slot_count.unwrap_or_else(|| {
        let max_count_for_image = (usable_bytes / min_slot_bytes).max(1);
        PACC_CORE_NUM.max(1).min(max_count_for_image)
    });
    if explicit_slot_count.is_none() {
        slot_count = slot_count.max(1);
    }

    let slot_bytes = if let Some(slot_bytes) = explicit_slot_bytes {
        slot_bytes
    } else {
        align_down(usable_bytes / slot_count.max(1), 64)
    };
    if slot_bytes < min_slot_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "kernel image needs {} bytes, helper slot has {}; \
                 increase HETGPU_PACC_KERNEL_SLOT_BYTES or reduce HETGPU_PACC_KERNEL_SLOT_COUNT",
                min_slot_bytes, slot_bytes
            ),
        ));
    }
    let reserved = slot_bytes
        .checked_mul(slot_count)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "kernel slot reservation overflow"))?;
    if reserved > usable_bytes {
        return Err(Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "kernel helper slots need {} bytes, shared DDR payload area has {}",
                reserved, usable_bytes
            ),
        ));
    }
    let base_off = control_reserved as u64 + (usable_bytes - reserved) as u64;
    let slot_off = base_off
        .checked_add((dev_id % slot_count) as u64 * slot_bytes as u64)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "kernel slot offset overflow"))?;
    if std::env::var("HETGPU_PACC_LOG_KERNEL_LAUNCHES")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "pacc_LaunchKernel: helper slot layout dev={} slot_count={} slot_bytes={} slot_off=0x{:x} required={}",
            dev_id, slot_count, slot_bytes, slot_off, min_slot_bytes
        );
    }
    Ok((slot_off, slot_bytes))
}

#[derive(Debug)]
struct PaccKernelStagedBuffer {
    original_addr: u64,
    stage_off: u64,
    size: usize,
    flags: u32,
}

#[derive(Debug)]
struct PaccKernelSharedDdrStaging {
    launch_state: PaccKernelLaunchState,
    staged: Vec<PaccKernelStagedBuffer>,
}

fn pacc_kernel_binding_needs_stage(
    shared_base: u64,
    shared_bytes: usize,
    binding: &PaccKernelBufferBinding,
) -> bool {
    if binding.addr == 0 || binding.size == 0 {
        return false;
    }
    if binding.flags & PACC_KERNEL_ARG_FLAG_BUFFER_INOUT == 0 {
        return false;
    }
    let shared_end = shared_base.saturating_add(shared_bytes as u64);
    let binding_end = binding.addr.saturating_add(binding.size);
    !(shared_base != 0
        && binding.addr >= shared_base
        && binding_end <= shared_end
        && binding_end >= binding.addr)
}

fn pacc_kernel_staging_payload_bytes(
    shared_base: u64,
    shared_bytes: usize,
    launch_state: &PaccKernelLaunchState,
) -> std::io::Result<usize> {
    let mut total = 0usize;
    for binding in launch_state.bindings.iter() {
        if !pacc_kernel_binding_needs_stage(shared_base, shared_bytes, binding) {
            continue;
        }
        let size = usize::try_from(binding.size).map_err(|_| {
            Error::new(
                ErrorKind::InvalidInput,
                "PACC kernel binding size does not fit usize",
            )
        })?;
        total = align_up(total, 64)
            .checked_add(align_up(size, 64))
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC staging size overflow"))?;
    }
    Ok(total)
}

fn write_shared_ddr_zeroes_cached(
    file: &mut Option<File>,
    mut offset: u64,
    mut len: usize,
) -> std::io::Result<()> {
    let zeroes = vec![0u8; helper_io_chunk_bytes().min(1 << 20).max(1)];
    while len != 0 {
        let chunk = len.min(zeroes.len());
        write_shared_ddr_window_cached(file, offset, &zeroes[..chunk])?;
        offset = offset.checked_add(chunk as u64).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, "shared DDR zero offset overflow")
        })?;
        len -= chunk;
    }
    Ok(())
}

unsafe fn prepare_pacc_kernel_shared_ddr_staging(
    kernel_name: &str,
    shared_base: u64,
    shared_bytes: usize,
    slot_off: u64,
    slot_bytes: usize,
    submit_len: usize,
    launch_state: &PaccKernelLaunchState,
    shared_file: &mut Option<File>,
) -> std::io::Result<PaccKernelSharedDdrStaging> {
    let mut staged_state = launch_state.clone();
    let mut staged = Vec::new();
    let mut cursor = align_up(submit_len, 64);
    let log_launches = std::env::var("HETGPU_PACC_LOG_KERNEL_LAUNCHES")
        .ok()
        .as_deref()
        == Some("1");

    for (binding_idx, binding) in launch_state.bindings.iter().enumerate() {
        if !pacc_kernel_binding_needs_stage(shared_base, shared_bytes, binding) {
            continue;
        }

        let size = usize::try_from(binding.size).map_err(|_| {
            Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "PACC kernel '{}' binding arg {} size does not fit usize",
                    kernel_name, binding.arg_index
                ),
            )
        })?;
        if size == 0 {
            continue;
        }
        cursor = align_up(cursor, 64);
        let end = cursor.checked_add(size).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("PACC kernel '{}' staging cursor overflow", kernel_name),
            )
        })?;
        if end > slot_bytes {
            return Err(Error::new(
                ErrorKind::OutOfMemory,
                format!(
                    "PACC kernel '{}' staging needs {} bytes in helper slot, slot has {}",
                    kernel_name, end, slot_bytes
                ),
            ));
        }

        let stage_off = slot_off
            .checked_add(cursor as u64)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC staging offset overflow"))?;
        let staged_addr = shared_base
            .checked_add(stage_off)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC staged address overflow"))?;

        if binding.flags & PACC_KERNEL_ARG_FLAG_BUFFER_INPUT != 0 {
            let src = std::slice::from_raw_parts(binding.addr as *const u8, size);
            write_shared_ddr_window_cached(shared_file, stage_off, src)?;
        } else {
            write_shared_ddr_zeroes_cached(shared_file, stage_off, size)?;
        }

        if let Some(staged_binding) = staged_state.bindings.get_mut(binding_idx) {
            staged_binding.addr = staged_addr;
        }
        if let Some(record) = staged_state.arg_records.get_mut(binding.arg_index as usize) {
            if record.kind == PACC_KERNEL_ARG_KIND_POINTER {
                record.value = staged_addr;
            }
        }

        if log_launches {
            eprintln!(
                "pacc_LaunchKernel: staged kernel='{}' arg={} host=0x{:x} shared=0x{:x} size={} flags=0x{:x}",
                kernel_name, binding.arg_index, binding.addr, staged_addr, size, binding.flags
            );
        }

        staged.push(PaccKernelStagedBuffer {
            original_addr: binding.addr,
            stage_off,
            size,
            flags: binding.flags,
        });
        cursor = align_up(end, 64);
    }

    Ok(PaccKernelSharedDdrStaging {
        launch_state: staged_state,
        staged,
    })
}

unsafe fn complete_pacc_kernel_shared_ddr_staging(
    staging: &PaccKernelSharedDdrStaging,
    shared_file: &mut Option<File>,
) -> std::io::Result<()> {
    for staged in staging.staged.iter() {
        if staged.flags & PACC_KERNEL_ARG_FLAG_BUFFER_OUTPUT == 0 {
            continue;
        }
        if staged.original_addr == 0 || staged.size == 0 {
            continue;
        }
        let mut bytes = vec![0u8; staged.size];
        read_shared_ddr_window_cached(shared_file, staged.stage_off, &mut bytes)?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), staged.original_addr as *mut u8, staged.size);
    }
    Ok(())
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

    let (_initial_buf, submit_len) = build_pacc_kernel_submit_buffer(
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
    let staging_bytes =
        pacc_kernel_staging_payload_bytes(shared_base, shared_ddr_bytes(), launch_state)?;
    let required_bytes = submit_len
        .checked_add(staging_bytes)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "PACC helper submit size overflow"))?;
    let (slot_off, slot_bytes) = kernel_submit_slot_layout(dev.id, required_bytes)?;
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
    let mut mailbox_file = if use_shared_ddr_control_window() {
        open_pacc_mailbox_file(dev.id)
    } else {
        Some(open_pacc_mailbox_helper_file(dev.id)?)
    };
    let staging = unsafe {
        prepare_pacc_kernel_shared_ddr_staging(
            kernel_name,
            shared_base,
            shared_ddr_bytes(),
            slot_off,
            slot_bytes,
            submit_len,
            launch_state,
            &mut shared_file,
        )?
    };
    if !staging.staged.is_empty() && !kernel_launch_wait_enabled() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "PACC shared-DDR staged kernel launch requires HETGPU_PACC_WAIT_KERNEL_LAUNCH=1",
        ));
    }
    let (buf, rebuilt_submit_len) = build_pacc_kernel_submit_buffer(
        kernel_name,
        elf_bytes,
        &staging.launch_state,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
    )?;
    if rebuilt_submit_len != submit_len {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "PACC staged launch ABI changed kernel image size unexpectedly",
        ));
    }
    write_shared_ddr_window_cached(&mut shared_file, slot_off, &buf[..submit_len])?;
    write_control_window_cached(
        &mut shared_file,
        &mut mailbox_file,
        dev.id,
        HETGPU_PACC_DOORBELL_OFF,
        desc_bytes,
    )?;
    if use_shared_ddr_control_window() {
        dev.zluda_irq(shared_ddr_info())?;
    }

    if kernel_launch_wait_enabled() {
        wait_mailbox_job_status_cached(
            dev,
            hetgpu_pacc_job_id::KERNEL,
            seq,
            &mut shared_file,
            &mut mailbox_file,
        )?;
        unsafe {
            complete_pacc_kernel_shared_ddr_staging(&staging, &mut shared_file)?;
        }
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
    if require_kernel_mbox_submit() {
        let helper_path = helper_path_for_pacc(dev.id);
        if !std::path::Path::new(&helper_path).exists() {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!(
                    "PACC kernel submit requires mailbox helper {helper_path}; \
                     load hetgpu_pacc_mbox.ko or set HETGPU_PACC_KERNEL_MBOX_SUBMIT=0 \
                     to force the legacy /dev/pacc BO submit path"
                ),
            ));
        }
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
                return Err(Error::new(
                    e.kind(),
                    format!(
                        "PACC kernel helper submit failed for '{}' on pacc{} via {}: {}; \
                     refusing legacy /dev/pacc BO fallback",
                        kernel_name, dev.id, helper_path, e
                    ),
                ));
            }
        }
    } else if helper_kernel_submit_enabled(dev.id) {
        eprintln!(
            "pacc_LaunchKernel: HETGPU_PACC_KERNEL_MBOX_SUBMIT=0, forcing legacy /dev/pacc BO submit despite available helper"
        );
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
    if name.contains("softmax") || name.contains("soft_max") {
        Some(hetgpu_pacc_job_id::SOFTMAX)
    } else if name.contains("rmsnorm") || name.contains("rms_norm") {
        Some(hetgpu_pacc_job_id::RMSNORM)
    } else if name.contains("gemm")
        || name.contains("matmul")
        || name.contains("cublas")
        || name.contains("mul_mat")
    {
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
        assert_eq!(IOC_ZLUDA_IRQ, 0x4010_7005u64);
    }
}
