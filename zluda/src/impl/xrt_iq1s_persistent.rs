use super::iq1s_layer_abi::{
    iq1s_command_crc32, Iq1sCommand, Iq1sCompletion, IQ1S_ABI_VERSION, IQ1S_COMMAND_BYTES,
    IQ1S_COMPLETION_BYTES, IQ1S_COMPLETION_MAGIC, IQ1S_COMPLETION_STATUS_OK, IQ1S_FAULT_CODE_NONE,
    IQ1S_REGISTER_MAGIC, IQ1S_REG_ABI_MAGIC_OFFSET, IQ1S_REG_ABI_VERSION_OFFSET,
    IQ1S_REG_COMMAND_BASE_HI_OFFSET, IQ1S_REG_COMMAND_BASE_LO_OFFSET,
    IQ1S_REG_COMMAND_CAPACITY_OFFSET, IQ1S_REG_COMMAND_CONSUMER_OFFSET,
    IQ1S_REG_COMMAND_PRODUCER_OFFSET, IQ1S_REG_COMPLETION_BASE_HI_OFFSET,
    IQ1S_REG_COMPLETION_BASE_LO_OFFSET, IQ1S_REG_COMPLETION_CAPACITY_OFFSET,
    IQ1S_REG_COMPLETION_CONSUMER_OFFSET, IQ1S_REG_COMPLETION_PRODUCER_OFFSET,
    IQ1S_REG_CONTROL_OFFSET, IQ1S_REG_DOORBELL_OFFSET, IQ1S_REG_FAULT_CODE_OFFSET,
    IQ1S_REG_QUIESCENT_OFFSET, IQ1S_REG_SESSION_GENERATION_HI_OFFSET,
    IQ1S_REG_SESSION_GENERATION_LO_OFFSET, IQ1S_ROLE_DOWN, IQ1S_ROLE_GATE, IQ1S_ROLE_UP,
};
use super::iq1s_layer_trace::{
    validate_compiled_layer_phase, ActivationRange, CompiledLayerPhase, ExpandedIq1sCounts,
};
use super::iq1s_trace::QWEN_MODEL_CONTEXT_LIMIT;
use super::iq1s_weight_arena::{ARENA_ALIGNMENT, ARENA_BANK_COUNT, ARENA_SUPERBLOCK_BYTES};
use super::xrt_tmatmul::{Handle, XrtOps, Xuid, XRT_BO_SYNC_FROM_DEVICE, XRT_BO_SYNC_TO_DEVICE};
use sha2::{Digest, Sha256};
use std::ffi::CString;
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const CONTROL_START: u32 = 1;
const CONTROL_SHUTDOWN: u32 = 2;
const COMMAND_RING_DEFAULT_CAPACITY: u32 = 512;
const PROGRAM_BYTES: usize = 4 * 1024 * 1024;
const ACTIVATION_BYTES: usize = 256 * 1024 * 1024;
const OUTPUT_BYTES: usize = 256 * 1024 * 1024;
const TOKEN_MAP_BYTES: usize = 16 * 1024 * 1024;
const MAX_BACKOFF_US: u64 = 1_000;
const MEMORY_GROUPS: [u32; ARENA_BANK_COUNT] = [0, 3, 2, 1];
const IP_NAMES: [&str; ARENA_BANK_COUNT] = [
    "iq1s_layer_big:iq1s_layer_big_1",
    "iq1s_layer_big:iq1s_layer_big_2",
    "iq1s_layer_big:iq1s_layer_big_3",
    "iq1s_layer_small:iq1s_layer_small_1",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersistentIq1sConfig {
    pub(crate) xclbin: PathBuf,
    pub(crate) device_index: u32,
    pub(crate) command_capacity: u32,
    pub(crate) timeout_ms: u32,
}

impl PersistentIq1sConfig {
    pub(crate) fn checked(
        xclbin: PathBuf,
        device_index: u32,
        command_capacity: Option<u32>,
        timeout_ms: u32,
    ) -> Result<Self, PersistentError> {
        let command_capacity = command_capacity.unwrap_or(COMMAND_RING_DEFAULT_CAPACITY);
        if xclbin.as_os_str().is_empty()
            || command_capacity == 0
            || !command_capacity.is_power_of_two()
            || timeout_ms == 0
        {
            return Err(PersistentError::Config(
                "xclbin, power-of-two ring capacity, and timeout must be valid".to_string(),
            ));
        }
        Ok(Self {
            xclbin,
            device_index,
            command_capacity,
            timeout_ms,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArenaChunkSpec {
    pub(crate) bank: u8,
    pub(crate) logical_offset: u64,
    pub(crate) bytes: usize,
    pub(crate) sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersistentFault {
    pub(crate) cu: Option<usize>,
    pub(crate) operation: &'static str,
    pub(crate) detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PersistentError {
    Config(String),
    Xrt {
        operation: &'static str,
        code: i32,
    },
    NullHandle(&'static str),
    InvalidPhase(String),
    RingFull {
        cu: usize,
        capacity: u32,
    },
    Timeout {
        operation: &'static str,
        timeout_ms: u32,
    },
    Fault(PersistentFault),
    Poisoned(PersistentFault),
    Shutdown(String),
}

impl fmt::Display for PersistentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(formatter, "persistent IQ1_S configuration: {message}"),
            Self::Xrt { operation, code } => {
                write!(
                    formatter,
                    "persistent IQ1_S XRT {operation} failed with code {code}"
                )
            }
            Self::NullHandle(operation) => {
                write!(formatter, "persistent IQ1_S XRT {operation} returned null")
            }
            Self::InvalidPhase(message) => write!(formatter, "invalid IQ1_S phase: {message}"),
            Self::RingFull { cu, capacity } => {
                write!(
                    formatter,
                    "IQ1_S CU {cu} command ring capacity {capacity} is full"
                )
            }
            Self::Timeout {
                operation,
                timeout_ms,
            } => write!(
                formatter,
                "persistent IQ1_S {operation} timed out after {timeout_ms} ms"
            ),
            Self::Fault(fault) => write!(
                formatter,
                "persistent IQ1_S fault during {} on {:?}: {}",
                fault.operation, fault.cu, fault.detail
            ),
            Self::Poisoned(fault) => write!(
                formatter,
                "persistent IQ1_S pool is poisoned by {} on {:?}: {}",
                fault.operation, fault.cu, fault.detail
            ),
            Self::Shutdown(message) => write!(formatter, "persistent IQ1_S shutdown: {message}"),
        }
    }
}

impl std::error::Error for PersistentError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PersistentDmaCounters {
    pub(crate) command_ranges: u64,
    pub(crate) activation_ranges: u64,
    pub(crate) result_ranges: u64,
    pub(crate) program_ranges: u64,
    pub(crate) weight_ranges: u64,
    pub(crate) weight_bytes: u64,
}

impl PersistentDmaCounters {
    fn checked_delta(self, baseline: Self) -> Result<Self, PersistentError> {
        let subtract = |value: u64, before: u64| {
            value.checked_sub(before).ok_or_else(|| {
                PersistentError::Config("persistent DMA counters regressed".to_string())
            })
        };
        Ok(Self {
            command_ranges: subtract(self.command_ranges, baseline.command_ranges)?,
            activation_ranges: subtract(self.activation_ranges, baseline.activation_ranges)?,
            result_ranges: subtract(self.result_ranges, baseline.result_ranges)?,
            program_ranges: subtract(self.program_ranges, baseline.program_ranges)?,
            weight_ranges: subtract(self.weight_ranges, baseline.weight_ranges)?,
            weight_bytes: subtract(self.weight_bytes, baseline.weight_bytes)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompletedLayerPhase {
    pub(crate) transaction_id: u64,
    pub(crate) semantic_sha256: [u8; 32],
    pub(crate) completions: [Vec<Iq1sCompletion>; ARENA_BANK_COUNT],
    pub(crate) expanded: ExpandedIq1sCounts,
    pub(crate) dma: PersistentDmaCounters,
}

#[derive(Debug)]
struct ArenaChunk {
    logical_offset: u64,
    bytes: usize,
    bo: Handle,
    address: u64,
}

#[derive(Debug)]
struct PersistentCu {
    ip_index: u32,
    command_bo: Handle,
    completion_bo: Handle,
    program_bo: Handle,
    activation_bo: Handle,
    output_bo: Handle,
    token_map_bo: Handle,
    command_address: u64,
    completion_address: u64,
    activation_address: u64,
    output_address: u64,
    token_map_address: u64,
    arena: Vec<ArenaChunk>,
    command_shadow: Vec<u8>,
    completion_shadow: Vec<u8>,
    program_shadow: Vec<u8>,
    command_producer: u32,
    command_consumer: u32,
    completion_consumer: u32,
    cached_program_id: Option<u64>,
}

impl PersistentCu {
    fn runtime_bos(&self) -> [Handle; 6] {
        [
            self.command_bo,
            self.completion_bo,
            self.program_bo,
            self.activation_bo,
            self.output_bo,
            self.token_map_bo,
        ]
    }
}

pub(crate) struct PersistentIq1sPool<O: XrtOps> {
    ops: O,
    device: Handle,
    native_device: Handle,
    xclbin_uuid: Xuid,
    generation: u64,
    config: PersistentIq1sConfig,
    cus: [PersistentCu; ARENA_BANK_COUNT],
    poisoned: Option<PersistentFault>,
    measured: bool,
    measurement_baseline: PersistentDmaCounters,
    dma: PersistentDmaCounters,
    closed: bool,
}

fn checked_code(operation: &'static str, code: i32) -> Result<(), PersistentError> {
    if code == 0 {
        Ok(())
    } else {
        Err(PersistentError::Xrt { operation, code })
    }
}

fn checked_handle(operation: &'static str, handle: Handle) -> Result<Handle, PersistentError> {
    if handle.is_null() {
        Err(PersistentError::NullHandle(operation))
    } else {
        Ok(handle)
    }
}

fn split_u64(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

fn struct_bytes<T>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

fn struct_from_bytes<T: Copy>(bytes: &[u8]) -> Result<T, PersistentError> {
    if bytes.len() != std::mem::size_of::<T>() {
        return Err(PersistentError::InvalidPhase(
            "record byte length does not match ABI".to_string(),
        ));
    }
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), value.as_mut_ptr().cast::<u8>(), bytes.len());
        Ok(value.assume_init())
    }
}

fn command_crc(command: &Iq1sCommand) -> u32 {
    let mut bytes = [0u8; IQ1S_COMMAND_BYTES];
    bytes.copy_from_slice(struct_bytes(command));
    iq1s_command_crc32(&bytes)
}

fn merge_ranges(ranges: impl IntoIterator<Item = (usize, usize)>) -> Vec<(usize, usize)> {
    let mut ranges = ranges
        .into_iter()
        .filter(|(_, bytes)| *bytes != 0)
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|(offset, _)| *offset);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (offset, bytes) in ranges {
        let Some(end) = offset.checked_add(bytes) else {
            return Vec::new();
        };
        if let Some((last_offset, last_bytes)) = merged.last_mut() {
            let last_end = *last_offset + *last_bytes;
            if offset <= last_end {
                *last_bytes = last_end.max(end) - *last_offset;
                continue;
            }
        }
        merged.push((offset, bytes));
    }
    merged
}

impl<O: XrtOps> PersistentIq1sPool<O> {
    pub(crate) fn open(
        ops: O,
        config: PersistentIq1sConfig,
        generation: u64,
        chunks: &[ArenaChunkSpec],
        mut read_chunk: impl FnMut(&ArenaChunkSpec) -> Result<Vec<u8>, String>,
    ) -> Result<Self, PersistentError> {
        if generation == 0 {
            return Err(PersistentError::Config(
                "session generation must be nonzero".to_string(),
            ));
        }
        let mut by_bank: [Vec<ArenaChunkSpec>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        for chunk in chunks {
            let bank = usize::from(chunk.bank);
            if bank >= ARENA_BANK_COUNT
                || chunk.bytes == 0
                || chunk.bytes as u64 > ARENA_SUPERBLOCK_BYTES
                || chunk.logical_offset % ARENA_ALIGNMENT != 0
                || chunk.sha256 == [0; 32]
            {
                return Err(PersistentError::Config(
                    "arena chunk violates bank, size, alignment, or hash contract".to_string(),
                ));
            }
            by_bank[bank].push(chunk.clone());
        }
        if by_bank.iter().any(Vec::is_empty) {
            return Err(PersistentError::Config(
                "all four banks require at least one arena chunk".to_string(),
            ));
        }
        for bank in &mut by_bank {
            bank.sort_by_key(|chunk| chunk.logical_offset);
            for pair in bank.windows(2) {
                let end = pair[0]
                    .logical_offset
                    .checked_add(pair[0].bytes as u64)
                    .ok_or_else(|| PersistentError::Config("arena chunk overflow".to_string()))?;
                if end > pair[1].logical_offset {
                    return Err(PersistentError::Config(
                        "arena chunks overlap within a bank".to_string(),
                    ));
                }
            }
        }

        let xclbin = CString::new(config.xclbin.to_string_lossy().as_bytes()).map_err(|_| {
            PersistentError::Config("xclbin path contains an interior NUL".to_string())
        })?;
        let device = checked_handle("device_open", ops.device_open(config.device_index))?;
        checked_code("load_xclbin_file", ops.load_xclbin_file(device, &xclbin))?;
        let mut xclbin_uuid = [0u8; 16];
        checked_code(
            "get_xclbin_uuid",
            ops.get_xclbin_uuid(device, &mut xclbin_uuid),
        )?;
        if xclbin_uuid == [0; 16] {
            return Err(PersistentError::Config(
                "loaded xclbin returned a zero UUID".to_string(),
            ));
        }
        let native_device = checked_handle("xcl_open", ops.xcl_open(config.device_index))?;

        let mut ip_indices = [0u32; ARENA_BANK_COUNT];
        for (cu, name) in IP_NAMES.iter().enumerate() {
            let name = CString::new(*name).map_err(|_| {
                PersistentError::Config("persistent CU name contains NUL".to_string())
            })?;
            let index = ops.xcl_ip_name_to_index(native_device, &name);
            if index < 0 {
                return Err(PersistentError::Xrt {
                    operation: "xcl_ip_name_to_index",
                    code: index,
                });
            }
            ip_indices[cu] = index as u32;
            checked_code(
                "xcl_open_context",
                ops.xcl_open_context(native_device, &xclbin_uuid, index as u32, false),
            )?;
            let mut magic = 0u32;
            let mut version = 0u32;
            checked_code(
                "read ABI magic",
                ops.xcl_reg_read(
                    native_device,
                    index as u32,
                    IQ1S_REG_ABI_MAGIC_OFFSET as u32,
                    &mut magic,
                ),
            )?;
            checked_code(
                "read ABI version",
                ops.xcl_reg_read(
                    native_device,
                    index as u32,
                    IQ1S_REG_ABI_VERSION_OFFSET as u32,
                    &mut version,
                ),
            )?;
            if magic != IQ1S_REGISTER_MAGIC || version != IQ1S_ABI_VERSION {
                return Err(PersistentError::Config(format!(
                    "CU {cu} ABI is magic=0x{magic:08x} version={version}, expected 0x{IQ1S_REGISTER_MAGIC:08x}/{IQ1S_ABI_VERSION}"
                )));
            }
        }

        let ring_bytes = usize::try_from(config.command_capacity)
            .ok()
            .and_then(|capacity| capacity.checked_mul(IQ1S_COMMAND_BYTES))
            .ok_or_else(|| PersistentError::Config("command ring size overflow".to_string()))?;
        let mut cus = Vec::with_capacity(ARENA_BANK_COUNT);
        let mut dma = PersistentDmaCounters::default();
        for cu in 0..ARENA_BANK_COUNT {
            let group = MEMORY_GROUPS[cu];
            let allocate = |bytes: usize| -> Result<Handle, PersistentError> {
                checked_handle("bo_alloc", ops.bo_alloc(device, bytes, 0, group))
            };
            let command_bo = allocate(ring_bytes)?;
            let completion_bo = allocate(ring_bytes)?;
            let program_bo = allocate(PROGRAM_BYTES)?;
            let activation_bo = allocate(ACTIVATION_BYTES)?;
            let output_bo = allocate(OUTPUT_BYTES)?;
            let token_map_bo = allocate(TOKEN_MAP_BYTES)?;
            let mut arena = Vec::new();
            for spec in &by_bank[cu] {
                let bo = allocate(spec.bytes)?;
                let bytes = read_chunk(spec).map_err(PersistentError::Config)?;
                if bytes.len() != spec.bytes
                    || <[u8; 32]>::from(Sha256::digest(&bytes)) != spec.sha256
                {
                    return Err(PersistentError::Config(format!(
                        "arena chunk bank {} offset {} failed length/hash verification",
                        spec.bank, spec.logical_offset
                    )));
                }
                checked_code("write arena chunk", ops.bo_write(bo, &bytes))?;
                checked_code(
                    "sync arena chunk",
                    ops.bo_sync(bo, XRT_BO_SYNC_TO_DEVICE, bytes.len(), 0),
                )?;
                dma.weight_ranges += 1;
                dma.weight_bytes = dma.weight_bytes.saturating_add(bytes.len() as u64);
                let address = ops.bo_address(bo);
                if address == 0 || address % ARENA_ALIGNMENT != 0 {
                    return Err(PersistentError::Config(
                        "arena BO device address is zero or unaligned".to_string(),
                    ));
                }
                arena.push(ArenaChunk {
                    logical_offset: spec.logical_offset,
                    bytes: spec.bytes,
                    bo,
                    address,
                });
            }
            let command_address = ops.bo_address(command_bo);
            let completion_address = ops.bo_address(completion_bo);
            let activation_address = ops.bo_address(activation_bo);
            let output_address = ops.bo_address(output_bo);
            let token_map_address = ops.bo_address(token_map_bo);
            if [
                command_address,
                completion_address,
                activation_address,
                output_address,
                token_map_address,
            ]
            .contains(&0)
            {
                return Err(PersistentError::Config(
                    "runtime BO has a zero device address".to_string(),
                ));
            }
            cus.push(PersistentCu {
                ip_index: ip_indices[cu],
                command_bo,
                completion_bo,
                program_bo,
                activation_bo,
                output_bo,
                token_map_bo,
                command_address,
                completion_address,
                activation_address,
                output_address,
                token_map_address,
                arena,
                command_shadow: vec![0; ring_bytes],
                completion_shadow: vec![0; ring_bytes],
                program_shadow: vec![0; PROGRAM_BYTES],
                command_producer: 0,
                command_consumer: 0,
                completion_consumer: 0,
                cached_program_id: None,
            });
        }
        let cus: [PersistentCu; ARENA_BANK_COUNT] = cus.try_into().map_err(|_| {
            PersistentError::Config("did not construct exactly four CUs".to_string())
        })?;
        let mut pool = Self {
            ops,
            device,
            native_device,
            xclbin_uuid,
            generation,
            config,
            cus,
            poisoned: None,
            measured: false,
            measurement_baseline: PersistentDmaCounters::default(),
            dma,
            closed: false,
        };
        for cu in 0..ARENA_BANK_COUNT {
            pool.configure_and_start(cu)?;
        }
        Ok(pool)
    }

    fn reg_write(&self, cu: usize, offset: usize, value: u32) -> Result<(), PersistentError> {
        checked_code(
            "register write",
            self.ops.xcl_reg_write(
                self.native_device,
                self.cus[cu].ip_index,
                offset as u32,
                value,
            ),
        )
    }

    fn reg_read(&self, cu: usize, offset: usize) -> Result<u32, PersistentError> {
        let mut value = 0;
        checked_code(
            "register read",
            self.ops.xcl_reg_read(
                self.native_device,
                self.cus[cu].ip_index,
                offset as u32,
                &mut value,
            ),
        )?;
        Ok(value)
    }

    fn configure_and_start(&mut self, cu: usize) -> Result<(), PersistentError> {
        let command_address = split_u64(self.cus[cu].command_address);
        let completion_address = split_u64(self.cus[cu].completion_address);
        let generation = split_u64(self.generation);
        for (offset, value) in [
            (IQ1S_REG_SESSION_GENERATION_LO_OFFSET, generation.0),
            (IQ1S_REG_SESSION_GENERATION_HI_OFFSET, generation.1),
            (IQ1S_REG_COMMAND_BASE_LO_OFFSET, command_address.0),
            (IQ1S_REG_COMMAND_BASE_HI_OFFSET, command_address.1),
            (
                IQ1S_REG_COMMAND_CAPACITY_OFFSET,
                self.config.command_capacity,
            ),
            (IQ1S_REG_COMMAND_PRODUCER_OFFSET, 0),
            (IQ1S_REG_COMPLETION_BASE_LO_OFFSET, completion_address.0),
            (IQ1S_REG_COMPLETION_BASE_HI_OFFSET, completion_address.1),
            (
                IQ1S_REG_COMPLETION_CAPACITY_OFFSET,
                self.config.command_capacity,
            ),
            (IQ1S_REG_COMPLETION_CONSUMER_OFFSET, 0),
            (IQ1S_REG_CONTROL_OFFSET, CONTROL_START),
        ] {
            self.reg_write(cu, offset, value)?;
        }
        Ok(())
    }

    pub(crate) fn measurement_begin(&mut self) -> Result<(), PersistentError> {
        if let Some(fault) = &self.poisoned {
            return Err(PersistentError::Poisoned(fault.clone()));
        }
        if self.measured {
            return Err(PersistentError::Config(
                "measurement is already active".to_string(),
            ));
        }
        self.measured = true;
        self.measurement_baseline = self.dma;
        Ok(())
    }

    pub(crate) fn measurement_end(&mut self) -> Result<PersistentDmaCounters, PersistentError> {
        if !self.measured {
            return Err(PersistentError::Config(
                "measurement is not active".to_string(),
            ));
        }
        self.measured = false;
        if self.dma.weight_ranges != self.measurement_baseline.weight_ranges
            || self.dma.weight_bytes != self.measurement_baseline.weight_bytes
        {
            return self.poison(PersistentFault {
                cu: None,
                operation: "weight residency",
                detail: "weight DMA occurred inside the measured window".to_string(),
            });
        }
        self.dma.checked_delta(self.measurement_baseline)
    }

    fn poison<T>(&mut self, fault: PersistentFault) -> Result<T, PersistentError> {
        if self.poisoned.is_none() {
            self.poisoned = Some(fault.clone());
        }
        Err(PersistentError::Fault(
            self.poisoned.clone().unwrap_or(fault),
        ))
    }

    fn resolve_arena(cu: &PersistentCu, logical: u64, bytes: u64) -> Result<u64, PersistentError> {
        let requested_end = logical.checked_add(bytes).ok_or_else(|| {
            PersistentError::InvalidPhase("arena descriptor range overflow".to_string())
        })?;
        for chunk in &cu.arena {
            let end = chunk
                .logical_offset
                .checked_add(chunk.bytes as u64)
                .ok_or_else(|| PersistentError::InvalidPhase("arena range overflow".to_string()))?;
            if logical >= chunk.logical_offset && requested_end <= end {
                return chunk
                    .address
                    .checked_add(logical - chunk.logical_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("arena relocation overflow".to_string())
                    });
            }
        }
        Err(PersistentError::InvalidPhase(format!(
            "arena offset {logical} is not resident"
        )))
    }

    fn descriptor_ranges(start: u32, count: usize, capacity: u32) -> Vec<(usize, usize)> {
        let slot = (start & (capacity - 1)) as usize;
        let capacity = capacity as usize;
        let first = count.min(capacity - slot);
        let mut ranges = vec![(slot * IQ1S_COMMAND_BYTES, first * IQ1S_COMMAND_BYTES)];
        if first < count {
            ranges.push((0, (count - first) * IQ1S_COMMAND_BYTES));
        }
        ranges
    }

    fn write_program_if_needed(
        &mut self,
        cu: usize,
        phase: &CompiledLayerPhase,
    ) -> Result<(), PersistentError> {
        let program_id = phase.commands[cu][0].program_id;
        if self.cus[cu].cached_program_id == Some(program_id) {
            return Ok(());
        }
        let encoded = &phase.programs[cu].encoded;
        if encoded.is_empty() || encoded.len() > PROGRAM_BYTES {
            return Err(PersistentError::InvalidPhase(
                "encoded program is empty or exceeds program BO".to_string(),
            ));
        }
        self.cus[cu].program_shadow.fill(0);
        self.cus[cu].program_shadow[..encoded.len()].copy_from_slice(encoded);
        checked_code(
            "write program BO",
            self.ops
                .bo_write(self.cus[cu].program_bo, &self.cus[cu].program_shadow),
        )?;
        checked_code(
            "sync program BO",
            self.ops.bo_sync(
                self.cus[cu].program_bo,
                XRT_BO_SYNC_TO_DEVICE,
                encoded.len(),
                0,
            ),
        )?;
        self.dma.program_ranges += 1;
        self.cus[cu].cached_program_id = Some(program_id);
        Ok(())
    }

    pub(crate) fn submit_phase(
        &mut self,
        phase: &CompiledLayerPhase,
        activations: &[ActivationRange],
    ) -> Result<CompletedLayerPhase, PersistentError> {
        let result = self.submit_phase_inner(phase, activations);
        if let Err(error) = &result {
            if self.poisoned.is_none()
                && !matches!(
                    error,
                    PersistentError::Poisoned(_) | PersistentError::Shutdown(_)
                )
            {
                self.poisoned = Some(PersistentFault {
                    cu: None,
                    operation: "phase submission",
                    detail: error.to_string(),
                });
            }
        }
        result
    }

    fn submit_phase_inner(
        &mut self,
        phase: &CompiledLayerPhase,
        activations: &[ActivationRange],
    ) -> Result<CompletedLayerPhase, PersistentError> {
        if let Some(fault) = &self.poisoned {
            return Err(PersistentError::Poisoned(fault.clone()));
        }
        if self.closed {
            return Err(PersistentError::Shutdown("pool is closed".to_string()));
        }
        validate_compiled_layer_phase(phase, QWEN_MODEL_CONTEXT_LIMIT)
            .map_err(PersistentError::InvalidPhase)?;
        if activations != phase.activations {
            return Err(PersistentError::InvalidPhase(
                "activation manifest differs from compiled phase".to_string(),
            ));
        }
        let dma_before = self.dma;
        let activation_ranges = merge_ranges(
            activations
                .iter()
                .map(|range| (range.slab_offset as usize, range.bytes as usize)),
        );
        if activation_ranges.is_empty()
            || activation_ranges.iter().any(|(offset, bytes)| {
                offset
                    .checked_add(*bytes)
                    .is_none_or(|end| end > ACTIVATION_BYTES)
            })
        {
            return Err(PersistentError::InvalidPhase(
                "activation DMA range is empty, overflowing, or outside the slab".to_string(),
            ));
        }

        let mut expected: [Vec<Iq1sCommand>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        for cu in 0..ARENA_BANK_COUNT {
            let hardware_consumer = self.reg_read(cu, IQ1S_REG_COMMAND_CONSUMER_OFFSET)?;
            self.cus[cu].command_consumer = hardware_consumer;
            let count = u32::try_from(phase.commands[cu].len()).map_err(|_| {
                PersistentError::InvalidPhase("command count does not fit u32".to_string())
            })?;
            let used = self.cus[cu]
                .command_producer
                .wrapping_sub(self.cus[cu].command_consumer);
            if used > self.config.command_capacity
                || count > self.config.command_capacity.saturating_sub(used)
            {
                let fault = PersistentFault {
                    cu: Some(cu),
                    operation: "command ring",
                    detail: format!(
                        "producer={} consumer={} count={} capacity={}",
                        self.cus[cu].command_producer,
                        self.cus[cu].command_consumer,
                        count,
                        self.config.command_capacity
                    ),
                };
                self.poisoned = Some(fault);
                return Err(PersistentError::RingFull {
                    cu,
                    capacity: self.config.command_capacity,
                });
            }
            self.write_program_if_needed(cu, phase)?;
            let producer = self.cus[cu].command_producer;
            for (index, source) in phase.commands[cu].iter().enumerate() {
                let mut command = *source;
                command.session_generation = self.generation;
                let input_columns = match u32::from(command.role) {
                    IQ1S_ROLE_GATE | IQ1S_ROLE_UP => 4096u64,
                    IQ1S_ROLE_DOWN => 1024u64,
                    _ => {
                        return Err(PersistentError::InvalidPhase(
                            "descriptor has an unsupported role".to_string(),
                        ));
                    }
                };
                let weight_bytes = u64::from(command.row_count)
                    .checked_mul(input_columns / 256)
                    .and_then(|blocks| blocks.checked_mul(50))
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase(
                            "descriptor weight byte count overflow".to_string(),
                        )
                    })?;
                command.arena_offset =
                    Self::resolve_arena(&self.cus[cu], source.arena_offset, weight_bytes)?;
                if source
                    .input_offset
                    .checked_add(u64::from(source.input_bytes))
                    .is_none_or(|end| end > ACTIVATION_BYTES as u64)
                    || source
                        .output_offset
                        .checked_add(u64::from(source.output_bytes))
                        .is_none_or(|end| end > OUTPUT_BYTES as u64)
                    || source
                        .token_map_offset
                        .checked_add(u64::from(source.lane_count) * 4)
                        .is_none_or(|end| end > TOKEN_MAP_BYTES as u64)
                {
                    return Err(PersistentError::InvalidPhase(
                        "descriptor activation, output, or token-map range exceeds its slab"
                            .to_string(),
                    ));
                }
                command.input_offset = self.cus[cu]
                    .activation_address
                    .checked_add(source.input_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("input relocation overflow".to_string())
                    })?;
                command.output_offset = self.cus[cu]
                    .output_address
                    .checked_add(source.output_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("output relocation overflow".to_string())
                    })?;
                command.token_map_offset = self.cus[cu]
                    .token_map_address
                    .checked_add(source.token_map_offset)
                    .ok_or_else(|| {
                        PersistentError::InvalidPhase("token-map relocation overflow".to_string())
                    })?;
                command.crc32 = 0;
                command.crc32 = command_crc(&command);
                let slot = producer.wrapping_add(index as u32) & (self.config.command_capacity - 1);
                let offset = slot as usize * IQ1S_COMMAND_BYTES;
                self.cus[cu].command_shadow[offset..offset + IQ1S_COMMAND_BYTES]
                    .copy_from_slice(struct_bytes(&command));
                expected[cu].push(command);
            }
            checked_code(
                "write command ring",
                self.ops
                    .bo_write(self.cus[cu].command_bo, &self.cus[cu].command_shadow),
            )?;
            for (offset, bytes) in Self::descriptor_ranges(
                producer,
                phase.commands[cu].len(),
                self.config.command_capacity,
            ) {
                checked_code(
                    "sync command ring",
                    self.ops.bo_sync(
                        self.cus[cu].command_bo,
                        XRT_BO_SYNC_TO_DEVICE,
                        bytes,
                        offset,
                    ),
                )?;
                self.dma.command_ranges += 1;
            }
            for (offset, bytes) in &activation_ranges {
                checked_code(
                    "sync activation range",
                    self.ops.bo_sync(
                        self.cus[cu].activation_bo,
                        XRT_BO_SYNC_TO_DEVICE,
                        *bytes,
                        *offset,
                    ),
                )?;
                self.dma.activation_ranges += 1;
            }
            let next_producer = producer.wrapping_add(count);
            self.reg_write(cu, IQ1S_REG_COMMAND_PRODUCER_OFFSET, next_producer)?;
            self.reg_write(cu, IQ1S_REG_DOORBELL_OFFSET, 1)?;
            self.cus[cu].command_producer = next_producer;
        }

        let deadline = Instant::now() + Duration::from_millis(u64::from(self.config.timeout_ms));
        let mut completed: [Vec<Iq1sCompletion>; ARENA_BANK_COUNT] =
            std::array::from_fn(|_| Vec::new());
        for cu in 0..ARENA_BANK_COUNT {
            let expected_end = self.cus[cu]
                .completion_consumer
                .wrapping_add(expected[cu].len() as u32);
            let mut backoff = 1u64;
            let observed_producer = loop {
                let fault_code = self.reg_read(cu, IQ1S_REG_FAULT_CODE_OFFSET)?;
                if fault_code != IQ1S_FAULT_CODE_NONE {
                    return self.poison(PersistentFault {
                        cu: Some(cu),
                        operation: "hardware fault register",
                        detail: format!("fault_code={fault_code}"),
                    });
                }
                let producer = self.reg_read(cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET)?;
                if producer.wrapping_sub(expected_end) < (1u32 << 31) {
                    break producer;
                }
                if Instant::now() >= deadline {
                    return self.poison(PersistentFault {
                        cu: Some(cu),
                        operation: "completion poll",
                        detail: format!("timeout after {} ms", self.config.timeout_ms),
                    });
                }
                std::thread::sleep(Duration::from_micros(backoff));
                backoff = (backoff * 2).min(MAX_BACKOFF_US);
            };
            if observed_producer != expected_end {
                return self.poison(PersistentFault {
                    cu: Some(cu),
                    operation: "completion producer",
                    detail: format!(
                        "producer={observed_producer} expected={expected_end}; no other submission may be in flight"
                    ),
                });
            }
            for (offset, bytes) in Self::descriptor_ranges(
                self.cus[cu].completion_consumer,
                expected[cu].len(),
                self.config.command_capacity,
            ) {
                checked_code(
                    "sync completion ring",
                    self.ops.bo_sync(
                        self.cus[cu].completion_bo,
                        XRT_BO_SYNC_FROM_DEVICE,
                        bytes,
                        offset,
                    ),
                )?;
            }
            checked_code(
                "read completion ring",
                self.ops.bo_read(
                    self.cus[cu].completion_bo,
                    &mut self.cus[cu].completion_shadow,
                ),
            )?;
            for (index, command) in expected[cu].iter().enumerate() {
                let counter = self.cus[cu].completion_consumer.wrapping_add(index as u32);
                let slot = counter & (self.config.command_capacity - 1);
                let offset = slot as usize * IQ1S_COMPLETION_BYTES;
                let completion = struct_from_bytes::<Iq1sCompletion>(
                    &self.cus[cu].completion_shadow[offset..offset + IQ1S_COMPLETION_BYTES],
                )?;
                let mismatch = completion.magic != IQ1S_COMPLETION_MAGIC
                    || completion.abi_version != IQ1S_ABI_VERSION as u16
                    || completion.completion_bytes != IQ1S_COMPLETION_BYTES as u16
                    || completion.status != IQ1S_COMPLETION_STATUS_OK
                    || completion.fault_code != IQ1S_FAULT_CODE_NONE
                    || completion.session_generation != self.generation
                    || completion.transaction_id != command.transaction_id
                    || completion.program_id != command.program_id
                    || completion.trace_id != command.trace_id
                    || completion.layer_id != command.layer_id
                    || completion.phase != command.phase
                    || usize::from(completion.cu_id) != cu
                    || completion.expert_id != command.expert_id
                    || completion.lane_mask != command.lane_mask
                    || completion.rows_completed != command.row_count as u16
                    || completion.descriptor_crc32 != command.crc32
                    || completion.command_index != counter
                    || completion.result_fence == 0;
                if mismatch {
                    return self.poison(PersistentFault {
                        cu: Some(cu),
                        operation: "completion validation",
                        detail: format!("completion {counter} does not match its descriptor"),
                    });
                }
                completed[cu].push(completion);
            }
            self.cus[cu].completion_consumer = expected_end;
            self.cus[cu].command_consumer = self.cus[cu].command_producer;
            self.reg_write(cu, IQ1S_REG_COMPLETION_CONSUMER_OFFSET, expected_end)?;
            for (offset, bytes) in &activation_ranges {
                checked_code(
                    "sync result range",
                    self.ops.bo_sync(
                        self.cus[cu].output_bo,
                        XRT_BO_SYNC_FROM_DEVICE,
                        *bytes,
                        *offset,
                    ),
                )?;
                self.dma.result_ranges += 1;
            }
        }
        if self.measured
            && (self.dma.weight_ranges != self.measurement_baseline.weight_ranges
                || self.dma.weight_bytes != self.measurement_baseline.weight_bytes)
        {
            return self.poison(PersistentFault {
                cu: None,
                operation: "weight residency",
                detail: "weight DMA occurred during submit".to_string(),
            });
        }
        let expanded =
            phase
                .programs
                .iter()
                .fold(ExpandedIq1sCounts::default(), |mut total, program| {
                    total.blocks = total.blocks.saturating_add(program.expanded.blocks);
                    total.grid_passes = total
                        .grid_passes
                        .saturating_add(program.expanded.grid_passes);
                    total.delta_passes = total
                        .delta_passes
                        .saturating_add(program.expanded.delta_passes);
                    total
                });
        Ok(CompletedLayerPhase {
            transaction_id: phase.transaction_id,
            semantic_sha256: phase.semantic_sha256,
            completions: completed,
            expanded,
            dma: self.dma.checked_delta(dma_before)?,
        })
    }

    pub(crate) fn shutdown(&mut self) -> Result<(), PersistentError> {
        if self.closed {
            return Ok(());
        }
        for cu in 0..ARENA_BANK_COUNT {
            self.reg_write(cu, IQ1S_REG_CONTROL_OFFSET, CONTROL_SHUTDOWN)?;
        }
        let deadline = Instant::now() + Duration::from_millis(u64::from(self.config.timeout_ms));
        for cu in 0..ARENA_BANK_COUNT {
            loop {
                if self.reg_read(cu, IQ1S_REG_QUIESCENT_OFFSET)? == 1 {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(PersistentError::Shutdown(format!(
                        "CU {cu} did not become quiescent"
                    )));
                }
                std::thread::yield_now();
            }
        }
        for cu in &self.cus {
            for chunk in &cu.arena {
                checked_code("free arena BO", self.ops.bo_free(chunk.bo))?;
            }
            for bo in cu.runtime_bos() {
                checked_code("free runtime BO", self.ops.bo_free(bo))?;
            }
            checked_code(
                "xcl_close_context",
                self.ops
                    .xcl_close_context(self.native_device, &self.xclbin_uuid, cu.ip_index),
            )?;
        }
        self.ops.xcl_close(self.native_device);
        checked_code("device_close", self.ops.device_close(self.device))?;
        self.closed = true;
        Ok(())
    }
}

impl<O: XrtOps> Drop for PersistentIq1sPool<O> {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::iq1s_layer_trace::{
        compile_layer_phase, LayerPhase, LayerPhasePlan, SemanticIq1sCommand,
    };
    use crate::r#impl::iq1s_weight_arena::ArenaShard;
    use crate::r#impl::iq1s_weight_registry::{Iq1sExpertRole, Iq1sTensorIdentity};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::ffi::CStr;
    use std::path::PathBuf;
    use std::sync::Arc;

    const DEVICE: usize = 1;
    const NATIVE_DEVICE: usize = 2;
    const FIRST_BO: usize = 100;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Event {
        DeviceOpen,
        LoadXclbin,
        XclOpen,
        OpenContext(u32),
        BoAlloc {
            bo: usize,
            bytes: usize,
            group: u32,
        },
        BoSync {
            bo: usize,
            direction: i32,
            offset: usize,
            bytes: usize,
        },
        RegisterRead {
            cu: u32,
            offset: u32,
        },
        RegisterWrite {
            cu: u32,
            offset: u32,
            value: u32,
        },
        CloseContext(u32),
        DeviceClose,
    }

    #[derive(Debug, Clone, Copy)]
    enum CompletionMutation {
        Transaction,
        Program,
        Trace,
        Generation,
        Cu,
        Crc,
    }

    #[derive(Default)]
    struct FakeState {
        events: Vec<Event>,
        next_bo: usize,
        memories: HashMap<usize, Vec<u8>>,
        addresses: HashMap<usize, u64>,
        registers: HashMap<(u32, u32), u32>,
        doorbell_consumer: HashMap<u32, u32>,
        completion_mutation: Option<CompletionMutation>,
        hold_completions: bool,
    }

    struct FakeXrt {
        state: RefCell<FakeState>,
    }

    impl FakeXrt {
        fn new() -> Self {
            let mut state = FakeState {
                next_bo: FIRST_BO,
                ..Default::default()
            };
            for cu in 0..4 {
                state
                    .registers
                    .insert((cu, IQ1S_REG_ABI_MAGIC_OFFSET as u32), IQ1S_REGISTER_MAGIC);
                state
                    .registers
                    .insert((cu, IQ1S_REG_ABI_VERSION_OFFSET as u32), IQ1S_ABI_VERSION);
                state
                    .registers
                    .insert((cu, IQ1S_REG_QUIESCENT_OFFSET as u32), 1);
                state
                    .registers
                    .insert((cu, IQ1S_REG_FAULT_CODE_OFFSET as u32), 0);
                state
                    .registers
                    .insert((cu, IQ1S_REG_COMMAND_CONSUMER_OFFSET as u32), 0);
                state
                    .registers
                    .insert((cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET as u32), 0);
            }
            Self {
                state: RefCell::new(state),
            }
        }

        fn events(&self) -> Vec<Event> {
            self.state.borrow().events.clone()
        }

        fn set_completion_mutation(&self, mutation: CompletionMutation) {
            self.state.borrow_mut().completion_mutation = Some(mutation);
        }

        fn ring_doorbell(state: &mut FakeState, cu: u32) {
            if state.hold_completions {
                return;
            }
            let command_producer = state.registers[&(cu, IQ1S_REG_COMMAND_PRODUCER_OFFSET as u32)];
            let start = *state.doorbell_consumer.get(&cu).unwrap_or(&0);
            let capacity = state.registers[&(cu, IQ1S_REG_COMMAND_CAPACITY_OFFSET as u32)];
            let command_address =
                u64::from(state.registers[&(cu, IQ1S_REG_COMMAND_BASE_LO_OFFSET as u32)])
                    | (u64::from(state.registers[&(cu, IQ1S_REG_COMMAND_BASE_HI_OFFSET as u32)])
                        << 32);
            let completion_address =
                u64::from(state.registers[&(cu, IQ1S_REG_COMPLETION_BASE_LO_OFFSET as u32)])
                    | (u64::from(
                        state.registers[&(cu, IQ1S_REG_COMPLETION_BASE_HI_OFFSET as u32)],
                    ) << 32);
            let command_bo = *state
                .addresses
                .iter()
                .find(|(_, address)| **address == command_address)
                .unwrap()
                .0;
            let completion_bo = *state
                .addresses
                .iter()
                .find(|(_, address)| **address == completion_address)
                .unwrap()
                .0;
            let commands = state.memories[&command_bo].clone();
            for counter in start..command_producer {
                let slot = counter & (capacity - 1);
                let offset = slot as usize * IQ1S_COMMAND_BYTES;
                let command = struct_from_bytes::<Iq1sCommand>(
                    &commands[offset..offset + IQ1S_COMMAND_BYTES],
                )
                .unwrap();
                let mut completion = Iq1sCompletion {
                    magic: IQ1S_COMPLETION_MAGIC,
                    abi_version: IQ1S_ABI_VERSION as u16,
                    completion_bytes: IQ1S_COMPLETION_BYTES as u16,
                    status: IQ1S_COMPLETION_STATUS_OK,
                    fault_code: 0,
                    session_generation: command.session_generation,
                    transaction_id: command.transaction_id,
                    program_id: command.program_id,
                    trace_id: command.trace_id,
                    layer_id: command.layer_id,
                    phase: command.phase,
                    role: command.role,
                    cu_id: cu as u16,
                    expert_id: command.expert_id,
                    lane_mask: command.lane_mask,
                    rows_completed: command.row_count as u16,
                    descriptor_crc32: command.crc32,
                    command_index: counter,
                    cycles: 100,
                    ddr_read_bytes: 1000,
                    ddr_write_bytes: 100,
                    iq1s_blocks: 1,
                    grid_passes: 8,
                    delta_passes: 8,
                    result_fence: counter as u64 + 1,
                    fault_detail: 0,
                };
                if let Some(mutation) = state.completion_mutation.take() {
                    match mutation {
                        CompletionMutation::Transaction => completion.transaction_id ^= 1,
                        CompletionMutation::Program => completion.program_id ^= 1,
                        CompletionMutation::Trace => completion.trace_id ^= 1,
                        CompletionMutation::Generation => completion.session_generation ^= 1,
                        CompletionMutation::Cu => completion.cu_id ^= 1,
                        CompletionMutation::Crc => completion.descriptor_crc32 ^= 1,
                    }
                }
                state.memories.get_mut(&completion_bo).unwrap()
                    [offset..offset + IQ1S_COMPLETION_BYTES]
                    .copy_from_slice(struct_bytes(&completion));
            }
            state.doorbell_consumer.insert(cu, command_producer);
            state.registers.insert(
                (cu, IQ1S_REG_COMMAND_CONSUMER_OFFSET as u32),
                command_producer,
            );
            state.registers.insert(
                (cu, IQ1S_REG_COMPLETION_PRODUCER_OFFSET as u32),
                command_producer,
            );
        }
    }

    impl XrtOps for FakeXrt {
        fn device_open(&self, _index: u32) -> Handle {
            self.state.borrow_mut().events.push(Event::DeviceOpen);
            DEVICE as Handle
        }
        fn device_close(&self, _device: Handle) -> i32 {
            self.state.borrow_mut().events.push(Event::DeviceClose);
            0
        }
        fn load_xclbin_file(&self, _device: Handle, _path: &CStr) -> i32 {
            self.state.borrow_mut().events.push(Event::LoadXclbin);
            0
        }
        fn get_xclbin_uuid(&self, _device: Handle, uuid: &mut Xuid) -> i32 {
            *uuid = [7; 16];
            0
        }
        fn kernel_open_exclusive(&self, _: Handle, _: &Xuid, _: &CStr) -> Handle {
            3 as Handle
        }
        fn kernel_close(&self, _: Handle) -> i32 {
            0
        }
        fn kernel_arg_group_id(&self, _: Handle, _: i32) -> i32 {
            0
        }
        fn kernel_read_register(&self, _: Handle, _: u32, _: &mut u32) -> i32 {
            -1
        }
        fn kernel_write_register(&self, _: Handle, _: u32, _: u32) -> i32 {
            -1
        }
        fn xcl_open(&self, _index: u32) -> Handle {
            self.state.borrow_mut().events.push(Event::XclOpen);
            NATIVE_DEVICE as Handle
        }
        fn xcl_close(&self, _device: Handle) {}
        fn xcl_ip_name_to_index(&self, _: Handle, name: &CStr) -> i32 {
            let name = name.to_string_lossy();
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
        fn xcl_open_context(&self, _: Handle, _: &Xuid, index: u32, _: bool) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::OpenContext(index));
            0
        }
        fn xcl_close_context(&self, _: Handle, _: &Xuid, index: u32) -> i32 {
            self.state
                .borrow_mut()
                .events
                .push(Event::CloseContext(index));
            0
        }
        fn xcl_reg_read(&self, _: Handle, index: u32, offset: u32, value: &mut u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::RegisterRead { cu: index, offset });
            *value = *state.registers.get(&(index, offset)).unwrap_or(&0);
            0
        }
        fn xcl_reg_write(&self, _: Handle, index: u32, offset: u32, value: u32) -> i32 {
            let mut state = self.state.borrow_mut();
            state.events.push(Event::RegisterWrite {
                cu: index,
                offset,
                value,
            });
            state.registers.insert((index, offset), value);
            if offset == IQ1S_REG_DOORBELL_OFFSET as u32 {
                Self::ring_doorbell(&mut state, index);
            }
            0
        }
        fn bo_alloc(&self, _: Handle, size: usize, _: u64, group: u32) -> Handle {
            let mut state = self.state.borrow_mut();
            let bo = state.next_bo;
            state.next_bo += 1;
            state
                .memories
                .insert(bo, vec![0; size.min(8 * 1024 * 1024)]);
            state.addresses.insert(bo, (bo as u64) << 20);
            state.events.push(Event::BoAlloc {
                bo,
                bytes: size,
                group,
            });
            bo as Handle
        }
        fn bo_free(&self, _: Handle) -> i32 {
            0
        }
        fn bo_address(&self, bo: Handle) -> u64 {
            self.state.borrow().addresses[&(bo as usize)]
        }
        fn bo_write(&self, bo: Handle, bytes: &[u8]) -> i32 {
            let mut state = self.state.borrow_mut();
            let memory = state.memories.get_mut(&(bo as usize)).unwrap();
            if bytes.len() > memory.len() {
                return -1;
            }
            memory[..bytes.len()].copy_from_slice(bytes);
            0
        }
        fn bo_read(&self, bo: Handle, bytes: &mut [u8]) -> i32 {
            let state = self.state.borrow();
            let memory = &state.memories[&(bo as usize)];
            if bytes.len() > memory.len() {
                return -1;
            }
            bytes.copy_from_slice(&memory[..bytes.len()]);
            0
        }
        fn bo_sync(&self, bo: Handle, direction: i32, size: usize, offset: usize) -> i32 {
            self.state.borrow_mut().events.push(Event::BoSync {
                bo: bo as usize,
                direction,
                offset,
                bytes: size,
            });
            0
        }
    }

    fn fixture_phase(transaction_id: u64, distinct: bool) -> CompiledLayerPhase {
        let role = Iq1sExpertRole::Gate;
        let tensor = Arc::new(Iq1sTensorIdentity {
            canonical_path: PathBuf::from("/tmp/qwen.gguf"),
            file_offset: 0,
            nbytes: 419_430_400,
            name: "blk.7.ffn_gate_exps.weight".to_string(),
            layer: 7,
            ne: [4096, 1024, 512, 1],
            nb: [50, 800, 819_200, 419_430_400],
            role,
            model_sha256: [1; 32],
            content_sha256: [2; 32],
            device: 1,
            inode: 2,
            modified_ns: 3,
        });
        let token_groups = if distinct {
            vec![vec![0], vec![1]]
        } else {
            vec![vec![0, 1]]
        };
        let mut commands = Vec::new();
        for tokens in token_groups {
            let expert = if distinct { tokens[0] as u16 } else { 0 };
            let mask = tokens.iter().fold(0u16, |mask, token| mask | (1 << token));
            for bank in 0..4 {
                commands.push(SemanticIq1sCommand {
                    layer_id: 7,
                    phase: LayerPhase::PhaseA,
                    role,
                    expert_id: expert,
                    lane_mask: mask,
                    token_ids: tokens.clone(),
                    row_shard: ArenaShard {
                        tensor: tensor.clone(),
                        expert,
                        bank,
                        row_start: u32::from(bank) * 256,
                        row_count: 256,
                        superblock: 0,
                        offset: u64::from(expert) * 1024 * 1024,
                        bytes: 204_800,
                        sha256: [3; 32],
                    },
                });
            }
        }
        compile_layer_phase(
            &LayerPhasePlan {
                transaction_id,
                phase: LayerPhase::PhaseA,
                commands,
                activations: vec![
                    ActivationRange {
                        cuda_ptr: 0x10000,
                        slab_offset: 0,
                        bytes: 4096,
                        stream: 1,
                    },
                    ActivationRange {
                        cuda_ptr: 0x20000,
                        slab_offset: 4096,
                        bytes: 4096,
                        stream: 1,
                    },
                ],
            },
            "compiler",
            QWEN_MODEL_CONTEXT_LIMIT,
        )
        .unwrap()
    }

    fn pool(capacity: u32) -> PersistentIq1sPool<FakeXrt> {
        let bytes = vec![0x5a; 2 * 1024 * 1024];
        let hash: [u8; 32] = Sha256::digest(&bytes).into();
        let chunks = (0..4)
            .map(|bank| ArenaChunkSpec {
                bank,
                logical_offset: 0,
                bytes: bytes.len(),
                sha256: hash,
            })
            .collect::<Vec<_>>();
        PersistentIq1sPool::open(
            FakeXrt::new(),
            PersistentIq1sConfig::checked(
                PathBuf::from("/tmp/qwen.xclbin"),
                0,
                Some(capacity),
                100,
            )
            .unwrap(),
            9,
            &chunks,
            |_| Ok(bytes.clone()),
        )
        .unwrap()
    }

    #[test]
    fn xrt_iq1s_persistent_opens_once_starts_four_and_bounds_arena_chunks() {
        let mut pool = pool(4);
        let events = pool.ops.events();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::DeviceOpen))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::LoadXclbin))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::OpenContext(_)))
                .count(),
            4
        );
        assert_eq!(events.iter().filter(|event| matches!(event, Event::RegisterWrite { offset, value: CONTROL_START, .. } if *offset == IQ1S_REG_CONTROL_OFFSET as u32)).count(), 4);
        assert!(events
            .iter()
            .filter_map(|event| match event {
                Event::BoAlloc { bytes, .. } => Some(*bytes),
                _ => None,
            })
            .all(|bytes| bytes <= ARENA_SUPERBLOCK_BYTES as usize));
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_coalesces_dma_and_keeps_weights_resident() {
        let mut pool = pool(4);
        pool.measurement_begin().unwrap();
        let phase = fixture_phase(41, false);
        let completed = pool.submit_phase(&phase, &phase.activations).unwrap();
        assert_eq!(completed.completions.iter().map(Vec::len).sum::<usize>(), 4);
        let measured = pool.measurement_end().unwrap();
        assert_eq!(measured.weight_ranges, 0);
        assert_eq!(measured.weight_bytes, 0);
        assert_eq!(measured.activation_ranges, 4);
        assert_eq!(measured.result_ranges, 4);
        assert_eq!(measured.command_ranges, 4);
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_wraps_without_overwrite() {
        let mut pool = pool(2);
        for transaction in 1..=3 {
            let phase = fixture_phase(transaction, true);
            pool.submit_phase(&phase, &phase.activations).unwrap();
        }
        assert_eq!(pool.cus[0].command_producer, 6);
        assert_eq!(pool.cus[0].completion_consumer, 6);
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_first_fault_poisons_later_submissions() {
        for mutation in [
            CompletionMutation::Transaction,
            CompletionMutation::Program,
            CompletionMutation::Trace,
            CompletionMutation::Generation,
            CompletionMutation::Cu,
            CompletionMutation::Crc,
        ] {
            let mut pool = pool(4);
            pool.ops.set_completion_mutation(mutation);
            let phase = fixture_phase(51, false);
            assert!(matches!(
                pool.submit_phase(&phase, &phase.activations),
                Err(PersistentError::Fault(_))
            ));
            assert!(matches!(
                pool.submit_phase(&phase, &phase.activations),
                Err(PersistentError::Poisoned(_))
            ));
            pool.shutdown().unwrap();
        }
    }

    #[test]
    fn xrt_iq1s_persistent_rejects_unconsumed_ring_overwrite() {
        let mut pool = pool(4);
        pool.cus[0].command_producer = 4;
        let phase = fixture_phase(61, false);
        assert!(matches!(
            pool.submit_phase(&phase, &phase.activations),
            Err(PersistentError::RingFull { cu: 0, capacity: 4 })
        ));
        assert!(matches!(
            pool.submit_phase(&phase, &phase.activations),
            Err(PersistentError::Poisoned(_))
        ));
        pool.shutdown().unwrap();
    }

    #[test]
    fn xrt_iq1s_persistent_shutdown_closes_only_after_quiescent() {
        let mut pool = pool(4);
        pool.shutdown().unwrap();
        let events = pool.ops.events();
        let last_shutdown = events.iter().rposition(|event| matches!(event, Event::RegisterWrite { offset, value: CONTROL_SHUTDOWN, .. } if *offset == IQ1S_REG_CONTROL_OFFSET as u32)).unwrap();
        let first_close = events
            .iter()
            .position(|event| matches!(event, Event::CloseContext(_)))
            .unwrap();
        let last_quiescent_read = events
            .iter()
            .rposition(|event| matches!(event, Event::RegisterRead { offset, .. } if *offset == IQ1S_REG_QUIESCENT_OFFSET as u32))
            .unwrap();
        assert!(last_shutdown < first_close);
        assert!(last_quiescent_read < first_close);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::CloseContext(_)))
                .count(),
            4
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::DeviceClose))
                .count(),
            1
        );
    }
}
