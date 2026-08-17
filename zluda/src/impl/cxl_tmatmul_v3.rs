use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::mem::size_of;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const V3_VERSION: u32 = 3;
pub(crate) const REQUIRED_CAPABILITIES: u64 = 0x7b;
pub(crate) const LANE_ANY: u32 = u32::MAX;
pub(crate) const BUFFER_TERNARY2: u32 = 1;
pub(crate) const BUFFER_Q8_8_S16: u32 = 2;
pub(crate) const BUFFER_RAW_S64: u32 = 3;
pub(crate) const BUFFER_READ: u32 = 1 << 0;
pub(crate) const BUFFER_WRITE: u32 = 1 << 1;
pub(crate) const BUFFER_MATRIX: u32 = 1 << 2;

const EXPECTED_INSTANCES: u32 = 16;
const EXPECTED_DIM_D: u32 = 2048;
const EXPECTED_DDR_BITS: u32 = 512;
const EXPECTED_DAX_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const EXPECTED_LANE_MASK: u64 = 0xffff;
const EXPECTED_CLOCK_HZ: u64 = 400_000_000;
const DEFAULT_TIMEOUT_MS: u32 = 5_000;
const MAX_SANE_QUEUE_DEPTH: u32 = 1 << 20;

const IOC_NRBITS: u32 = 8;
const IOC_TYPEBITS: u32 = 8;
const IOC_SIZEBITS: u32 = 14;
const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

const fn ioc<T>(direction: u32, magic: u8, number: u8) -> u64 {
    ((direction as u64) << IOC_DIRSHIFT)
        | ((magic as u64) << IOC_TYPESHIFT)
        | ((number as u64) << IOC_NRSHIFT)
        | ((size_of::<T>() as u64) << IOC_SIZESHIFT)
}

pub(crate) const fn iow<T>(magic: u8, number: u8) -> u64 {
    ioc::<T>(IOC_WRITE, magic, number)
}

pub(crate) const fn iowr<T>(magic: u8, number: u8) -> u64 {
    ioc::<T>(IOC_READ | IOC_WRITE, magic, number)
}

pub(crate) const QUERY_CAPS_V3: u64 = 0xC080_CE10;
pub(crate) const REGISTER_BUFFER_V3: u64 = 0xC040_CE11;
pub(crate) const UNREGISTER_BUFFER_V3: u64 = 0x4040_CE12;
pub(crate) const COMMIT_BUFFER_V3: u64 = 0xC040_CE13;
pub(crate) const SUBMIT_V3: u64 = 0xC040_CE14;
pub(crate) const WAIT_V3: u64 = 0xC040_CE15;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapsV3 {
    pub(crate) size: u32,
    pub(crate) version: u32,
    pub(crate) capabilities: u64,
    pub(crate) num_instances: u32,
    pub(crate) dim_d: u32,
    pub(crate) max_batch: u32,
    pub(crate) max_descriptors: u32,
    pub(crate) max_inflight_submissions: u32,
    pub(crate) max_timeout_ms: u32,
    pub(crate) ddr_data_width_bits: u32,
    pub(crate) dax_alignment_bytes: u32,
    pub(crate) dax_bytes: u64,
    pub(crate) per_lane_counter_mask: u64,
    pub(crate) accelerator_clock_hz: u64,
    reserved: [u64; 7],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BufferV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) dpa_offset: u64,
    pub(crate) length: u64,
    pub(crate) format: u32,
    pub(crate) handle: u32,
    pub(crate) generation: u64,
    reserved: [u64; 3],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommitV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) handle: u32,
    reserved0: u32,
    pub(crate) expected_generation: u64,
    pub(crate) new_generation: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TaskV3 {
    pub(crate) request_id: u64,
    pub(crate) flags: u32,
    pub(crate) lane: u32,
    pub(crate) batch: u32,
    pub(crate) valid_out: u32,
    pub(crate) valid_in: u32,
    pub(crate) matrix_handle: u32,
    pub(crate) input_handle: u32,
    pub(crate) output_handle: u32,
    reserved0: u32,
    pub(crate) matrix_generation: u64,
    pub(crate) matrix_offset: u64,
    pub(crate) input_offset: u64,
    pub(crate) input_stride_bytes: u64,
    pub(crate) output_offset: u64,
    pub(crate) output_stride_bytes: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubmitV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) count: u32,
    reserved0: u32,
    pub(crate) tasks_ptr: u64,
    pub(crate) submission_id: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompletionV3 {
    pub(crate) request_id: u64,
    pub(crate) status: i32,
    pub(crate) lane_used: u32,
    pub(crate) flags: u32,
    reserved0: u32,
    pub(crate) accelerator_cycles: u64,
    pub(crate) matrix_bytes_read: u64,
    pub(crate) input_bytes_read: u64,
    pub(crate) output_bytes_written: u64,
    pub(crate) start_cycle: u64,
    pub(crate) end_cycle: u64,
    reserved: [u64; 1],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WaitV3 {
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) timeout_ms: u32,
    pub(crate) max_completions: u32,
    pub(crate) submission_id: u64,
    pub(crate) completions_ptr: u64,
    pub(crate) completed: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

const _: [(); 128] = [(); size_of::<CapsV3>()];
const _: [(); 64] = [(); size_of::<BufferV3>()];
const _: [(); 64] = [(); size_of::<CommitV3>()];
const _: [(); 128] = [(); size_of::<TaskV3>()];
const _: [(); 64] = [(); size_of::<SubmitV3>()];
const _: [(); 80] = [(); size_of::<CompletionV3>()];
const _: [(); 64] = [(); size_of::<WaitV3>()];

pub(crate) trait IoctlOps {
    fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String>;
    fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String>;
    fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String>;
    fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String>;
    fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String>;
    fn wait(&mut self, wait: &mut WaitV3, completions: &mut [CompletionV3]) -> Result<(), String>;
}

impl<T: IoctlOps + ?Sized> IoctlOps for &mut T {
    fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String> {
        (**self).query_caps(caps)
    }
    fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
        (**self).register_buffer(buffer)
    }
    fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
        (**self).unregister_buffer(buffer)
    }
    fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String> {
        (**self).commit_buffer(commit)
    }
    fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String> {
        (**self).submit(submit)
    }
    fn wait(&mut self, wait: &mut WaitV3, completions: &mut [CompletionV3]) -> Result<(), String> {
        (**self).wait(wait, completions)
    }
}

#[derive(Debug)]
pub(crate) struct LinuxIoctl {
    file: File,
}

impl LinuxIoctl {
    fn open(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|error| format!("open {}: {error}", path.display()))?;
        Ok(Self { file })
    }

    fn call<T>(&mut self, request: u64, value: &mut T) -> Result<(), String> {
        let result = unsafe {
            libc::ioctl(
                self.file.as_raw_fd(),
                request as libc::c_ulong,
                value as *mut T,
            )
        };
        if result < 0 {
            Err(format!(
                "ioctl request=0x{request:x}: {}",
                std::io::Error::last_os_error()
            ))
        } else {
            Ok(())
        }
    }
}

impl IoctlOps for LinuxIoctl {
    fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String> {
        self.call(QUERY_CAPS_V3, caps)
    }
    fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
        self.call(REGISTER_BUFFER_V3, buffer)
    }
    fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
        self.call(UNREGISTER_BUFFER_V3, buffer)
    }
    fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String> {
        self.call(COMMIT_BUFFER_V3, commit)
    }
    fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String> {
        self.call(SUBMIT_V3, submit)
    }
    fn wait(&mut self, wait: &mut WaitV3, _completions: &mut [CompletionV3]) -> Result<(), String> {
        self.call(WAIT_V3, wait)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BufferLease {
    owner: u64,
    handle: u32,
    generation: u64,
}

impl BufferLease {
    pub(crate) fn handle(self) -> u32 {
        self.handle
    }
    pub(crate) fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone)]
struct BufferState {
    wire: BufferV3,
    quarantined: bool,
    recyclable: bool,
}

static NEXT_SESSION_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub(crate) struct V3Session<I: IoctlOps> {
    io: I,
    caps: CapsV3,
    owner: u64,
    timeout_ms: u32,
    last_submission_id: u64,
    buffers: HashMap<u32, BufferState>,
}

impl V3Session<LinuxIoctl> {
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::with_io(LinuxIoctl::open(path.as_ref())?)
    }
}

impl<I: IoctlOps> V3Session<I> {
    pub(crate) fn with_io(mut io: I) -> Result<Self, String> {
        let mut caps = CapsV3 {
            size: size_of::<CapsV3>() as u32,
            ..CapsV3::default()
        };
        io.query_caps(&mut caps)?;
        validate_caps(&caps)?;
        let timeout_ms = DEFAULT_TIMEOUT_MS.min(caps.max_timeout_ms);
        let owner = NEXT_SESSION_OWNER.fetch_add(1, Ordering::Relaxed);
        if owner == 0 {
            return Err("session owner identifier wrapped".into());
        }
        Ok(Self {
            io,
            caps,
            owner,
            timeout_ms,
            last_submission_id: 0,
            buffers: HashMap::new(),
        })
    }

    pub(crate) fn caps(&self) -> CapsV3 {
        self.caps
    }

    pub(crate) fn io(&self) -> &I {
        &self.io
    }

    pub(crate) fn io_mut(&mut self) -> &mut I {
        &mut self.io
    }

    pub(crate) fn set_timeout_ms(&mut self, timeout_ms: u32) -> Result<(), String> {
        if timeout_ms == 0 || timeout_ms > self.caps.max_timeout_ms {
            return Err(format!(
                "timeout_ms {timeout_ms} outside 1..={}",
                self.caps.max_timeout_ms
            ));
        }
        self.timeout_ms = timeout_ms;
        Ok(())
    }

    pub(crate) fn register_buffer(
        &mut self,
        dpa_offset: u64,
        length: u64,
        format: u32,
        flags: u32,
    ) -> Result<BufferLease, String> {
        let alignment = u64::from(self.caps.dax_alignment_bytes);
        if length == 0 || dpa_offset % alignment != 0 || length % alignment != 0 {
            return Err(format!(
                "buffer range must be nonempty and {alignment}-byte aligned"
            ));
        }
        let end = dpa_offset
            .checked_add(length)
            .ok_or_else(|| "buffer range overflow".to_string())?;
        if end > self.caps.dax_bytes {
            return Err(format!("buffer range ends outside DAX at {end}"));
        }
        if self.buffers.values().any(|state| {
            let other_start = state.wire.dpa_offset;
            let other_end = other_start + state.wire.length;
            dpa_offset < other_end && other_start < end
        }) {
            return Err("buffer range overlaps an existing registration".into());
        }
        if !matches!(format, BUFFER_TERNARY2 | BUFFER_Q8_8_S16 | BUFFER_RAW_S64) {
            return Err(format!("unsupported buffer format {format}"));
        }
        let valid_flags = BUFFER_READ | BUFFER_WRITE | BUFFER_MATRIX;
        if flags == 0
            || flags & !valid_flags != 0
            || flags & BUFFER_MATRIX != 0 && flags & BUFFER_READ == 0
        {
            return Err(format!("invalid buffer flags 0x{flags:x}"));
        }
        let mut wire = BufferV3 {
            size: size_of::<BufferV3>() as u32,
            flags,
            dpa_offset,
            length,
            format,
            ..BufferV3::default()
        };
        self.io.register_buffer(&mut wire)?;
        if wire.handle == 0 || wire.generation == 0 {
            return Err("REGISTER_BUFFER returned zero handle or generation".into());
        }
        if self.buffers.contains_key(&wire.handle) {
            return Err(format!(
                "REGISTER_BUFFER reused live handle {}",
                wire.handle
            ));
        }
        let lease = BufferLease {
            owner: self.owner,
            handle: wire.handle,
            generation: wire.generation,
        };
        self.buffers.insert(
            wire.handle,
            BufferState {
                wire,
                quarantined: false,
                recyclable: false,
            },
        );
        Ok(lease)
    }

    pub(crate) fn commit_buffer(&mut self, lease: BufferLease) -> Result<BufferLease, String> {
        let state = self.checked_state(lease)?;
        if state.quarantined {
            return Err(format!("handle {} is quarantined", lease.handle));
        }
        if state.wire.flags & BUFFER_MATRIX == 0 {
            return Err(format!("handle {} is not a matrix buffer", lease.handle));
        }
        let mut commit = CommitV3 {
            size: size_of::<CommitV3>() as u32,
            handle: lease.handle,
            expected_generation: lease.generation,
            ..CommitV3::default()
        };
        if let Err(error) = self.io.commit_buffer(&mut commit) {
            if let Some(state) = self.buffers.get_mut(&lease.handle) {
                state.quarantined = true;
                state.recyclable = false;
            }
            return Err(format!("COMMIT_BUFFER attempted: {error}"));
        }
        if commit.new_generation <= lease.generation {
            if let Some(state) = self.buffers.get_mut(&lease.handle) {
                state.quarantined = true;
                state.recyclable = false;
            }
            return Err(format!(
                "COMMIT_BUFFER returned non-increasing generation {}",
                commit.new_generation
            ));
        }
        self.buffers
            .get_mut(&lease.handle)
            .ok_or_else(|| format!("buffer handle {} disappeared during commit", lease.handle))?
            .wire
            .generation = commit.new_generation;
        Ok(BufferLease {
            generation: commit.new_generation,
            ..lease
        })
    }

    pub(crate) fn unregister_buffer(&mut self, lease: BufferLease) -> Result<(), String> {
        let state = self.checked_state(lease)?;
        if state.quarantined {
            return Err(format!("handle {} is quarantined", lease.handle));
        }
        let mut wire = state.wire;
        self.io.unregister_buffer(&mut wire)?;
        self.buffers.remove(&lease.handle);
        Ok(())
    }

    pub(crate) fn is_quarantined(&self, lease: BufferLease) -> bool {
        self.checked_state(lease)
            .map(|state| state.quarantined)
            .unwrap_or(false)
    }

    pub(crate) fn is_recyclable(&self, lease: BufferLease) -> bool {
        self.checked_state(lease)
            .map(|state| state.recyclable && !state.quarantined)
            .unwrap_or(false)
    }

    pub(crate) fn run_tasks(&mut self, tasks: &[TaskV3]) -> Result<Vec<CompletionV3>, String> {
        if let Err(error) = self.validate_tasks(tasks) {
            for handle in referenced_handles(tasks) {
                if let Some(state) = self.buffers.get_mut(&handle) {
                    if state.wire.flags & BUFFER_MATRIX == 0 && !state.quarantined {
                        state.recyclable = true;
                    }
                }
            }
            return Err(error);
        }
        let mut all = HashMap::with_capacity(tasks.len());
        let max_descriptors = self.caps.max_descriptors as usize;
        for batch in tasks.chunks(max_descriptors) {
            let handles = referenced_handles(batch);
            let batch_result = self.run_batch(batch);
            match batch_result {
                Ok(completions) => {
                    for completion in completions {
                        all.insert(completion.request_id, completion);
                    }
                    for handle in handles {
                        if let Some(state) = self.buffers.get_mut(&handle) {
                            if state.wire.flags & BUFFER_MATRIX == 0 {
                                state.recyclable = true;
                            }
                        }
                    }
                }
                Err(error) => {
                    for handle in handles {
                        if let Some(state) = self.buffers.get_mut(&handle) {
                            state.quarantined = true;
                            state.recyclable = false;
                        }
                    }
                    return Err(error);
                }
            }
        }
        tasks
            .iter()
            .map(|task| {
                all.remove(&task.request_id)
                    .ok_or_else(|| format!("missing completion request_id={}", task.request_id))
            })
            .collect()
    }

    fn run_batch(&mut self, tasks: &[TaskV3]) -> Result<Vec<CompletionV3>, String> {
        let mut submit = SubmitV3 {
            size: size_of::<SubmitV3>() as u32,
            count: tasks.len() as u32,
            tasks_ptr: tasks.as_ptr() as usize as u64,
            ..SubmitV3::default()
        };
        self.io
            .submit(&mut submit)
            .map_err(|error| format!("SUBMIT attempted: {error}"))?;
        if submit.submission_id == 0 || submit.submission_id <= self.last_submission_id {
            return Err(format!(
                "invalid non-increasing submission_id={} after {}",
                submit.submission_id, self.last_submission_id
            ));
        }
        self.last_submission_id = submit.submission_id;

        let expected: HashMap<u64, &TaskV3> =
            tasks.iter().map(|task| (task.request_id, task)).collect();
        let mut found = HashMap::with_capacity(tasks.len());
        while found.len() < tasks.len() {
            let remaining = tasks.len() - found.len();
            let mut wire_completions = vec![CompletionV3::default(); remaining];
            let mut wait = WaitV3 {
                size: size_of::<WaitV3>() as u32,
                timeout_ms: self.timeout_ms,
                max_completions: remaining as u32,
                submission_id: submit.submission_id,
                completions_ptr: wire_completions.as_mut_ptr() as usize as u64,
                ..WaitV3::default()
            };
            self.io
                .wait(&mut wait, &mut wire_completions)
                .map_err(|error| format!("WAIT submission_id={}: {error}", submit.submission_id))?;
            let completed = wait.completed as usize;
            if completed == 0 {
                let mut missing: Vec<u64> = expected
                    .keys()
                    .filter(|request_id| !found.contains_key(*request_id))
                    .copied()
                    .collect();
                missing.sort_unstable();
                return Err(format!(
                    "WAIT zero progress for submission_id={}; missing request_ids={missing:?}",
                    submit.submission_id,
                ));
            }
            if completed > wire_completions.len() {
                return Err(format!(
                    "WAIT completed count {completed} exceeds requested {remaining}"
                ));
            }
            for completion in wire_completions.into_iter().take(completed) {
                let task = expected.get(&completion.request_id).ok_or_else(|| {
                    format!("unknown completion request_id={}", completion.request_id)
                })?;
                if found.contains_key(&completion.request_id) {
                    return Err(format!(
                        "duplicate completion request_id={}",
                        completion.request_id
                    ));
                }
                self.validate_completion(task, &completion)?;
                found.insert(completion.request_id, completion);
            }
        }
        tasks
            .iter()
            .map(|task| {
                found
                    .remove(&task.request_id)
                    .ok_or_else(|| format!("missing completion request_id={}", task.request_id))
            })
            .collect()
    }

    fn validate_tasks(&self, tasks: &[TaskV3]) -> Result<(), String> {
        if tasks.is_empty() {
            return Err("task list is empty".into());
        }
        let mut request_ids = HashSet::with_capacity(tasks.len());
        for task in tasks {
            if task.request_id == 0 {
                return Err("request_id must be nonzero".into());
            }
            if !request_ids.insert(task.request_id) {
                return Err(format!("duplicate request_id={}", task.request_id));
            }
            if task.flags != 0 || task.reserved0 != 0 || task.reserved != [0; 4] {
                return Err(format!(
                    "request_id={} has nonzero flags/reserved",
                    task.request_id
                ));
            }
            if task.lane != LANE_ANY
                && (task.lane >= self.caps.num_instances
                    || self.caps.per_lane_counter_mask & (1_u64 << task.lane) == 0)
            {
                return Err(format!(
                    "request_id={} has invalid lane {}",
                    task.request_id, task.lane
                ));
            }
            if task.batch == 0 || task.batch > self.caps.max_batch {
                return Err(format!("request_id={} has invalid batch", task.request_id));
            }
            if task.valid_out == 0
                || task.valid_out > self.caps.dim_d
                || task.valid_in == 0
                || task.valid_in > self.caps.dim_d
            {
                return Err(format!(
                    "request_id={} has invalid geometry",
                    task.request_id
                ));
            }
            let matrix = self.state_for_handle(task.matrix_handle, "matrix")?;
            let input = self.state_for_handle(task.input_handle, "input")?;
            let output = self.state_for_handle(task.output_handle, "output")?;
            if task.matrix_handle == task.input_handle
                || task.matrix_handle == task.output_handle
                || task.input_handle == task.output_handle
            {
                return Err(format!(
                    "request_id={} must reference three distinct handles",
                    task.request_id
                ));
            }
            if matrix.quarantined || input.quarantined || output.quarantined {
                return Err(format!(
                    "request_id={} references quarantined lease",
                    task.request_id
                ));
            }
            if matrix.wire.flags & (BUFFER_MATRIX | BUFFER_READ) != BUFFER_MATRIX | BUFFER_READ {
                return Err(format!(
                    "request_id={} matrix handle has wrong flags",
                    task.request_id
                ));
            }
            if matrix.wire.format != BUFFER_TERNARY2
                || input.wire.format != BUFFER_Q8_8_S16
                || output.wire.format != BUFFER_RAW_S64
            {
                return Err(format!(
                    "request_id={} buffer formats do not match matrix/input/output roles",
                    task.request_id
                ));
            }
            if input.wire.flags & BUFFER_READ == 0 || output.wire.flags & BUFFER_WRITE == 0 {
                return Err(format!(
                    "request_id={} input/output handle has wrong flags",
                    task.request_id
                ));
            }
            if task.matrix_generation != matrix.wire.generation {
                return Err(format!(
                    "request_id={} matrix generation mismatch",
                    task.request_id
                ));
            }
            validate_task_range(task.matrix_offset, 1, 1, matrix.wire.length, "matrix")?;
            validate_task_range(
                task.input_offset,
                task.input_stride_bytes,
                task.batch,
                input.wire.length,
                "input",
            )?;
            validate_task_range(
                task.output_offset,
                task.output_stride_bytes,
                task.batch,
                output.wire.length,
                "output",
            )?;
        }
        Ok(())
    }

    fn validate_completion(&self, task: &TaskV3, completion: &CompletionV3) -> Result<(), String> {
        if completion.status != 0 {
            return Err(format!(
                "request_id={} completion status={}",
                completion.request_id, completion.status
            ));
        }
        if completion.lane_used >= self.caps.num_instances
            || self.caps.per_lane_counter_mask & (1_u64 << completion.lane_used) == 0
        {
            return Err(format!(
                "request_id={} invalid completion lane",
                completion.request_id
            ));
        }
        if task.lane != LANE_ANY && completion.lane_used != task.lane {
            return Err(format!(
                "request_id={} completion lane {} does not match requested lane {}",
                completion.request_id, completion.lane_used, task.lane
            ));
        }
        if completion.flags != 0 || completion.reserved0 != 0 || completion.reserved != [0; 1] {
            return Err(format!(
                "request_id={} invalid completion flags/reserved",
                completion.request_id
            ));
        }
        if completion.end_cycle < completion.start_cycle
            || completion.accelerator_cycles > completion.end_cycle - completion.start_cycle
        {
            return Err(format!(
                "request_id={} invalid completion cycle range",
                completion.request_id
            ));
        }
        let matrix = &self.buffers[&task.matrix_handle].wire;
        let input = &self.buffers[&task.input_handle].wire;
        let output = &self.buffers[&task.output_handle].wire;
        if completion.matrix_bytes_read > matrix.length - task.matrix_offset {
            return Err(format!(
                "request_id={} matrix_bytes_read out of range",
                completion.request_id
            ));
        }
        if completion.input_bytes_read > input.length - task.input_offset {
            return Err(format!(
                "request_id={} input_bytes_read out of range",
                completion.request_id
            ));
        }
        if completion.output_bytes_written > output.length - task.output_offset {
            return Err(format!(
                "request_id={} output_bytes_written out of range",
                completion.request_id
            ));
        }
        Ok(())
    }

    fn checked_state(&self, lease: BufferLease) -> Result<&BufferState, String> {
        if lease.owner != self.owner {
            return Err("buffer lease belongs to a different session owner".into());
        }
        let state = self
            .buffers
            .get(&lease.handle)
            .ok_or_else(|| format!("unknown buffer handle {}", lease.handle))?;
        if state.wire.generation != lease.generation {
            return Err(format!(
                "buffer handle {} generation mismatch",
                lease.handle
            ));
        }
        Ok(state)
    }

    fn state_for_handle(&self, handle: u32, role: &str) -> Result<&BufferState, String> {
        if handle == 0 {
            return Err(format!("{role} handle is zero"));
        }
        self.buffers
            .get(&handle)
            .ok_or_else(|| format!("unknown {role} handle {handle}"))
    }
}

fn validate_caps(caps: &CapsV3) -> Result<(), String> {
    if caps.size != size_of::<CapsV3>() as u32 {
        return Err(format!("caps size {} is not 128", caps.size));
    }
    if caps.version != V3_VERSION {
        return Err(format!("caps version {} is not 3", caps.version));
    }
    if caps.capabilities & REQUIRED_CAPABILITIES != REQUIRED_CAPABILITIES {
        return Err(format!(
            "caps capabilities 0x{:x} lack 0x{REQUIRED_CAPABILITIES:x}",
            caps.capabilities
        ));
    }
    if caps.num_instances != EXPECTED_INSTANCES {
        return Err(format!(
            "caps num_instances {} is not 16",
            caps.num_instances
        ));
    }
    if caps.dim_d != EXPECTED_DIM_D {
        return Err(format!("caps dim_d {} is not 2048", caps.dim_d));
    }
    if caps.max_batch == 0 {
        return Err("caps max_batch is zero".into());
    }
    if caps.max_descriptors == 0 || caps.max_descriptors > MAX_SANE_QUEUE_DEPTH {
        return Err(format!(
            "caps max_descriptors {} is not sane",
            caps.max_descriptors
        ));
    }
    if caps.max_inflight_submissions == 0 || caps.max_inflight_submissions > MAX_SANE_QUEUE_DEPTH {
        return Err(format!(
            "caps max_inflight {} is not sane",
            caps.max_inflight_submissions
        ));
    }
    if caps.max_timeout_ms == 0 {
        return Err("caps max_timeout is zero".into());
    }
    if caps.ddr_data_width_bits != EXPECTED_DDR_BITS {
        return Err(format!(
            "caps ddr width {} is not 512",
            caps.ddr_data_width_bits
        ));
    }
    if caps.dax_alignment_bytes == 0 || !caps.dax_alignment_bytes.is_power_of_two() {
        return Err(format!(
            "caps alignment {} is invalid",
            caps.dax_alignment_bytes
        ));
    }
    if caps.dax_bytes != EXPECTED_DAX_BYTES {
        return Err(format!("caps dax bytes {} is not 32 GiB", caps.dax_bytes));
    }
    if caps.per_lane_counter_mask != EXPECTED_LANE_MASK {
        return Err(format!(
            "caps lane_mask 0x{:x} is not 0xffff",
            caps.per_lane_counter_mask
        ));
    }
    if caps.accelerator_clock_hz != EXPECTED_CLOCK_HZ {
        return Err(format!(
            "caps clock {} is not 400 MHz",
            caps.accelerator_clock_hz
        ));
    }
    if caps.reserved != [0; 7] {
        return Err("caps reserved fields are nonzero".into());
    }
    Ok(())
}

fn validate_task_range(
    offset: u64,
    stride: u64,
    batch: u32,
    length: u64,
    role: &str,
) -> Result<(), String> {
    if stride == 0 {
        return Err(format!("{role} stride is zero"));
    }
    let last = stride
        .checked_mul(u64::from(batch - 1))
        .and_then(|delta| offset.checked_add(delta))
        .ok_or_else(|| format!("{role} range overflow"))?;
    if offset >= length || last >= length {
        return Err(format!("{role} task range outside registered buffer"));
    }
    Ok(())
}

fn referenced_handles(tasks: &[TaskV3]) -> HashSet<u32> {
    let mut handles = HashSet::new();
    for task in tasks {
        handles.insert(task.matrix_handle);
        handles.insert(task.input_handle);
        handles.insert(task.output_handle);
    }
    handles
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::mem::size_of;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug, Default)]
    struct FakeIoctl {
        caps: CapsV3,
        calls: Vec<String>,
        next_handle: u32,
        next_generation: u64,
        next_submission: u64,
        waits: VecDeque<Result<Vec<CompletionV3>, String>>,
        submit_error: Option<String>,
        commit_error: Option<String>,
        commit_generation: Option<u64>,
        wait_timeout_seen: Vec<u32>,
        drops: Option<Arc<AtomicUsize>>,
    }

    impl Drop for FakeIoctl {
        fn drop(&mut self) {
            if let Some(drops) = &self.drops {
                drops.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    impl FakeIoctl {
        fn valid() -> Self {
            Self {
                caps: valid_caps(),
                calls: Vec::new(),
                next_handle: 1,
                next_generation: 1,
                next_submission: 1,
                waits: VecDeque::new(),
                submit_error: None,
                commit_error: None,
                commit_generation: None,
                wait_timeout_seen: Vec::new(),
                drops: None,
            }
        }

        fn with_waits(waits: Vec<Vec<CompletionV3>>) -> Self {
            let mut io = Self::valid();
            io.waits = waits.into_iter().map(Ok).collect();
            io
        }
    }

    impl IoctlOps for FakeIoctl {
        fn query_caps(&mut self, caps: &mut CapsV3) -> Result<(), String> {
            self.calls.push("query".into());
            *caps = self.caps;
            Ok(())
        }

        fn register_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
            self.calls.push(format!("register:{}", buffer.dpa_offset));
            buffer.handle = self.next_handle;
            buffer.generation = self.next_generation;
            self.next_handle += 1;
            Ok(())
        }

        fn unregister_buffer(&mut self, buffer: &mut BufferV3) -> Result<(), String> {
            self.calls.push(format!("unregister:{}", buffer.handle));
            Ok(())
        }

        fn commit_buffer(&mut self, commit: &mut CommitV3) -> Result<(), String> {
            self.calls.push(format!("commit:{}", commit.handle));
            if let Some(error) = self.commit_error.take() {
                return Err(error);
            }
            commit.new_generation = self
                .commit_generation
                .unwrap_or(commit.expected_generation + 1);
            Ok(())
        }

        fn submit(&mut self, submit: &mut SubmitV3) -> Result<(), String> {
            self.calls.push(format!("submit:{}", submit.count));
            if let Some(error) = self.submit_error.take() {
                return Err(error);
            }
            submit.submission_id = self.next_submission;
            self.next_submission += 1;
            Ok(())
        }

        fn wait(
            &mut self,
            wait: &mut WaitV3,
            completions: &mut [CompletionV3],
        ) -> Result<(), String> {
            self.calls.push(format!("wait:{}", wait.submission_id));
            self.wait_timeout_seen.push(wait.timeout_ms);
            let result = self.waits.pop_front().unwrap_or_else(|| Ok(Vec::new()))?;
            assert!(result.len() <= completions.len());
            completions[..result.len()].copy_from_slice(&result);
            wait.completed = result.len() as u32;
            Ok(())
        }
    }

    fn valid_caps() -> CapsV3 {
        CapsV3 {
            size: size_of::<CapsV3>() as u32,
            version: V3_VERSION,
            capabilities: REQUIRED_CAPABILITIES,
            num_instances: 16,
            dim_d: 2048,
            max_batch: 64,
            max_descriptors: 2,
            max_inflight_submissions: 4,
            max_timeout_ms: 5000,
            ddr_data_width_bits: 512,
            dax_alignment_bytes: 4096,
            dax_bytes: 32 * 1024 * 1024 * 1024,
            per_lane_counter_mask: 0xffff,
            accelerator_clock_hz: 400_000_000,
            ..CapsV3::default()
        }
    }

    fn completion(request_id: u64) -> CompletionV3 {
        CompletionV3 {
            request_id,
            lane_used: 15,
            accelerator_cycles: 10,
            start_cycle: 20,
            end_cycle: 30,
            matrix_bytes_read: 4096,
            input_bytes_read: 4096,
            output_bytes_written: 4096,
            ..CompletionV3::default()
        }
    }

    fn register_triplet<I: IoctlOps>(session: &mut V3Session<I>) -> [BufferLease; 3] {
        [
            session
                .register_buffer(0, 8192, BUFFER_TERNARY2, BUFFER_READ | BUFFER_MATRIX)
                .unwrap(),
            session
                .register_buffer(8192, 8192, BUFFER_Q8_8_S16, BUFFER_READ)
                .unwrap(),
            session
                .register_buffer(16384, 8192, BUFFER_RAW_S64, BUFFER_WRITE)
                .unwrap(),
        ]
    }

    fn task(request_id: u64, leases: &[BufferLease; 3]) -> TaskV3 {
        TaskV3 {
            request_id,
            lane: LANE_ANY,
            batch: 1,
            valid_out: 1,
            valid_in: 1,
            matrix_handle: leases[0].handle(),
            input_handle: leases[1].handle(),
            output_handle: leases[2].handle(),
            matrix_generation: leases[0].generation(),
            input_stride_bytes: 4096,
            output_stride_bytes: 4096,
            ..TaskV3::default()
        }
    }

    #[test]
    fn v3_uapi_layout_and_ioctl_numbers_match_loaded_driver() {
        assert_eq!(size_of::<CapsV3>(), 128);
        assert_eq!(size_of::<BufferV3>(), 64);
        assert_eq!(size_of::<CommitV3>(), 64);
        assert_eq!(size_of::<TaskV3>(), 128);
        assert_eq!(size_of::<SubmitV3>(), 64);
        assert_eq!(size_of::<CompletionV3>(), 80);
        assert_eq!(size_of::<WaitV3>(), 64);
        assert_eq!(QUERY_CAPS_V3, 0xC080_CE10);
        assert_eq!(REGISTER_BUFFER_V3, 0xC040_CE11);
        assert_eq!(UNREGISTER_BUFFER_V3, 0x4040_CE12);
        assert_eq!(COMMIT_BUFFER_V3, 0xC040_CE13);
        assert_eq!(SUBMIT_V3, 0xC040_CE14);
        assert_eq!(WAIT_V3, 0xC040_CE15);
        assert_eq!(QUERY_CAPS_V3, iowr::<CapsV3>(0xCE, 0x10));
        assert_eq!(REGISTER_BUFFER_V3, iowr::<BufferV3>(0xCE, 0x11));
        assert_eq!(UNREGISTER_BUFFER_V3, iow::<BufferV3>(0xCE, 0x12));
        assert_eq!(COMMIT_BUFFER_V3, iowr::<CommitV3>(0xCE, 0x13));
        assert_eq!(SUBMIT_V3, iowr::<SubmitV3>(0xCE, 0x14));
        assert_eq!(WAIT_V3, iowr::<WaitV3>(0xCE, 0x15));
    }

    #[test]
    fn caps_fail_closed_one_field_at_a_time() {
        let invalid: [(&str, fn(&mut CapsV3)); 12] = [
            ("version", |c: &mut CapsV3| c.version = 2),
            ("capabilities", |c: &mut CapsV3| c.capabilities = 0x3b),
            ("num_instances", |c: &mut CapsV3| c.num_instances = 15),
            ("dim_d", |c: &mut CapsV3| c.dim_d = 1024),
            ("max_descriptors", |c: &mut CapsV3| c.max_descriptors = 0),
            ("max_inflight", |c: &mut CapsV3| {
                c.max_inflight_submissions = 0
            }),
            ("max_timeout", |c: &mut CapsV3| c.max_timeout_ms = 0),
            ("ddr", |c: &mut CapsV3| c.ddr_data_width_bits = 256),
            ("alignment", |c: &mut CapsV3| c.dax_alignment_bytes = 0),
            ("dax", |c: &mut CapsV3| c.dax_bytes -= 1),
            ("lane_mask", |c: &mut CapsV3| {
                c.per_lane_counter_mask = 0x7fff
            }),
            ("clock", |c: &mut CapsV3| {
                c.accelerator_clock_hz = 399_000_000
            }),
        ];
        for (label, mutate) in invalid {
            let mut io = FakeIoctl::valid();
            mutate(&mut io.caps);
            assert!(
                V3Session::with_io(io).unwrap_err().contains(label),
                "{label}"
            );
        }

        let mut io = FakeIoctl::valid();
        io.caps.max_descriptors = u32::MAX;
        assert!(V3Session::with_io(io)
            .unwrap_err()
            .contains("max_descriptors"));
    }

    #[test]
    fn range_registration_rejects_alignment_bounds_and_overlap() {
        let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        assert!(session
            .register_buffer(1, 4096, BUFFER_RAW_S64, BUFFER_READ)
            .is_err());
        assert!(session
            .register_buffer(0, 4095, BUFFER_RAW_S64, BUFFER_READ)
            .is_err());
        assert!(session
            .register_buffer(
                32 * 1024 * 1024 * 1024 - 4096,
                8192,
                BUFFER_RAW_S64,
                BUFFER_READ
            )
            .is_err());
        session
            .register_buffer(0, 8192, BUFFER_RAW_S64, BUFFER_READ)
            .unwrap();
        assert!(session
            .register_buffer(4096, 4096, BUFFER_RAW_S64, BUFFER_READ)
            .is_err());
    }

    #[test]
    fn commit_validates_owner_handle_and_generation() {
        let mut first = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let lease = first
            .register_buffer(0, 4096, BUFFER_TERNARY2, BUFFER_READ | BUFFER_MATRIX)
            .unwrap();
        let committed = first.commit_buffer(lease).unwrap();
        assert_eq!(committed.generation(), lease.generation() + 1);
        assert!(first
            .commit_buffer(lease)
            .unwrap_err()
            .contains("generation"));

        let mut second = V3Session::with_io(FakeIoctl::valid()).unwrap();
        assert!(second
            .commit_buffer(committed)
            .unwrap_err()
            .contains("owner"));
    }

    #[test]
    fn ambiguous_or_invalid_commit_quarantines_the_matrix_lease() {
        for invalid_generation in [false, true] {
            let mut io = FakeIoctl::valid();
            if invalid_generation {
                io.commit_generation = Some(1);
            } else {
                io.commit_error = Some("commit EIO".into());
            }
            let mut session = V3Session::with_io(io).unwrap();
            let lease = session
                .register_buffer(0, 4096, BUFFER_TERNARY2, BUFFER_READ | BUFFER_MATRIX)
                .unwrap();
            assert!(session.commit_buffer(lease).is_err());
            assert!(session.is_quarantined(lease));
            assert!(session
                .unregister_buffer(lease)
                .unwrap_err()
                .contains("quarantined"));
        }
    }

    #[test]
    fn partial_wait_accepts_lane_15_and_preserves_input_order() {
        let io = FakeIoctl::with_waits(vec![vec![completion(2)], vec![completion(1)]]);
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        let got = session
            .run_tasks(&[task(1, &leases), task(2, &leases)])
            .unwrap();
        assert_eq!(
            got.iter().map(|c| c.request_id).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn completion_lane_must_match_an_explicit_lane_request() {
        let mut session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)]])).unwrap();
        let leases = register_triplet(&mut session);
        let mut explicit = task(1, &leases);
        explicit.lane = 0;
        assert!(session
            .run_tasks(&[explicit])
            .unwrap_err()
            .contains("requested lane"));
    }

    #[test]
    fn batches_at_max_descriptors_and_requires_monotonic_submission_ids() {
        let mut io = FakeIoctl::with_waits(vec![
            vec![completion(1), completion(2)],
            vec![completion(3)],
        ]);
        io.next_submission = 40;
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        session
            .run_tasks(&[task(1, &leases), task(2, &leases), task(3, &leases)])
            .unwrap();
        assert_eq!(
            session
                .io()
                .calls
                .iter()
                .filter(|c| c.starts_with("submit"))
                .count(),
            2
        );

        session.io_mut().next_submission = 40;
        session.io_mut().waits.push_back(Ok(vec![completion(4)]));
        assert!(session
            .run_tasks(&[task(4, &leases)])
            .unwrap_err()
            .contains("submission_id"));
    }

    #[test]
    fn timeout_is_bounded_to_caps() {
        let mut session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)]])).unwrap();
        assert!(session.set_timeout_ms(5001).is_err());
        session.set_timeout_ms(5000).unwrap();
        let leases = register_triplet(&mut session);
        session.run_tasks(&[task(1, &leases)]).unwrap();
        assert_eq!(session.io().wait_timeout_seen, vec![5000]);
    }

    #[test]
    fn wait_rejects_zero_progress_duplicate_unknown_and_bad_completions() {
        let cases = [
            (vec![vec![]], "zero progress"),
            (
                vec![vec![completion(1), completion(1)]],
                "duplicate completion request_id=1",
            ),
            (
                vec![vec![completion(99)]],
                "unknown completion request_id=99",
            ),
        ];
        for (waits, expected) in cases {
            let mut session = V3Session::with_io(FakeIoctl::with_waits(waits)).unwrap();
            let leases = register_triplet(&mut session);
            assert!(session
                .run_tasks(&[task(1, &leases), task(2, &leases)])
                .unwrap_err()
                .contains(expected));
        }

        for (bad, expected) in [
            (
                {
                    let mut c = completion(1);
                    c.status = -5;
                    c
                },
                "status",
            ),
            (
                {
                    let mut c = completion(1);
                    c.lane_used = 16;
                    c
                },
                "lane",
            ),
            (
                {
                    let mut c = completion(1);
                    c.flags = 1;
                    c
                },
                "flags",
            ),
            (
                {
                    let mut c = completion(1);
                    c.end_cycle = 19;
                    c
                },
                "cycle",
            ),
            (
                {
                    let mut c = completion(1);
                    c.output_bytes_written = 9000;
                    c
                },
                "output_bytes",
            ),
        ] {
            let mut session = V3Session::with_io(FakeIoctl::with_waits(vec![vec![bad]])).unwrap();
            let leases = register_triplet(&mut session);
            assert!(
                session
                    .run_tasks(&[task(1, &leases)])
                    .unwrap_err()
                    .contains(expected),
                "{expected}: {bad:?}"
            );
        }
    }

    #[test]
    fn completion_byte_ranges_are_relative_to_task_offsets() {
        let mut done = completion(1);
        done.output_bytes_written = 5000;
        let mut session = V3Session::with_io(FakeIoctl::with_waits(vec![vec![done]])).unwrap();
        let leases = register_triplet(&mut session);
        let mut offset = task(1, &leases);
        offset.output_offset = 4096;
        assert!(session
            .run_tasks(&[offset])
            .unwrap_err()
            .contains("output_bytes"));
    }

    #[test]
    fn missing_completion_is_rejected_when_driver_overreports_count() {
        let mut io = FakeIoctl::valid();
        io.waits
            .push_back(Err("timeout with missing request_id=2".into()));
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        assert!(session
            .run_tasks(&[task(1, &leases), task(2, &leases)])
            .unwrap_err()
            .contains("missing"));
    }

    #[test]
    fn zero_progress_names_request_ids_still_missing_after_partial_wait() {
        let mut session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)], vec![]])).unwrap();
        let leases = register_triplet(&mut session);
        let error = session
            .run_tasks(&[task(1, &leases), task(2, &leases)])
            .unwrap_err();
        assert!(error.contains("missing request_ids=[2]"), "{error}");
    }

    #[test]
    fn task_requires_distinct_handles_with_exact_role_formats() {
        let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let leases = register_triplet(&mut session);
        let mut aliased = task(1, &leases);
        aliased.input_handle = aliased.matrix_handle;
        assert!(session
            .run_tasks(&[aliased])
            .unwrap_err()
            .contains("distinct"));

        let mut other = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let wrong_matrix = other
            .register_buffer(0, 8192, BUFFER_Q8_8_S16, BUFFER_READ | BUFFER_MATRIX)
            .unwrap();
        let input = other
            .register_buffer(8192, 8192, BUFFER_Q8_8_S16, BUFFER_READ)
            .unwrap();
        let output = other
            .register_buffer(16384, 8192, BUFFER_RAW_S64, BUFFER_WRITE)
            .unwrap();
        let wrong = task(2, &[wrong_matrix, input, output]);
        assert!(other.run_tasks(&[wrong]).unwrap_err().contains("formats"));
    }

    #[test]
    fn task_ids_and_handles_are_validated_before_submit_without_quarantine() {
        let mut session = V3Session::with_io(FakeIoctl::valid()).unwrap();
        let leases = register_triplet(&mut session);
        assert!(session.run_tasks(&[]).is_err());
        assert!(session.run_tasks(&[task(0, &leases)]).is_err());
        assert!(session
            .run_tasks(&[task(1, &leases), task(1, &leases)])
            .is_err());
        let mut bad = task(2, &leases);
        bad.matrix_handle = 999;
        assert!(session.run_tasks(&[bad]).is_err());
        assert!(!session.is_quarantined(leases[1]));
        assert!(session.is_recyclable(leases[1]));
        assert!(session.is_recyclable(leases[2]));
        assert!(!session.is_recyclable(leases[0]));
        assert_eq!(
            session
                .io()
                .calls
                .iter()
                .filter(|c| c.starts_with("submit"))
                .count(),
            0
        );
    }

    #[test]
    fn any_failure_after_submit_attempt_quarantines_all_referenced_leases() {
        let mut io = FakeIoctl::valid();
        io.submit_error = Some("submit EIO".into());
        let mut session = V3Session::with_io(io).unwrap();
        let leases = register_triplet(&mut session);
        assert!(session.run_tasks(&[task(1, &leases)]).is_err());
        for lease in leases {
            assert!(session.is_quarantined(lease));
            assert!(session
                .unregister_buffer(lease)
                .unwrap_err()
                .contains("quarantined"));
        }
        assert!(!session
            .io()
            .calls
            .iter()
            .any(|c| c.starts_with("unregister")));
    }

    #[test]
    fn timeout_after_submit_quarantines_and_session_drop_drops_fd_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let mut io = FakeIoctl::valid();
            io.drops = Some(drops.clone());
            io.waits.push_back(Err("ETIMEDOUT".into()));
            let mut session = V3Session::with_io(io).unwrap();
            let leases = register_triplet(&mut session);
            assert!(session
                .run_tasks(&[task(1, &leases)])
                .unwrap_err()
                .contains("ETIMEDOUT"));
            assert!(leases
                .into_iter()
                .all(|lease| session.is_quarantined(lease)));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn successful_submission_recycles_transients_but_keeps_matrix_owned() {
        let mut session =
            V3Session::with_io(FakeIoctl::with_waits(vec![vec![completion(1)]])).unwrap();
        let leases = register_triplet(&mut session);
        session.run_tasks(&[task(1, &leases)]).unwrap();
        assert!(session.is_recyclable(leases[1]));
        assert!(session.is_recyclable(leases[2]));
        assert!(!session.is_recyclable(leases[0]));
        session.unregister_buffer(leases[1]).unwrap();
        assert!(session.io().calls.iter().any(|c| c == "unregister:2"));
    }
}
