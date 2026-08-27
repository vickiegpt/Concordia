use super::tq1_tmatmul::{Q8KBlock, TensorIdentity, Tq1Block, Tq1TensorSource, TQ1_VALUES};
use super::xrt_tmatmul::{XrtTmatmulPool, XrtWaveCompletion, XrtWaveJob};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub(crate) const AU250_DIM: usize = 1024;
pub(crate) const HALF_GROUP: usize = 128;
pub(crate) const TQ1_BLOCKS_PER_K_TILE: usize = 4;
pub(crate) const HALF_GROUPS_PER_K_TILE: usize = 8;
pub(crate) const AU250_MATRIX_BYTES: usize = 262_144;
pub(crate) const RAW_DOT_MIN: i32 = -16_384;
pub(crate) const RAW_DOT_MAX: i32 = 16_384;
const DEFAULT_MATRIX_CACHE_BYTES: usize = 1 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Tq1LogicalGroup {
    pub(crate) token: usize,
    pub(crate) expert_slot: usize,
    pub(crate) expert: usize,
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) block: usize,
    pub(crate) half: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tq1LaneAssignment {
    pub(crate) lane: usize,
    pub(crate) group: Tq1LogicalGroup,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedTq1Job {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) expert: usize,
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) assignments: Vec<Tq1LaneAssignment>,
}

#[derive(Debug)]
pub(crate) struct Tq1MulMatIdOperation {
    pub(crate) tensor: Arc<Tq1TensorSource>,
    pub(crate) activations: Vec<f32>,
    pub(crate) expert_ids: Vec<usize>,
    pub(crate) token_count: usize,
    pub(crate) expert_slots: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Tq1XrtEvidence {
    pub(crate) backend: &'static str,
    pub(crate) eligible_operations: u64,
    pub(crate) handled_operations: u64,
    pub(crate) submission_count: u64,
    pub(crate) completion_count: u64,
    pub(crate) per_cu_submissions: Vec<u64>,
    pub(crate) per_cu_completions: Vec<u64>,
    pub(crate) stall_codes: Vec<u32>,
    pub(crate) raw_min: i32,
    pub(crate) raw_max: i32,
    pub(crate) matrix_bytes: u64,
    pub(crate) input_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) program_bytes: u64,
    pub(crate) dispatch_to_stall_ns: u64,
    pub(crate) clock_hz: u64,
    pub(crate) derived_accelerator_cycles: u64,
    pub(crate) decode_ns: u64,
    pub(crate) pack_ns: u64,
    pub(crate) xrt_ns: u64,
    pub(crate) reconstruct_ns: u64,
}

#[derive(Debug)]
pub(crate) struct Tq1XrtResult {
    pub(crate) outputs: Vec<f32>,
    pub(crate) evidence: Tq1XrtEvidence,
}

pub(crate) trait Tq1WaveExecutor {
    fn lane_capacities(&self) -> Vec<usize>;
    fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String>;
}

impl Tq1WaveExecutor for XrtTmatmulPool {
    fn lane_capacities(&self) -> Vec<usize> {
        XrtTmatmulPool::lane_capacities(self)
    }

    fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String> {
        XrtTmatmulPool::run_wave(self, jobs).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PackedTq1Tile {
    pub(crate) matrix: Arc<[u8]>,
    pub(crate) scales: Vec<[f32; TQ1_BLOCKS_PER_K_TILE]>,
    content_hash: u64,
    decode_ns: u64,
    pack_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PackedMatrixKey {
    Tile {
        identity: TensorIdentity,
        expert: usize,
        row_tile: usize,
        k_tile: usize,
        content_hash: u64,
    },
    #[cfg(test)]
    Test(u64),
}

#[derive(Debug)]
struct PackedMatrixEntry {
    value: Arc<PackedTq1Tile>,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug)]
pub(crate) struct PackedMatrixCache {
    capacity_bytes: usize,
    resident_bytes: usize,
    clock: u64,
    entries: HashMap<PackedMatrixKey, PackedMatrixEntry>,
}

static MATRIX_CACHE: OnceLock<Mutex<Result<PackedMatrixCache, String>>> = OnceLock::new();

impl PackedMatrixCache {
    fn new(capacity_bytes: usize) -> Result<Self, String> {
        if capacity_bytes == 0 {
            return Err("HETGPU_TQ1_MATRIX_CACHE_BYTES must be positive".to_string());
        }
        Ok(Self {
            capacity_bytes,
            resident_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
        })
    }

    fn from_env() -> Result<Self, String> {
        let value = match std::env::var("HETGPU_TQ1_MATRIX_CACHE_BYTES") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(format!("read HETGPU_TQ1_MATRIX_CACHE_BYTES: {error}")),
        };
        Self::new(parse_matrix_cache_capacity(value.as_deref())?)
    }

    fn next_clock(&mut self) -> Result<u64, String> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| "TQ1_0 matrix cache clock overflow".to_string())?;
        Ok(self.clock)
    }

    fn get(&mut self, key: &PackedMatrixKey) -> Result<Option<Arc<PackedTq1Tile>>, String> {
        let clock = self.next_clock()?;
        Ok(self.entries.get_mut(key).map(|entry| {
            entry.last_used = clock;
            Arc::clone(&entry.value)
        }))
    }

    fn get_tile(
        &mut self,
        identity: &TensorIdentity,
        expert: usize,
        row_tile: usize,
        k_tile: usize,
    ) -> Result<Option<Arc<PackedTq1Tile>>, String> {
        let key = self
            .entries
            .keys()
            .find(|key| {
                matches!(
                    key,
                    PackedMatrixKey::Tile {
                        identity: candidate,
                        expert: candidate_expert,
                        row_tile: candidate_row_tile,
                        k_tile: candidate_k_tile,
                        ..
                    } if candidate == identity
                        && *candidate_expert == expert
                        && *candidate_row_tile == row_tile
                        && *candidate_k_tile == k_tile
                )
            })
            .cloned();
        match key {
            Some(key) => self.get(&key),
            None => Ok(None),
        }
    }

    fn insert(
        &mut self,
        key: PackedMatrixKey,
        value: PackedTq1Tile,
    ) -> Result<Arc<PackedTq1Tile>, String> {
        let bytes = value
            .matrix
            .len()
            .checked_add(value.scales.len() * std::mem::size_of::<[f32; 4]>())
            .ok_or_else(|| "TQ1_0 matrix cache entry size overflow".to_string())?;
        if bytes > self.capacity_bytes {
            return Err(format!(
                "packed TQ1_0 matrix requires {bytes} bytes, exceeding cache capacity {}",
                self.capacity_bytes
            ));
        }
        while self
            .resident_bytes
            .checked_add(bytes)
            .ok_or_else(|| "TQ1_0 matrix cache size overflow".to_string())?
            > self.capacity_bytes
        {
            let evict = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
                .ok_or_else(|| "TQ1_0 matrix cache cannot evict an entry".to_string())?;
            let removed = self.entries.remove(&evict).expect("selected entry exists");
            self.resident_bytes -= removed.bytes;
        }
        let clock = self.next_clock()?;
        let value = Arc::new(value);
        self.resident_bytes += bytes;
        self.entries.insert(
            key,
            PackedMatrixEntry {
                value: Arc::clone(&value),
                bytes,
                last_used: clock,
            },
        );
        Ok(value)
    }

    #[cfg(test)]
    fn insert_for_test(&mut self, key: u64, matrix: Vec<u8>) -> Result<(), String> {
        self.insert(
            PackedMatrixKey::Test(key),
            PackedTq1Tile {
                matrix: Arc::from(matrix),
                scales: Vec::new(),
                content_hash: key,
                decode_ns: 1,
                pack_ns: 1,
            },
        )?;
        Ok(())
    }

    #[cfg(test)]
    fn get_for_test(&mut self, key: u64) -> Option<Arc<PackedTq1Tile>> {
        self.get(&PackedMatrixKey::Test(key)).ok().flatten()
    }
}

pub(crate) fn parse_matrix_cache_capacity(value: Option<&str>) -> Result<usize, String> {
    match value {
        None => Ok(DEFAULT_MATRIX_CACHE_BYTES),
        Some(value) => {
            let parsed = value.parse::<usize>().map_err(|_| {
                "HETGPU_TQ1_MATRIX_CACHE_BYTES must be a positive integer".to_string()
            })?;
            if parsed == 0 {
                return Err("HETGPU_TQ1_MATRIX_CACHE_BYTES must be positive".to_string());
            }
            Ok(parsed)
        }
    }
}

pub(crate) fn derived_cycles(elapsed_ns: u64, clock_hz: u64) -> Result<u64, String> {
    if elapsed_ns == 0 || clock_hz == 0 {
        return Err("dispatch-to-STALL time and XRT clock must be positive".to_string());
    }
    let numerator = u128::from(elapsed_ns)
        .checked_mul(u128::from(clock_hz))
        .ok_or_else(|| "derived accelerator cycle multiplication overflow".to_string())?;
    u64::try_from((numerator + 999_999_999) / 1_000_000_000)
        .map_err(|_| "derived accelerator cycle count does not fit u64".to_string())
}

fn required_clock_hz() -> Result<u64, String> {
    let value = std::env::var("HETGPU_XRT_CLOCK_HZ")
        .map_err(|_| "HETGPU_XRT_CLOCK_HZ is required for strict TQ1_0 execution".to_string())?;
    let clock_hz = value
        .parse::<u64>()
        .map_err(|_| "HETGPU_XRT_CLOCK_HZ must be a positive integer".to_string())?;
    if clock_hz == 0 {
        return Err("HETGPU_XRT_CLOCK_HZ must be positive".to_string());
    }
    Ok(clock_hz)
}

pub(crate) fn trit_code(trit: i8) -> Result<u8, String> {
    match trit {
        -1 => Ok(0b11),
        0 => Ok(0b00),
        1 => Ok(0b01),
        value => Err(format!("invalid trit {value}")),
    }
}

pub(crate) fn pack_four(trits: [i8; 4]) -> Result<u8, String> {
    Ok(trit_code(trits[0])?
        | (trit_code(trits[1])? << 2)
        | (trit_code(trits[2])? << 4)
        | (trit_code(trits[3])? << 6))
}

pub(crate) fn pack_matrix_tile(
    source: &Tq1TensorSource,
    expert: usize,
    row_tile: usize,
    k_tile: usize,
) -> Result<PackedTq1Tile, String> {
    let rows = usize::try_from(source.identity.ne[1])
        .map_err(|_| "TQ1_0 row count does not fit usize".to_string())?;
    let experts = usize::try_from(source.identity.ne[2])
        .map_err(|_| "TQ1_0 expert count does not fit usize".to_string())?;
    let blocks = usize::try_from(source.identity.ne[0] / TQ1_VALUES as u64)
        .map_err(|_| "TQ1_0 block count does not fit usize".to_string())?;
    if expert >= experts {
        return Err("TQ1_0 tile expert is out of bounds".to_string());
    }
    let row_start = row_tile
        .checked_mul(AU250_DIM)
        .ok_or_else(|| "TQ1_0 tile row start overflow".to_string())?;
    let block_start = k_tile
        .checked_mul(TQ1_BLOCKS_PER_K_TILE)
        .ok_or_else(|| "TQ1_0 tile block start overflow".to_string())?;
    if row_start >= rows || block_start >= blocks {
        return Err("TQ1_0 tile coordinate is out of bounds".to_string());
    }
    let valid_rows = (rows - row_start).min(AU250_DIM);
    let valid_blocks = (blocks - block_start).min(TQ1_BLOCKS_PER_K_TILE);

    let decode_started = Instant::now();
    let mut decoded = vec![0i8; valid_rows * AU250_DIM];
    let mut scales = vec![[0.0f32; TQ1_BLOCKS_PER_K_TILE]; valid_rows];
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for local_row in 0..valid_rows {
        let bytes = source.read_row_blocks(
            expert as u64,
            (row_start + local_row) as u64,
            block_start as u64,
            valid_blocks as u64,
        )?;
        bytes.hash(&mut hasher);
        for block in 0..valid_blocks {
            let start = block * super::tq1_tmatmul::TQ1_BLOCK_BYTES;
            let decoded_block =
                Tq1Block::decode(&bytes[start..start + super::tq1_tmatmul::TQ1_BLOCK_BYTES])?;
            let destination = local_row * AU250_DIM + block * TQ1_VALUES;
            decoded[destination..destination + TQ1_VALUES].copy_from_slice(&decoded_block.trits);
            scales[local_row][block] = decoded_block.scale;
        }
    }
    let decode_ns = u64::try_from(decode_started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1);

    let pack_started = Instant::now();
    let mut matrix = vec![0u8; AU250_MATRIX_BYTES];
    for (byte, trits) in matrix[..valid_rows * AU250_DIM / 4]
        .iter_mut()
        .zip(decoded.chunks_exact(4))
    {
        *byte = pack_four(trits.try_into().expect("exact four-trit chunk"))?;
    }
    let pack_ns = u64::try_from(pack_started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1);
    Ok(PackedTq1Tile {
        matrix: Arc::from(matrix),
        scales,
        content_hash: hasher.finish(),
        decode_ns,
        pack_ns,
    })
}

pub(crate) fn logical_half_groups(
    token: usize,
    expert_slot: usize,
    expert: usize,
    row_tile: usize,
    k_tile: usize,
) -> Vec<Tq1LaneAssignment> {
    let mut assignments = Vec::with_capacity(HALF_GROUPS_PER_K_TILE);
    for block in 0..TQ1_BLOCKS_PER_K_TILE {
        for half in 0..2 {
            assignments.push(Tq1LaneAssignment {
                lane: assignments.len(),
                group: Tq1LogicalGroup {
                    token,
                    expert_slot,
                    expert,
                    row_tile,
                    k_tile,
                    block,
                    half,
                },
            });
        }
    }
    assignments
}

fn plan_tile_jobs_from(
    groups: &[Tq1LaneAssignment],
    lane_capacities: &[usize],
    mut request_id: u64,
    mut cu_cursor: usize,
) -> Result<(Vec<PlannedTq1Job>, u64, usize), String> {
    if groups.is_empty() {
        return Err("TQ1_0 tile has no half-groups".to_string());
    }
    if lane_capacities.is_empty() || lane_capacities.iter().any(|lanes| *lanes == 0) {
        return Err("TQ1_0 XRT lane capacities must be positive".to_string());
    }
    let first = groups[0].group;
    if groups.iter().any(|assignment| {
        let group = assignment.group;
        group.token != first.token
            || group.expert_slot != first.expert_slot
            || group.expert != first.expert
            || group.row_tile != first.row_tile
            || group.k_tile != first.k_tile
    }) {
        return Err("TQ1_0 tile assignments do not share one matrix tile".to_string());
    }

    let mut jobs = Vec::new();
    let mut next = 0;
    while next < groups.len() {
        let cu_index = cu_cursor % lane_capacities.len();
        cu_cursor = cu_cursor
            .checked_add(1)
            .ok_or_else(|| "TQ1_0 CU cursor overflow".to_string())?;
        let count = lane_capacities[cu_index].min(groups.len() - next);
        let assignments = groups[next..next + count]
            .iter()
            .enumerate()
            .map(|(lane, assignment)| Tq1LaneAssignment {
                lane,
                group: assignment.group,
            })
            .collect();
        jobs.push(PlannedTq1Job {
            request_id,
            cu_index,
            expert: first.expert,
            row_tile: first.row_tile,
            k_tile: first.k_tile,
            assignments,
        });
        request_id = request_id
            .checked_add(1)
            .ok_or_else(|| "TQ1_0 request ID overflow".to_string())?;
        next += count;
    }
    Ok((jobs, request_id, cu_cursor))
}

pub(crate) fn plan_tile_jobs(
    groups: &[Tq1LaneAssignment],
    lane_capacities: &[usize],
) -> Result<Vec<PlannedTq1Job>, String> {
    Ok(plan_tile_jobs_from(groups, lane_capacities, 1, 0)?.0)
}

pub(crate) fn pack_lane_input(
    lanes: usize,
    assignments: &[Tq1LaneAssignment],
    q8_blocks: &[Q8KBlock],
) -> Result<Vec<u8>, String> {
    if lanes == 0 || assignments.len() != q8_blocks.len() {
        return Err("TQ1_0 lane assignments and Q8_K blocks do not match".to_string());
    }
    let mut seen_lanes = HashSet::new();
    let mut output = vec![0u8; AU250_DIM * lanes * 2];
    for (assignment, q8) in assignments.iter().zip(q8_blocks) {
        if assignment.lane >= lanes || !seen_lanes.insert(assignment.lane) {
            return Err("TQ1_0 lane assignment is invalid or duplicated".to_string());
        }
        if assignment.group.block >= TQ1_BLOCKS_PER_K_TILE || assignment.group.half >= 2 {
            return Err("TQ1_0 half-group is outside a D=1024 tile".to_string());
        }
        let k_start = assignment.group.block * TQ1_VALUES + assignment.group.half * HALF_GROUP;
        let q_start = assignment.group.half * HALF_GROUP;
        for offset in 0..HALF_GROUP {
            let destination = ((k_start + offset) * lanes + assignment.lane) * 2;
            output[destination..destination + 2]
                .copy_from_slice(&i16::from(q8.qs[q_start + offset]).to_le_bytes());
        }
    }
    Ok(output)
}

fn checked_add(target: &mut u64, value: usize, label: &str) -> Result<(), String> {
    *target = target
        .checked_add(u64::try_from(value).map_err(|_| format!("{label} does not fit u64"))?)
        .ok_or_else(|| format!("{label} counter overflow"))?;
    Ok(())
}

fn raw_slot_index(
    logical: usize,
    row: usize,
    k_tile: usize,
    block: usize,
    half: usize,
    rows: usize,
    k_tiles: usize,
) -> Result<usize, String> {
    logical
        .checked_mul(rows)
        .and_then(|value| value.checked_add(row))
        .and_then(|value| value.checked_mul(k_tiles))
        .and_then(|value| value.checked_add(k_tile))
        .and_then(|value| value.checked_mul(TQ1_BLOCKS_PER_K_TILE))
        .and_then(|value| value.checked_add(block))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(half))
        .ok_or_else(|| "TQ1_0 raw slot index overflow".to_string())
}

fn cached_tile(
    source: &Tq1TensorSource,
    expert: usize,
    row_tile: usize,
    k_tile: usize,
) -> Result<(Arc<PackedTq1Tile>, bool), String> {
    let state = MATRIX_CACHE.get_or_init(|| Mutex::new(PackedMatrixCache::from_env()));
    {
        let mut guard = state
            .lock()
            .map_err(|_| "TQ1_0 matrix cache lock poisoned".to_string())?;
        let cache = match &mut *guard {
            Ok(cache) => cache,
            Err(error) => return Err(error.clone()),
        };
        if let Some(existing) = cache.get_tile(&source.identity, expert, row_tile, k_tile)? {
            return Ok((existing, false));
        }
    }

    let built = pack_matrix_tile(source, expert, row_tile, k_tile)?;
    let key = PackedMatrixKey::Tile {
        identity: source.identity.clone(),
        expert,
        row_tile,
        k_tile,
        content_hash: built.content_hash,
    };
    let mut guard = state
        .lock()
        .map_err(|_| "TQ1_0 matrix cache lock poisoned".to_string())?;
    let cache = match &mut *guard {
        Ok(cache) => cache,
        Err(error) => return Err(error.clone()),
    };
    if let Some(existing) = cache.get(&key)? {
        return Ok((existing, false));
    }
    Ok((cache.insert(key, built)?, true))
}

fn validate_operation(operation: &Tq1MulMatIdOperation) -> Result<(usize, usize, usize), String> {
    let logical_groups = operation
        .token_count
        .checked_mul(operation.expert_slots)
        .ok_or_else(|| "TQ1_0 logical group count overflow".to_string())?;
    if operation.token_count == 0 || operation.expert_slots == 0 {
        return Err("TQ1_0 token and expert-slot counts must be positive".to_string());
    }
    if operation.expert_ids.len() != logical_groups {
        return Err("TQ1_0 expert ID count does not match logical groups".to_string());
    }
    let k = usize::try_from(operation.tensor.identity.ne[0])
        .map_err(|_| "TQ1_0 K does not fit usize".to_string())?;
    let rows = usize::try_from(operation.tensor.identity.ne[1])
        .map_err(|_| "TQ1_0 row count does not fit usize".to_string())?;
    let experts = usize::try_from(operation.tensor.identity.ne[2])
        .map_err(|_| "TQ1_0 expert count does not fit usize".to_string())?;
    if k == 0 || !k.is_multiple_of(AU250_DIM) {
        return Err("strict TQ1_0 XRT execution requires K divisible by 1024".to_string());
    }
    let activation_count = logical_groups
        .checked_mul(k)
        .ok_or_else(|| "TQ1_0 activation count overflow".to_string())?;
    if operation.activations.len() != activation_count
        || operation.activations.iter().any(|value| !value.is_finite())
    {
        return Err("TQ1_0 activations must contain the exact finite f32 extent".to_string());
    }
    if operation.expert_ids.iter().any(|expert| *expert >= experts) {
        return Err("TQ1_0 expert ID is out of bounds".to_string());
    }
    Ok((logical_groups, k, rows))
}

pub(crate) fn execute_mul_mat_id_with(
    operation: &Tq1MulMatIdOperation,
    backend: &mut impl Tq1WaveExecutor,
) -> Result<Tq1XrtResult, String> {
    let clock_hz = required_clock_hz()?;
    let (logical_groups, k, rows) = validate_operation(operation)?;
    let lane_capacities = backend.lane_capacities();
    if lane_capacities.is_empty() || lane_capacities.iter().any(|lanes| *lanes == 0) {
        return Err("TQ1_0 XRT backend has no usable lanes".to_string());
    }
    let row_tiles = rows.div_ceil(AU250_DIM);
    let k_tiles = k / AU250_DIM;
    let blocks_per_row = k / TQ1_VALUES;

    let quantize_started = Instant::now();
    let mut q8 = Vec::with_capacity(logical_groups * blocks_per_row);
    for values in operation.activations.chunks_exact(k) {
        for block in values.chunks_exact(TQ1_VALUES) {
            q8.push(Q8KBlock::quantize(block)?);
        }
    }
    let mut decode_ns = u64::try_from(quantize_started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1);
    let mut pack_ns = 0u64;

    let mut planned_jobs = Vec::new();
    let mut request_id = 1u64;
    let mut cu_cursor = 0usize;
    for logical in 0..logical_groups {
        let token = logical / operation.expert_slots;
        let expert_slot = logical % operation.expert_slots;
        let expert = operation.expert_ids[logical];
        for row_tile in 0..row_tiles {
            for k_tile in 0..k_tiles {
                let groups = logical_half_groups(token, expert_slot, expert, row_tile, k_tile);
                let (mut jobs, next_request, next_cu) =
                    plan_tile_jobs_from(&groups, &lane_capacities, request_id, cu_cursor)?;
                planned_jobs.append(&mut jobs);
                request_id = next_request;
                cu_cursor = next_cu;
            }
        }
    }

    let raw_count = logical_groups
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(k_tiles))
        .and_then(|value| value.checked_mul(HALF_GROUPS_PER_K_TILE))
        .ok_or_else(|| "TQ1_0 raw storage size overflow".to_string())?;
    let mut raw = vec![None::<i16>; raw_count];
    let mut tiles = HashMap::<(usize, usize, usize), Arc<PackedTq1Tile>>::new();
    let mut per_cu_submissions = vec![0u64; lane_capacities.len()];
    let mut per_cu_completions = vec![0u64; lane_capacities.len()];
    let mut submission_count = 0u64;
    let mut completion_count = 0u64;
    let mut stall_codes = Vec::new();
    let mut raw_min = None::<i32>;
    let mut raw_max = None::<i32>;
    let mut matrix_bytes = 0u64;
    let mut input_bytes = 0u64;
    let mut output_bytes = 0u64;
    let mut program_bytes = 0u64;
    let mut dispatch_to_stall_ns = 0u64;
    let mut xrt_ns = 0u64;

    let mut next_job = 0;
    while next_job < planned_jobs.len() {
        let mut used = HashSet::new();
        let wave_start = next_job;
        while next_job < planned_jobs.len() && used.insert(planned_jobs[next_job].cu_index) {
            next_job += 1;
        }
        let wave = &planned_jobs[wave_start..next_job];
        let mut xrt_jobs = Vec::with_capacity(wave.len());
        for planned in wave {
            let tile_key = (planned.expert, planned.row_tile, planned.k_tile);
            let tile = if let Some(tile) = tiles.get(&tile_key) {
                Arc::clone(tile)
            } else {
                let (tile, built) = cached_tile(
                    &operation.tensor,
                    planned.expert,
                    planned.row_tile,
                    planned.k_tile,
                )?;
                if built {
                    decode_ns = decode_ns
                        .checked_add(tile.decode_ns)
                        .ok_or_else(|| "TQ1_0 decode timing overflow".to_string())?;
                    pack_ns = pack_ns
                        .checked_add(tile.pack_ns)
                        .ok_or_else(|| "TQ1_0 pack timing overflow".to_string())?;
                }
                tiles.insert(tile_key, Arc::clone(&tile));
                tile
            };
            let mut assignment_q8 = Vec::with_capacity(planned.assignments.len());
            for assignment in &planned.assignments {
                let logical = assignment
                    .group
                    .token
                    .checked_mul(operation.expert_slots)
                    .and_then(|value| value.checked_add(assignment.group.expert_slot))
                    .ok_or_else(|| "TQ1_0 logical assignment index overflow".to_string())?;
                let global_block =
                    assignment.group.k_tile * TQ1_BLOCKS_PER_K_TILE + assignment.group.block;
                assignment_q8.push(q8[logical * blocks_per_row + global_block].clone());
            }
            let input = pack_lane_input(
                lane_capacities[planned.cu_index],
                &planned.assignments,
                &assignment_q8,
            )?;
            checked_add(&mut matrix_bytes, tile.matrix.len(), "matrix bytes")?;
            checked_add(&mut input_bytes, input.len(), "input bytes")?;
            xrt_jobs.push(XrtWaveJob {
                request_id: planned.request_id,
                cu_index: planned.cu_index,
                matrix: Arc::clone(&tile.matrix),
                input,
            });
            per_cu_submissions[planned.cu_index] = per_cu_submissions[planned.cu_index]
                .checked_add(1)
                .ok_or_else(|| "TQ1_0 per-CU submission count overflow".to_string())?;
            submission_count = submission_count
                .checked_add(1)
                .ok_or_else(|| "TQ1_0 submission count overflow".to_string())?;
        }

        let xrt_started = Instant::now();
        let completions = backend.run_wave(xrt_jobs)?;
        xrt_ns = xrt_ns
            .checked_add(
                u64::try_from(xrt_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX)
                    .max(1),
            )
            .ok_or_else(|| "TQ1_0 XRT timing overflow".to_string())?;
        if completions.len() != wave.len() {
            return Err(format!(
                "TQ1_0 completion count {} does not match planned count {}",
                completions.len(),
                wave.len()
            ));
        }
        let planned_by_id = wave
            .iter()
            .map(|job| (job.request_id, job))
            .collect::<HashMap<_, _>>();
        let mut seen = HashSet::new();
        for completion in &completions {
            let planned = planned_by_id.get(&completion.request_id).ok_or_else(|| {
                format!(
                    "TQ1_0 completion has unknown request id {}",
                    completion.request_id
                )
            })?;
            if !seen.insert(completion.request_id) {
                return Err(format!(
                    "TQ1_0 completion duplicated request id {}",
                    completion.request_id
                ));
            }
            if completion.cu_index != planned.cu_index {
                return Err(format!(
                    "TQ1_0 completion request {} returned CU {}, expected {}",
                    completion.request_id, completion.cu_index, planned.cu_index
                ));
            }
            if completion.stall_code != 1 {
                return Err(format!(
                    "TQ1_0 completion request {} has non-terminal STALL code {}",
                    completion.request_id, completion.stall_code
                ));
            }
            if completion.dispatch_to_stall_ns == 0 || completion.program_bytes == 0 {
                return Err("TQ1_0 completion lacks timing or program-byte evidence".to_string());
            }
            let lanes = lane_capacities[planned.cu_index];
            let expected_output = AU250_DIM
                .checked_mul(lanes)
                .and_then(|value| value.checked_mul(2))
                .ok_or_else(|| "TQ1_0 output byte size overflow".to_string())?;
            if completion.output.len() != expected_output {
                return Err(format!(
                    "TQ1_0 completion request {} has {} output bytes, expected {}",
                    completion.request_id,
                    completion.output.len(),
                    expected_output
                ));
            }
            checked_add(&mut output_bytes, completion.output.len(), "output bytes")?;
            checked_add(
                &mut program_bytes,
                completion.program_bytes,
                "program bytes",
            )?;
            dispatch_to_stall_ns = dispatch_to_stall_ns
                .checked_add(completion.dispatch_to_stall_ns)
                .ok_or_else(|| "TQ1_0 dispatch timing overflow".to_string())?;
            stall_codes.push(completion.stall_code);
            per_cu_completions[planned.cu_index] = per_cu_completions[planned.cu_index]
                .checked_add(1)
                .ok_or_else(|| "TQ1_0 per-CU completion count overflow".to_string())?;
            completion_count = completion_count
                .checked_add(1)
                .ok_or_else(|| "TQ1_0 completion count overflow".to_string())?;

            let valid_rows = (rows - planned.row_tile * AU250_DIM).min(AU250_DIM);
            let assignment_by_lane = planned
                .assignments
                .iter()
                .map(|assignment| (assignment.lane, assignment))
                .collect::<HashMap<_, _>>();
            for local_row in 0..AU250_DIM {
                for lane in 0..lanes {
                    let offset = (local_row * lanes + lane) * 2;
                    let value = i16::from_le_bytes(
                        completion.output[offset..offset + 2]
                            .try_into()
                            .expect("validated two-byte output"),
                    );
                    let Some(assignment) = assignment_by_lane.get(&lane).copied() else {
                        if value != 0 {
                            return Err(
                                "TQ1_0 completion has nonzero unused-lane padding".to_string()
                            );
                        }
                        continue;
                    };
                    if local_row >= valid_rows {
                        if value != 0 {
                            return Err("TQ1_0 completion has nonzero row padding".to_string());
                        }
                        continue;
                    }
                    let value_i32 = i32::from(value);
                    if !(RAW_DOT_MIN..=RAW_DOT_MAX).contains(&value_i32) {
                        return Err(format!(
                            "TQ1_0 raw value {value_i32} is outside [{RAW_DOT_MIN}, {RAW_DOT_MAX}]"
                        ));
                    }
                    let logical = assignment.group.token * operation.expert_slots
                        + assignment.group.expert_slot;
                    let row = planned.row_tile * AU250_DIM + local_row;
                    let slot = raw_slot_index(
                        logical,
                        row,
                        assignment.group.k_tile,
                        assignment.group.block,
                        assignment.group.half,
                        rows,
                        k_tiles,
                    )?;
                    if raw[slot].replace(value).is_some() {
                        return Err("TQ1_0 raw half-group was completed twice".to_string());
                    }
                    raw_min = Some(raw_min.map_or(value_i32, |current| current.min(value_i32)));
                    raw_max = Some(raw_max.map_or(value_i32, |current| current.max(value_i32)));
                }
            }
        }
        if seen.len() != wave.len() {
            return Err("TQ1_0 completion set is missing a request".to_string());
        }
    }

    if let Some(index) = raw.iter().position(Option::is_none) {
        return Err(format!("TQ1_0 raw half-group slot {index} is missing"));
    }
    let reconstruct_started = Instant::now();
    let mut outputs = vec![0.0f32; logical_groups * rows];
    for logical in 0..logical_groups {
        let expert = operation.expert_ids[logical];
        for row_tile in 0..row_tiles {
            let valid_rows = (rows - row_tile * AU250_DIM).min(AU250_DIM);
            for local_row in 0..valid_rows {
                let row = row_tile * AU250_DIM + local_row;
                for k_tile in 0..k_tiles {
                    let tile = tiles.get(&(expert, row_tile, k_tile)).ok_or_else(|| {
                        "TQ1_0 reconstruction is missing a matrix tile".to_string()
                    })?;
                    for block in 0..TQ1_BLOCKS_PER_K_TILE {
                        let low = raw
                            [raw_slot_index(logical, row, k_tile, block, 0, rows, k_tiles)?]
                        .expect("all raw slots checked");
                        let high = raw
                            [raw_slot_index(logical, row, k_tile, block, 1, rows, k_tiles)?]
                        .expect("all raw slots checked");
                        let q8_block =
                            &q8[logical * blocks_per_row + k_tile * TQ1_BLOCKS_PER_K_TILE + block];
                        let block_integer_dot = i32::from(low) + i32::from(high);
                        let contribution = block_integer_dot as f32
                            * tile.scales[local_row][block]
                            * q8_block.scale;
                        if !contribution.is_finite() {
                            return Err(
                                "TQ1_0 reconstruction produced a non-finite value".to_string()
                            );
                        }
                        outputs[logical * rows + row] += contribution;
                    }
                }
            }
        }
    }
    if outputs.iter().any(|value| !value.is_finite()) {
        return Err("TQ1_0 reconstructed output is not finite".to_string());
    }
    let reconstruct_ns = u64::try_from(reconstruct_started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1);
    let derived_accelerator_cycles = derived_cycles(dispatch_to_stall_ns, clock_hz)?;

    Ok(Tq1XrtResult {
        outputs,
        evidence: Tq1XrtEvidence {
            backend: "xrt-tq1-v1",
            eligible_operations: 1,
            handled_operations: 1,
            submission_count,
            completion_count,
            per_cu_submissions,
            per_cu_completions,
            stall_codes,
            raw_min: raw_min.ok_or_else(|| "TQ1_0 execution produced no raw values".to_string())?,
            raw_max: raw_max.ok_or_else(|| "TQ1_0 execution produced no raw values".to_string())?,
            matrix_bytes,
            input_bytes,
            output_bytes,
            program_bytes,
            dispatch_to_stall_ns,
            clock_hz,
            derived_accelerator_cycles,
            decode_ns,
            pack_ns: pack_ns.max(1),
            xrt_ns,
            reconstruct_ns,
        },
    })
}

pub(crate) fn execute_mul_mat_id(operation: &Tq1MulMatIdOperation) -> Result<Tq1XrtResult, String> {
    super::xrt_tmatmul::with_persistent_pool(|pool| execute_mul_mat_id_with(operation, pool))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::tq1_tmatmul::{
        ExpertRole, TensorRegistry, Tq1TensorRegistration, TQ1_BLOCK_BYTES,
    };
    use crate::r#impl::xrt_tmatmul::{XrtWaveCompletion, XrtWaveJob};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn all_positive_tq1_block() -> [u8; TQ1_BLOCK_BYTES] {
        let mut block = [0xff; TQ1_BLOCK_BYTES];
        block[52..54].copy_from_slice(&0x3c00u16.to_le_bytes());
        block
    }

    fn fixture_operation_shape(
        logical_groups: usize,
        rows: usize,
        k: usize,
    ) -> (NamedTempFile, Tq1MulMatIdOperation) {
        let mut file = NamedTempFile::new().unwrap();
        let blocks_per_row = k / 256;
        for _ in 0..rows * blocks_per_row {
            file.write_all(&all_positive_tq1_block()).unwrap();
        }
        file.flush().unwrap();
        let row_bytes = blocks_per_row * TQ1_BLOCK_BYTES;
        let nbytes = rows * row_bytes;
        let source = TensorRegistry::default()
            .register(Tq1TensorRegistration {
                path: file.path().to_path_buf(),
                file_offset: 0,
                nbytes: nbytes as u64,
                name: "blk.20.ffn_down_exps.weight".to_string(),
                ne: [k as u64, rows as u64, 1, 1],
                nb: [54, row_bytes as u64, nbytes as u64, nbytes as u64],
                role: ExpertRole::Down,
            })
            .unwrap();
        let operation = Tq1MulMatIdOperation {
            tensor: source,
            activations: vec![1.0; logical_groups * k],
            expert_ids: vec![0; logical_groups],
            token_count: logical_groups,
            expert_slots: 1,
        };
        (file, operation)
    }

    fn fixture_operation(logical_groups: usize) -> (NamedTempFile, Tq1MulMatIdOperation) {
        fixture_operation_shape(logical_groups, 1, AU250_DIM)
    }

    fn decode_matrix_trit(matrix: &[u8], position: usize) -> i8 {
        match (matrix[position / 4] >> (2 * (position % 4))) & 0b11 {
            0b00 => 0,
            0b01 => 1,
            0b11 => -1,
            code => panic!("invalid packed trit code {code}"),
        }
    }

    #[derive(Default)]
    struct CpuWaveExecutor {
        capacities: Vec<usize>,
        reverse: bool,
        jobs: Vec<XrtWaveJob>,
    }

    #[derive(Clone, Copy)]
    enum CompletionFault {
        WrongCu,
        UnknownRequest,
        DuplicateRequest,
        Missing,
        BadLength,
        BadStall,
        NonzeroLanePadding,
        RawOutOfBounds,
    }

    struct FaultingWaveExecutor {
        inner: CpuWaveExecutor,
        fault: CompletionFault,
    }

    impl Tq1WaveExecutor for FaultingWaveExecutor {
        fn lane_capacities(&self) -> Vec<usize> {
            self.inner.lane_capacities()
        }

        fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String> {
            let mut completions = self.inner.run_wave(jobs)?;
            match self.fault {
                CompletionFault::WrongCu => completions[0].cu_index += 1,
                CompletionFault::UnknownRequest => completions[0].request_id = u64::MAX,
                CompletionFault::DuplicateRequest if completions.len() >= 2 => {
                    completions[1].request_id = completions[0].request_id;
                }
                CompletionFault::DuplicateRequest => {
                    return Err("fault fixture needs two jobs".into())
                }
                CompletionFault::Missing => {
                    completions.pop();
                }
                CompletionFault::BadLength => {
                    completions[0].output.pop();
                }
                CompletionFault::BadStall => completions[0].stall_code = 2,
                CompletionFault::NonzeroLanePadding => {
                    let lanes = self.inner.capacities[completions[0].cu_index];
                    let offset = (lanes - 1) * 2;
                    completions[0].output[offset..offset + 2].copy_from_slice(&1i16.to_le_bytes());
                }
                CompletionFault::RawOutOfBounds => {
                    completions[0].output[..2].copy_from_slice(&20_000i16.to_le_bytes());
                }
            }
            Ok(completions)
        }
    }

    impl CpuWaveExecutor {
        fn new(capacities: Vec<usize>) -> Self {
            Self {
                capacities,
                reverse: true,
                jobs: Vec::new(),
            }
        }
    }

    impl Tq1WaveExecutor for CpuWaveExecutor {
        fn lane_capacities(&self) -> Vec<usize> {
            self.capacities.clone()
        }

        fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String> {
            let mut completions = Vec::new();
            for job in &jobs {
                let lanes = self.capacities[job.cu_index];
                let mut output = vec![0u8; AU250_DIM * lanes * 2];
                for row in 0..AU250_DIM {
                    for lane in 0..lanes {
                        let mut dot = 0i32;
                        for k in 0..AU250_DIM {
                            let offset = (k * lanes + lane) * 2;
                            let quant = i16::from_le_bytes(
                                job.input[offset..offset + 2].try_into().unwrap(),
                            );
                            dot += i32::from(decode_matrix_trit(&job.matrix, row * AU250_DIM + k))
                                * i32::from(quant);
                        }
                        let raw = i16::try_from(dot).map_err(|_| "fake raw overflow")?;
                        let offset = (row * lanes + lane) * 2;
                        output[offset..offset + 2].copy_from_slice(&raw.to_le_bytes());
                    }
                }
                completions.push(XrtWaveCompletion {
                    request_id: job.request_id,
                    cu_index: job.cu_index,
                    stall_code: 1,
                    output,
                    dispatch_to_stall_ns: 500,
                    program_bytes: 96,
                });
            }
            self.jobs.extend(jobs);
            if self.reverse {
                completions.reverse();
            }
            Ok(completions)
        }
    }

    #[test]
    fn ternary_packing_has_exact_codes_size_and_zero_padding() {
        assert_eq!(trit_code(-1).unwrap(), 3);
        assert_eq!(trit_code(0).unwrap(), 0);
        assert_eq!(trit_code(1).unwrap(), 1);
        assert!(trit_code(2).is_err());
        assert_eq!(pack_four([-1, 0, 1, -1]).unwrap(), 0b11_01_00_11);

        let (_file, operation) = fixture_operation(1);
        let packed = pack_matrix_tile(&operation.tensor, 0, 0, 0).unwrap();
        assert_eq!(packed.matrix.len(), AU250_MATRIX_BYTES);
        assert!(packed.matrix[..AU250_DIM / 4]
            .iter()
            .all(|byte| *byte == 0x55));
        assert!(packed.matrix[AU250_DIM / 4..].iter().all(|byte| *byte == 0));
        assert_eq!(packed.scales[0], [1.0; 4]);
    }

    #[test]
    fn nine_lane_cu_gets_one_job_and_six_lane_cu_gets_six_plus_two() {
        let groups = logical_half_groups(0, 0, 0, 0, 0);
        let nine = plan_tile_jobs(&groups, &[9]).unwrap();
        assert_eq!(
            nine.iter()
                .map(|job| job.assignments.len())
                .collect::<Vec<_>>(),
            vec![8]
        );
        let six = plan_tile_jobs(&groups, &[6]).unwrap();
        assert_eq!(
            six.iter()
                .map(|job| job.assignments.len())
                .collect::<Vec<_>>(),
            vec![6, 2]
        );
        assert_ne!(six[0].request_id, six[1].request_id);
    }

    #[test]
    fn lane_input_uses_only_assigned_half_group_and_zeroes_unused_lanes() {
        let q8 = crate::r#impl::tq1_tmatmul::Q8KBlock::quantize(&[1.0; 256]).unwrap();
        let assignments = vec![Tq1LaneAssignment {
            lane: 0,
            group: logical_half_groups(0, 0, 0, 0, 0)[3].group,
        }];
        let input = pack_lane_input(6, &assignments, &[q8]).unwrap();
        for k in 0..AU250_DIM {
            for lane in 0..6 {
                let offset = (k * 6 + lane) * 2;
                let value = i16::from_le_bytes(input[offset..offset + 2].try_into().unwrap());
                let expected = if lane == 0 && (384..512).contains(&k) {
                    -127
                } else {
                    0
                };
                assert_eq!(value, expected, "k={k} lane={lane}");
            }
        }
    }

    #[test]
    fn reversed_four_cu_completions_reconstruct_in_logical_order() {
        let _env = crate::r#impl::test_env::lock();
        std::env::set_var("HETGPU_XRT_CLOCK_HZ", "370000000");
        let (_file, operation) = fixture_operation(4);
        let mut fake = CpuWaveExecutor::new(vec![9, 9, 9, 6]);

        let result = execute_mul_mat_id_with(&operation, &mut fake).unwrap();

        std::env::remove_var("HETGPU_XRT_CLOCK_HZ");
        assert_eq!(result.outputs, vec![1024.0; 4]);
        assert_eq!(result.evidence.per_cu_submissions.len(), 4);
        assert!(result
            .evidence
            .per_cu_submissions
            .iter()
            .all(|count| *count > 0));
        assert_eq!(result.evidence.completion_count, 5);
        assert!(result.evidence.dispatch_to_stall_ns > 0);
        assert!(result.evidence.derived_accelerator_cycles > 0);
        assert_eq!(result.evidence.raw_min, -16_256);
        assert_eq!(result.evidence.raw_max, -16_256);
        assert_eq!(result.evidence.program_bytes, 5 * 96);
        let request_ids = fake
            .jobs
            .iter()
            .map(|job| job.request_id)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(request_ids.len(), fake.jobs.len());
    }

    #[test]
    fn multi_row_and_multi_k_tiles_reconstruct_all_outputs() {
        let _env = crate::r#impl::test_env::lock();
        std::env::set_var("HETGPU_XRT_CLOCK_HZ", "370000000");
        let (_file, operation) = fixture_operation_shape(1, 2, 2048);
        let mut fake = CpuWaveExecutor::new(vec![9]);

        let result = execute_mul_mat_id_with(&operation, &mut fake).unwrap();

        std::env::remove_var("HETGPU_XRT_CLOCK_HZ");
        assert_eq!(result.outputs, vec![2048.0, 2048.0]);
        assert_eq!(result.evidence.submission_count, 2);
        assert_eq!(result.evidence.completion_count, 2);
    }

    #[test]
    fn completion_validation_rejects_ownership_shape_stall_padding_and_bounds() {
        let _env = crate::r#impl::test_env::lock();
        std::env::set_var("HETGPU_XRT_CLOCK_HZ", "370000000");
        let cases = [
            (CompletionFault::WrongCu, vec![9], "returned CU"),
            (CompletionFault::UnknownRequest, vec![9], "unknown request"),
            (CompletionFault::Missing, vec![9], "completion count"),
            (CompletionFault::BadLength, vec![9], "output bytes"),
            (CompletionFault::BadStall, vec![9], "STALL code"),
            (
                CompletionFault::NonzeroLanePadding,
                vec![9],
                "unused-lane padding",
            ),
            (CompletionFault::RawOutOfBounds, vec![9], "outside"),
            (
                CompletionFault::DuplicateRequest,
                vec![4, 4],
                "duplicated request",
            ),
        ];
        for (fault, capacities, expected) in cases {
            let (_file, operation) = fixture_operation(1);
            let mut backend = FaultingWaveExecutor {
                inner: CpuWaveExecutor::new(capacities),
                fault,
            };
            let error = execute_mul_mat_id_with(&operation, &mut backend).unwrap_err();
            assert!(
                error.contains(expected),
                "expected {expected:?} in {error:?}"
            );
        }
        std::env::remove_var("HETGPU_XRT_CLOCK_HZ");
    }

    #[test]
    fn clock_normalized_cycles_are_checked_and_rounded_up() {
        assert_eq!(derived_cycles(500, 370_000_000).unwrap(), 185);
        assert!(derived_cycles(0, 370_000_000).is_err());
        assert!(derived_cycles(500, 0).is_err());
    }

    #[test]
    fn cache_capacity_is_positive_and_lru_eviction_is_deterministic() {
        assert!(parse_matrix_cache_capacity(None).unwrap() >= AU250_MATRIX_BYTES);
        assert!(parse_matrix_cache_capacity(Some("0")).is_err());
        assert!(parse_matrix_cache_capacity(Some("bad")).is_err());
        let mut cache = PackedMatrixCache::new(2 * AU250_MATRIX_BYTES).unwrap();
        cache
            .insert_for_test(1, vec![1; AU250_MATRIX_BYTES])
            .unwrap();
        cache
            .insert_for_test(2, vec![2; AU250_MATRIX_BYTES])
            .unwrap();
        assert!(cache.get_for_test(1).is_some());
        cache
            .insert_for_test(3, vec![3; AU250_MATRIX_BYTES])
            .unwrap();
        assert!(cache.get_for_test(1).is_some());
        assert!(cache.get_for_test(2).is_none());
        assert!(cache.get_for_test(3).is_some());
    }
}
