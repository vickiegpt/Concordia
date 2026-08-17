//! Exact IQ1_S/MMQ decomposition used by the fail-closed TernIP V3 route.

use crate::r#impl::cxl_tmatmul::{copy_cuda_to_host, copy_host_to_cuda};
use crate::r#impl::cxl_tmatmul_v3::{
    BufferLease, CompletedTaskV3, IoctlOps, TaskV3, V3Session, LANE_ANY,
};
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) const IQ1S_BLOCK_BYTES: usize = 50;
pub(crate) const IQ1S_BLOCK_VALUES: usize = 256;
pub(crate) const Q8_1_MMQ_BYTES: usize = 144;
pub(crate) const TILE_DIM: usize = 2048;
pub(crate) const TILE_PACKED_BYTES: usize = TILE_DIM * TILE_DIM / 4;
pub(crate) const GRID_ENTRIES: usize = 2048;
const GROUP_VALUES: usize = 32;
const DEFAULT_LIBGGML: &str =
    "/home/eabban/BitNet/build-cuda128-gcc12/3rdparty/llama.cpp/ggml/src/libggml.so";

pub(crate) type GridTable = [[i8; 8]; GRID_ENTRIES];

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub(crate) struct GgmlType19Signature {
    pub(crate) kernel: String,
    pub(crate) ne00: u64,
    pub(crate) ne01: u64,
    pub(crate) stride01: u64,
    pub(crate) ne10: u64,
    pub(crate) ne11: u64,
    pub(crate) stride11: u64,
    pub(crate) ne0: u64,
}

impl GgmlType19Signature {
    pub(crate) fn validate(&self) -> Result<Self, String> {
        if self.kernel != "mul_mat_q"
            && !(self.kernel.starts_with("_Z9mul_mat_qI")
                && !self.kernel[14..].is_empty()
                && self.kernel[14..]
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_'))
        {
            return Err("kernel is not a qualified IQ1_S mul_mat_q symbol".into());
        }
        if [
            self.ne00,
            self.ne01,
            self.stride01,
            self.ne10,
            self.ne11,
            self.stride11,
            self.ne0,
        ]
        .contains(&0)
        {
            return Err("signature dimensions and strides must be positive".into());
        }
        if self.ne00 != self.ne10 || self.ne01 != self.ne0 {
            return Err("ne10/ne0 do not match ne00/ne01".into());
        }
        if self.ne11 != 1 || self.stride11 != self.ne11 {
            return Err("supported MMQ requires ne11 == stride11 == 1".into());
        }
        if !self.ne00.is_multiple_of(IQ1S_BLOCK_VALUES as u64) {
            return Err("ne00 must be divisible by 256".into());
        }
        if self.stride01 < self.ne00 / IQ1S_BLOCK_VALUES as u64 {
            return Err("stride01 is smaller than the IQ1_S block count".into());
        }
        usize::try_from(self.ne00).map_err(|_| "ne00 does not fit usize")?;
        usize::try_from(self.ne01).map_err(|_| "ne01 does not fit usize")?;
        Ok(self.clone())
    }

    pub(crate) fn matrix_storage_bytes(&self) -> Result<usize, String> {
        self.validate()?;
        let stride = self
            .stride01
            .checked_mul(IQ1S_BLOCK_BYTES as u64)
            .ok_or("matrix stride overflow")?;
        let bytes = self
            .ne01
            .checked_mul(stride)
            .ok_or("matrix size overflow")?;
        usize::try_from(bytes).map_err(|_| "matrix size does not fit usize".into())
    }

    pub(crate) fn activation_storage_bytes(&self) -> Result<usize, String> {
        self.validate()?;
        let groups = self.ne10 / 128;
        usize::try_from(
            groups
                .checked_mul(Q8_1_MMQ_BYTES as u64)
                .ok_or("activation size overflow")?,
        )
        .map_err(|_| "activation size does not fit usize".into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Iq1sGroup {
    pub(crate) odd_scale: u8,
    pub(crate) delta_sign: i8,
    pub(crate) grid_indices: [usize; 4],
    pub(crate) grid_values: [i8; GROUP_VALUES],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Iq1sBlock {
    pub(crate) d: f32,
    pub(crate) groups: [Iq1sGroup; 8],
}

impl Iq1sBlock {
    pub(crate) fn parse(packed: &[u8], grid: &GridTable) -> Result<Self, String> {
        if packed.len() != IQ1S_BLOCK_BYTES {
            return Err("packed IQ1_S block must be exactly 50 bytes".into());
        }
        let d = half_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
        if !d.is_finite() {
            return Err("IQ1_S d must be finite".into());
        }
        let mut groups = [Iq1sGroup {
            odd_scale: 1,
            delta_sign: 1,
            grid_indices: [0; 4],
            grid_values: [0; GROUP_VALUES],
        }; 8];
        for (group_index, group) in groups.iter_mut().enumerate() {
            let qh_offset = 34 + group_index * 2;
            let qh = u16::from_le_bytes([packed[qh_offset], packed[qh_offset + 1]]);
            group.odd_scale = ((((qh >> 12) & 7) * 2) + 1) as u8;
            group.delta_sign = if qh & 0x8000 == 0 { 1 } else { -1 };
            for position in 0..4 {
                let low = usize::from(packed[2 + group_index * 4 + position]);
                let high = usize::from((qh >> (3 * position)) & 7);
                let index = low | high << 8;
                group.grid_indices[position] = index;
                group.grid_values[position * 8..(position + 1) * 8].copy_from_slice(&grid[index]);
            }
        }
        Ok(Self { d, groups })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Q8_1Block {
    pub(crate) d: f32,
    pub(crate) s: f32,
    pub(crate) qs: [i8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Q8_1MmqBlock {
    pub(crate) subblocks: [Q8_1Block; 4],
}

impl Q8_1MmqBlock {
    pub(crate) fn parse(packed: &[u8]) -> Result<Self, String> {
        if packed.len() != Q8_1_MMQ_BYTES {
            return Err("Q8_1 MMQ DS4 block must be exactly 144 bytes".into());
        }
        let mut subblocks = [Q8_1Block {
            d: 0.0,
            s: 0.0,
            qs: [0; 32],
        }; 4];
        for (index, block) in subblocks.iter_mut().enumerate() {
            block.d = half_to_f32(u16::from_le_bytes([
                packed[index * 4],
                packed[index * 4 + 1],
            ]));
            block.s = half_to_f32(u16::from_le_bytes([
                packed[index * 4 + 2],
                packed[index * 4 + 3],
            ]));
            if !block.d.is_finite() || !block.s.is_finite() {
                return Err(format!("Q8_1 MMQ d/s pair {index} must be finite"));
            }
            for (dst, src) in block
                .qs
                .iter_mut()
                .zip(&packed[16 + index * 32..16 + (index + 1) * 32])
            {
                *dst = *src as i8;
            }
        }
        Ok(Self { subblocks })
    }
}

pub(crate) fn iter_q8_1_mmq<'a>(
    signature: &'a GgmlType19Signature,
    packed: &'a [u8],
) -> Result<impl Iterator<Item = Result<Q8_1MmqBlock, String>> + 'a, String> {
    let expected = signature.activation_storage_bytes()?;
    if packed.len() != expected {
        return Err(format!("Q8_1 MMQ storage must be exactly {expected} bytes"));
    }
    Ok(packed.chunks_exact(Q8_1_MMQ_BYTES).map(Q8_1MmqBlock::parse))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(crate) enum ComponentKind {
    Grid,
    Delta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct TileGeometry {
    pub(crate) row_tile: usize,
    pub(crate) column_tile: usize,
    pub(crate) valid_out: usize,
    pub(crate) valid_in: usize,
    pub(crate) group_count: usize,
}

pub(crate) fn plan_tiles(signature: &GgmlType19Signature) -> Result<Vec<TileGeometry>, String> {
    signature.validate()?;
    let rows = usize::try_from(signature.ne0).map_err(|_| "row count overflow")?;
    let columns = usize::try_from(signature.ne00).map_err(|_| "column count overflow")?;
    let mut result = Vec::new();
    for row_start in (0..rows).step_by(TILE_DIM) {
        for column_start in (0..columns).step_by(TILE_DIM) {
            let valid_in = (columns - column_start).min(TILE_DIM);
            result.push(TileGeometry {
                row_tile: row_start / TILE_DIM,
                column_tile: column_start / TILE_DIM,
                valid_out: (rows - row_start).min(TILE_DIM),
                valid_in,
                group_count: valid_in.div_ceil(GROUP_VALUES),
            });
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComponentTile {
    pub(crate) geometry: TileGeometry,
    pub(crate) group32: usize,
    pub(crate) kind: ComponentKind,
    /// Compact row-major valid_out x 32 signed ternary values.
    pub(crate) values: Vec<i8>,
}

impl ComponentTile {
    #[cfg(test)]
    pub(crate) fn from_rows(
        geometry: TileGeometry,
        group32: usize,
        kind: ComponentKind,
        rows: &[Iq1sBlock],
    ) -> Result<Self, String> {
        if geometry.valid_out == 0
            || geometry.valid_out > TILE_DIM
            || geometry.valid_in == 0
            || geometry.valid_in > TILE_DIM
            || rows.len() != geometry.valid_out
            || group32 >= geometry.group_count
        {
            return Err("component tile geometry is invalid".into());
        }
        let global_group = geometry.column_tile * (TILE_DIM / GROUP_VALUES) + group32;
        let group_index = global_group % 8;
        let mut values = Vec::with_capacity(rows.len() * GROUP_VALUES);
        for row in rows {
            let group = row.groups[group_index];
            match kind {
                ComponentKind::Grid => values.extend_from_slice(&group.grid_values),
                ComponentKind::Delta => {
                    values.extend(std::iter::repeat_n(group.delta_sign, GROUP_VALUES))
                }
            }
        }
        Ok(Self {
            geometry,
            group32,
            kind,
            values,
        })
    }

    pub(crate) fn pack_ternary2(&self) -> Result<Vec<u8>, String> {
        if self.values.len() != self.geometry.valid_out * GROUP_VALUES {
            return Err("component payload length does not match valid geometry".into());
        }
        let mut packed = vec![0_u8; TILE_PACKED_BYTES];
        let column_start = self.group32 * GROUP_VALUES;
        for row in 0..self.geometry.valid_out {
            for column in 0..GROUP_VALUES {
                let value = self.values[row * GROUP_VALUES + column];
                let code = match value {
                    -1 => 3,
                    0 => 0,
                    1 => 1,
                    _ => return Err("component is not ternary".into()),
                };
                let element = row * TILE_DIM + column_start + column;
                packed[element / 4] |= code << (2 * (element % 4));
            }
        }
        Ok(packed)
    }
}

pub(crate) fn decompose_component_tiles(
    signature: &GgmlType19Signature,
    packed_matrix: &[u8],
    grid: &GridTable,
) -> Result<Vec<ComponentTile>, String> {
    let expected = signature.matrix_storage_bytes()?;
    if packed_matrix.len() != expected {
        return Err(format!(
            "packed IQ1_S matrix must be exactly {expected} bytes"
        ));
    }
    let row_stride = usize::try_from(signature.stride01)
        .ok()
        .and_then(|blocks| blocks.checked_mul(IQ1S_BLOCK_BYTES))
        .ok_or("IQ1_S row stride overflow")?;
    let logical_blocks = usize::try_from(signature.ne00 / IQ1S_BLOCK_VALUES as u64)
        .map_err(|_| "IQ1_S block count overflow")?;
    let logical_bytes = logical_blocks
        .checked_mul(IQ1S_BLOCK_BYTES)
        .ok_or("IQ1_S logical row bytes overflow")?;
    for row in 0..usize::try_from(signature.ne01).map_err(|_| "row count overflow")? {
        let padding = &packed_matrix[row * row_stride + logical_bytes..(row + 1) * row_stride];
        if padding.iter().any(|byte| *byte != 0) {
            return Err(format!("IQ1_S row {row} padding must be zero"));
        }
    }

    let geometries = plan_tiles(signature)?;
    let mut result = Vec::with_capacity(
        geometries
            .iter()
            .map(|geometry| geometry.group_count * 2)
            .sum(),
    );
    for geometry in geometries {
        let row_start = geometry.row_tile * TILE_DIM;
        for group32 in 0..geometry.group_count {
            let global_group = geometry.column_tile * (TILE_DIM / GROUP_VALUES) + group32;
            let block_index = global_group / 8;
            let group_index = global_group % 8;
            if block_index >= logical_blocks {
                return Err("component group exceeds the logical matrix width".into());
            }
            let mut grid_values = Vec::with_capacity(geometry.valid_out * GROUP_VALUES);
            let mut delta_values = Vec::with_capacity(geometry.valid_out * GROUP_VALUES);
            for row in row_start..row_start + geometry.valid_out {
                let offset = row
                    .checked_mul(row_stride)
                    .and_then(|base| base.checked_add(block_index * IQ1S_BLOCK_BYTES))
                    .ok_or("IQ1_S block offset overflow")?;
                let parsed =
                    Iq1sBlock::parse(&packed_matrix[offset..offset + IQ1S_BLOCK_BYTES], grid)?;
                let group = parsed.groups[group_index];
                grid_values.extend_from_slice(&group.grid_values);
                delta_values.extend(std::iter::repeat_n(group.delta_sign, GROUP_VALUES));
            }
            result.push(ComponentTile {
                geometry,
                group32,
                kind: ComponentKind::Grid,
                values: grid_values,
            });
            result.push(ComponentTile {
                geometry,
                group32,
                kind: ComponentKind::Delta,
                values: delta_values,
            });
        }
    }
    Ok(result)
}

pub(crate) fn raw_component_dots(group: &Iq1sGroup, q8: &Q8_1Block) -> (i64, i64) {
    let grid = group
        .grid_values
        .iter()
        .zip(q8.qs)
        .map(|(&w, q)| i64::from(w) * i64::from(q))
        .sum();
    let delta = i64::from(group.delta_sign) * q8.qs.iter().map(|&q| i64::from(q)).sum::<i64>();
    (grid, delta)
}

pub(crate) fn encode_q8_8(qs: &[i8; 32]) -> [i16; 32] {
    qs.map(|quant| i16::from(quant) << 8)
}

pub(crate) fn validate_raw_q8_8(raw: i64, expected: i64) -> Result<i64, String> {
    if raw % 256 != 0 {
        return Err("raw Q8.8 result is not divisible by 256".into());
    }
    let decoded = raw / 256;
    if decoded != expected {
        return Err(format!(
            "raw Q8.8 mismatch: got {decoded}, expected {expected}"
        ));
    }
    Ok(decoded)
}

pub(crate) fn reconstruct_from_raw(
    group: &Iq1sGroup,
    iq1s_d: f32,
    q8: &Q8_1Block,
    grid_raw: i64,
    delta_raw: i64,
) -> Result<f32, String> {
    if !iq1s_d.is_finite() || !q8.d.is_finite() || !q8.s.is_finite() {
        return Err("reconstruction factors must be finite".into());
    }
    let (expected_grid, expected_delta) = raw_component_dots(group, q8);
    let grid_dot = validate_raw_q8_8(grid_raw, expected_grid)?;
    let delta_dot = validate_raw_q8_8(delta_raw, expected_delta)?;
    let d1q = (iq1s_d * f32::from(group.odd_scale)) as f32;
    let wd = round_to_half(d1q)?;
    let delta_factor = (-1.0_f32 + f32::from(group.delta_sign) * 0.125) as f32;
    let wdelta = round_to_half((d1q * delta_factor) as f32)?;
    let unsigned_grid = grid_dot
        .checked_add(i64::from(group.delta_sign) * delta_dot)
        .ok_or("unsigned grid sum overflow")?;
    let scaled_grid = ((wd * q8.d) as f32 * unsigned_grid as f32) as f32;
    let result = (scaled_grid + (wdelta * q8.s) as f32) as f32;
    if !result.is_finite() {
        return Err("reconstructed MMQ output is not finite".into());
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MatrixCacheIdentity {
    pub(crate) matrix_ptr: usize,
    pub(crate) signature: GgmlType19Signature,
    pub(crate) allocation_generation: u64,
    pub(crate) content_hash: [u8; 32],
}

#[derive(Debug, Default)]
pub(crate) struct ComponentCache {
    identity: Option<MatrixCacheIdentity>,
    components: Option<Arc<[ComponentTile]>>,
}

impl ComponentCache {
    pub(crate) fn matches(&self, identity: &MatrixCacheIdentity) -> bool {
        self.identity.as_ref() == Some(identity)
    }
    pub(crate) fn get(&self, identity: &MatrixCacheIdentity) -> Option<Arc<[ComponentTile]>> {
        self.matches(identity)
            .then(|| self.components.clone())
            .flatten()
    }
    pub(crate) fn insert(
        &mut self,
        identity: MatrixCacheIdentity,
        components: Arc<[ComponentTile]>,
    ) {
        self.identity = Some(identity);
        self.components = Some(components);
    }
    pub(crate) fn invalidate(&mut self) {
        self.identity = None;
        self.components = None;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LogicalLaunch {
    pub(crate) matrix_ptr: usize,
    pub(crate) activation_ptr: usize,
    pub(crate) output_ptr: usize,
    pub(crate) allocation_generation: u64,
    pub(crate) content_hash: [u8; 32],
    pub(crate) signature: GgmlType19Signature,
}

impl LogicalLaunch {
    pub(crate) fn validate_before_copy(&self) -> Result<(), String> {
        self.signature.validate()?;
        if self.matrix_ptr == 0 || self.activation_ptr == 0 || self.output_ptr == 0 {
            return Err("IQ1_S launch contains a null CUDA pointer".into());
        }
        self.signature.matrix_storage_bytes()?;
        self.signature.activation_storage_bytes()?;
        usize::try_from(self.signature.ne0)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or("output size overflow")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CapturedLaunch {
    pub(crate) launch: LogicalLaunch,
    pub(crate) components: Arc<[ComponentTile]>,
    packed_activations: Arc<[u8]>,
}

impl CapturedLaunch {
    pub(crate) fn activation_blocks(
        &self,
    ) -> Result<impl Iterator<Item = Result<Q8_1MmqBlock, String>> + '_, String> {
        iter_q8_1_mmq(&self.launch.signature, &self.packed_activations)
    }
}

static COMPONENT_CACHE: OnceLock<Mutex<ComponentCache>> = OnceLock::new();

pub(crate) fn capture_from_host(
    launch: LogicalLaunch,
    packed_matrix: &[u8],
    packed_activations: &[u8],
    grid: &GridTable,
) -> Result<CapturedLaunch, String> {
    launch.validate_before_copy()?;
    let matrix_bytes = launch.signature.matrix_storage_bytes()?;
    let activation_bytes = launch.signature.activation_storage_bytes()?;
    if packed_matrix.len() != matrix_bytes || packed_activations.len() != activation_bytes {
        return Err("captured CUDA storage length does not match the qualified signature".into());
    }
    // Force a complete finite DS4 validation while retaining the packed bytes;
    // execution can subsequently iterate the groups without a dense activation.
    for block in iter_q8_1_mmq(&launch.signature, packed_activations)? {
        block?;
    }
    let identity = MatrixCacheIdentity {
        matrix_ptr: launch.matrix_ptr,
        signature: launch.signature.clone(),
        allocation_generation: launch.allocation_generation,
        content_hash: launch.content_hash,
    };
    let cache = COMPONENT_CACHE.get_or_init(|| Mutex::new(ComponentCache::default()));
    let cached = {
        let guard = cache.lock().map_err(|_| "component cache poisoned")?;
        guard.get(&identity)
    };
    let components = if let Some(cached) = cached {
        cached
    } else {
        let built: Arc<[ComponentTile]> = Arc::from(decompose_component_tiles(
            &launch.signature,
            packed_matrix,
            grid,
        )?);
        cache
            .lock()
            .map_err(|_| "component cache poisoned")?
            .insert(identity, built.clone());
        built
    };
    Ok(CapturedLaunch {
        launch,
        components,
        packed_activations: Arc::from(packed_activations),
    })
}

pub(crate) unsafe fn capture_launch(launch: LogicalLaunch) -> Result<CapturedLaunch, String> {
    launch.validate_before_copy()?;
    let matrix = copy_cuda_to_host(launch.matrix_ptr, launch.signature.matrix_storage_bytes()?)
        .map_err(|error| error.to_string())?;
    let activations = copy_cuda_to_host(
        launch.activation_ptr,
        launch.signature.activation_storage_bytes()?,
    )
    .map_err(|error| error.to_string())?;
    let grid = validated_grid(None)?;
    capture_from_host(launch, &matrix, &activations, &grid)
}

pub(crate) unsafe fn copy_outputs_to_cuda(
    captured: &CapturedLaunch,
    outputs: &[f32],
) -> Result<(), String> {
    let expected =
        usize::try_from(captured.launch.signature.ne0).map_err(|_| "output count overflow")?;
    if outputs.len() != expected || outputs.iter().any(|value| !value.is_finite()) {
        return Err("output must contain the full finite qualified f32 result".into());
    }
    let mut bytes = Vec::with_capacity(outputs.len().checked_mul(4).ok_or("output size overflow")?);
    for value in outputs {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    copy_host_to_cuda(captured.launch.output_ptr, &bytes).map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TaskLeases {
    pub(crate) matrix: BufferLease,
    pub(crate) input: BufferLease,
    pub(crate) output: BufferLease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhysicalResult {
    pub(crate) kind: ComponentKind,
    pub(crate) row_tile: usize,
    pub(crate) column_tile: usize,
    pub(crate) group32: usize,
    pub(crate) completed: CompletedTaskV3,
}

pub(crate) fn run_component_tasks<I: IoctlOps>(
    session: &mut V3Session<I>,
    leases: TaskLeases,
    components: &[(ComponentKind, TileGeometry, usize, TaskV3)],
) -> Result<Vec<PhysicalResult>, String> {
    let tasks: Vec<TaskV3> = components.iter().map(|item| item.3).collect();
    for task in &tasks {
        if task.matrix_handle != leases.matrix.handle()
            || task.matrix_generation != leases.matrix.generation()
            || task.input_handle != leases.input.handle()
            || task.output_handle != leases.output.handle()
        {
            return Err("planned V3 task does not match committed leases".into());
        }
    }
    let completed = session.run_tasks(&tasks)?;
    if completed.len() != components.len() {
        return Err("V3 returned an incomplete component set".into());
    }
    Ok(components
        .iter()
        .zip(completed)
        .map(|((kind, geometry, group32, _), completed)| PhysicalResult {
            kind: *kind,
            row_tile: geometry.row_tile,
            column_tile: geometry.column_tile,
            group32: *group32,
            completed,
        })
        .collect())
}

pub(crate) fn build_task(
    request_id: u64,
    geometry: TileGeometry,
    leases: TaskLeases,
    output_offset: u64,
    dax_alignment: u64,
) -> Result<TaskV3, String> {
    if dax_alignment == 0 || !dax_alignment.is_power_of_two() || !output_offset.is_multiple_of(8) {
        return Err("invalid DAX/output alignment".into());
    }
    let stride = align_up((TILE_DIM * 2) as u64, dax_alignment)?;
    let mut task = TaskV3::default();
    task.request_id = request_id;
    task.lane = LANE_ANY;
    task.batch = 1;
    task.valid_out = geometry.valid_out as u32;
    task.valid_in = geometry.valid_in as u32;
    task.matrix_handle = leases.matrix.handle();
    task.input_handle = leases.input.handle();
    task.output_handle = leases.output.handle();
    task.matrix_generation = leases.matrix.generation();
    task.input_stride_bytes = stride;
    task.output_offset = output_offset;
    task.output_stride_bytes = align_up((TILE_DIM * 8) as u64, dax_alignment)?;
    Ok(task)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or("alignment overflow".into())
}

#[derive(Serialize)]
struct Fixture<'a> {
    iq1s_hex: String,
    q8_1_mmq_hex: String,
    raw_components: &'a [(i64, i64)],
    outputs_f32_bits: Vec<u32>,
}

pub(crate) fn fixture_json(
    iq1s: &[u8],
    q8: &[u8],
    raw: &[(i64, i64)],
    output: &[f32],
) -> Result<String, String> {
    if output.iter().any(|value| !value.is_finite()) {
        return Err("fixture output is nonfinite".into());
    }
    serde_json::to_string(&Fixture {
        iq1s_hex: hex(iq1s),
        q8_1_mmq_hex: hex(q8),
        raw_components: raw,
        outputs_f32_bits: output.iter().map(|value| value.to_bits()).collect(),
    })
    .map_err(|error| error.to_string())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 15) as usize] as char);
    }
    result
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let f32_bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = fraction.leading_zeros() - 22;
            let normalized = (fraction << (leading + 1)) & 0x03ff;
            sign | ((127 - 15 - leading) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent + (127 - 15)) << 23) | (fraction << 13),
    };
    f32::from_bits(f32_bits)
}

fn f32_to_half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    let half_exp = exponent - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mant = mantissa | 0x80_0000;
        let shift = 14 - half_exp;
        let mut rounded = mant >> shift;
        let remainder = mant & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }
    let mut rounded = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
        rounded += 1;
    }
    let mut exp = half_exp as u16;
    if rounded == 0x400 {
        rounded = 0;
        exp += 1;
    }
    if exp >= 0x1f {
        sign | 0x7c00
    } else {
        sign | (exp << 10) | rounded as u16
    }
}

fn round_to_half(value: f32) -> Result<f32, String> {
    let rounded = half_to_f32(f32_to_half_bits(value));
    if rounded.is_finite() {
        Ok(rounded)
    } else {
        Err("value does not round to finite binary16".into())
    }
}

#[repr(C)]
struct GgmlInitParams {
    mem_size: usize,
    mem_buffer: *mut libc::c_void,
    no_alloc: bool,
}
type DequantizeIq1s = unsafe extern "C" fn(*const libc::c_void, *mut libc::c_void, i64);
type GgmlInit = unsafe extern "C" fn(GgmlInitParams) -> *mut libc::c_void;
type GgmlFree = unsafe extern "C" fn(*mut libc::c_void);

struct OracleLibrary {
    handle: *mut libc::c_void,
    dequantize: DequantizeIq1s,
}
unsafe impl Send for OracleLibrary {}
unsafe impl Sync for OracleLibrary {}
impl Drop for OracleLibrary {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle);
        }
    }
}

static GRID_CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<GridTable>>>> = OnceLock::new();
static GRID_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();

pub(crate) fn validated_grid(path: Option<&Path>) -> Result<Arc<GridTable>, String> {
    let selected = path
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("HETGPU_LIBGGML").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_LIBGGML));
    let path = selected
        .canonicalize()
        .map_err(|error| format!("libggml {}: {error}", selected.display()))?;
    let cache = GRID_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(value) = cache
        .lock()
        .map_err(|_| "grid cache poisoned")?
        .get(&path)
        .cloned()
    {
        return Ok(value);
    }
    let key_lock = {
        let locks = GRID_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
        locks
            .lock()
            .map_err(|_| "grid lock map poisoned")?
            .entry(path.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    };
    let _guard = key_lock.lock().map_err(|_| "grid key lock poisoned")?;
    if let Some(value) = cache
        .lock()
        .map_err(|_| "grid cache poisoned")?
        .get(&path)
        .cloned()
    {
        return Ok(value);
    }
    let oracle = unsafe { OracleLibrary::open(&path)? };
    let table = Arc::new(recover_grid(&oracle)?);
    cache
        .lock()
        .map_err(|_| "grid cache poisoned")?
        .insert(path, table.clone());
    Ok(table)
}

impl OracleLibrary {
    unsafe fn open(path: &Path) -> Result<Self, String> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| "libggml path contains NUL")?;
        let handle = libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if handle.is_null() {
            return Err(format!("dlopen libggml failed: {}", dlerror()));
        }
        let result = (|| {
            let dequantize = symbol::<DequantizeIq1s>(handle, b"dequantize_row_iq1_s\0")?;
            let init = symbol::<GgmlInit>(handle, b"ggml_init\0")?;
            let free = symbol::<GgmlFree>(handle, b"ggml_free\0")?;
            let context = init(GgmlInitParams {
                mem_size: 0,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            });
            if context.is_null() {
                return Err("ggml_init failed".into());
            }
            free(context);
            Ok(Self { handle, dequantize })
        })();
        if result.is_err() {
            libc::dlclose(handle);
        }
        result
    }
}

unsafe fn symbol<T: Copy>(handle: *mut libc::c_void, name: &'static [u8]) -> Result<T, String> {
    let pointer = libc::dlsym(handle, name.as_ptr().cast());
    if pointer.is_null() {
        Err(format!(
            "libggml missing symbol {}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        ))
    } else {
        Ok(std::mem::transmute_copy(&pointer))
    }
}

unsafe fn dlerror() -> String {
    let error = libc::dlerror();
    if error.is_null() {
        "unknown loader error".into()
    } else {
        CStr::from_ptr(error).to_string_lossy().into_owned()
    }
}

fn recover_grid(oracle: &OracleLibrary) -> Result<GridTable, String> {
    let mut table = [[0_i8; 8]; GRID_ENTRIES];
    let mut packed = [0_u8; IQ1S_BLOCK_BYTES];
    packed[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let mut output = [0_f32; IQ1S_BLOCK_VALUES];
    for (index, entry) in table.iter_mut().enumerate() {
        packed[2..].fill(0);
        packed[2] = index as u8;
        packed[34..36].copy_from_slice(&((index >> 8) as u16).to_le_bytes());
        output.fill(f32::NAN);
        unsafe {
            (oracle.dequantize)(
                packed.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                IQ1S_BLOCK_VALUES as i64,
            );
        }
        for (dst, &value) in entry.iter_mut().zip(&output[..8]) {
            let candidate = value - 0.125;
            if !candidate.is_finite() || !matches!(candidate, -1.0 | 0.0 | 1.0) {
                return Err(format!(
                    "libggml oracle returned invalid grid value for index {index}"
                ));
            }
            *dst = candidate as i8;
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> GridTable {
        let mut entries = [[0_i8; 8]; GRID_ENTRIES];
        for (index, entry) in entries.iter_mut().enumerate() {
            for (column, value) in entry.iter_mut().enumerate() {
                *value = match (index + column) % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
            }
        }
        entries
    }

    fn block(odd: u8, negative: bool, high_index: u16) -> [u8; IQ1S_BLOCK_BYTES] {
        let mut packed = [0_u8; IQ1S_BLOCK_BYTES];
        packed[..2].copy_from_slice(&0x3c00_u16.to_le_bytes()); // half(1)
        packed[2] = high_index as u8;
        let qh = ((u16::from(odd) - 1) / 2) << 12
            | if negative { 0x8000 } else { 0 }
            | ((high_index >> 8) & 7);
        packed[34..36].copy_from_slice(&qh.to_le_bytes());
        packed
    }

    fn kimi_signature() -> GgmlType19Signature {
        GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 7168,
            ne01: 2048,
            stride01: 28,
            ne10: 7168,
            ne11: 1,
            stride11: 1,
            ne0: 2048,
        }
    }

    #[test]
    fn validates_real_kimi_block_unit_signature() {
        let signature = kimi_signature();
        assert_eq!(signature.validate().unwrap(), signature);
        let mut obsolete = signature.clone();
        obsolete.stride01 = 1400;
        obsolete.stride11 = 8064;
        assert!(obsolete.validate().unwrap_err().contains("stride11"));
    }

    #[test]
    fn decomposes_all_odd_scales_both_signs_and_high_grid_bits() {
        let grid = grid();
        for odd in [1_u8, 3, 5, 7, 9, 11, 13, 15] {
            for negative in [false, true] {
                let parsed = Iq1sBlock::parse(&block(odd, negative, 0x7ff), &grid).unwrap();
                assert_eq!(parsed.groups[0].odd_scale, odd);
                assert_eq!(parsed.groups[0].delta_sign, if negative { -1 } else { 1 });
                assert_eq!(parsed.groups[0].grid_indices[0], 0x7ff);
                assert_eq!(parsed.groups[0].grid_values[..8], grid[0x7ff]);
            }
        }
    }

    #[test]
    fn parses_exact_144_byte_ds4_and_rejects_bad_storage() {
        let mut packed = [0_u8; Q8_1_MMQ_BYTES];
        for pair in 0..4 {
            packed[pair * 4..pair * 4 + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
            packed[pair * 4 + 2..pair * 4 + 4].copy_from_slice(&0x3e00_u16.to_le_bytes());
            packed[16 + pair * 32..16 + (pair + 1) * 32].fill((pair + 1) as u8);
        }
        let parsed = Q8_1MmqBlock::parse(&packed).unwrap();
        assert_eq!((parsed.subblocks[0].d, parsed.subblocks[0].s), (0.5, 1.5));
        assert_eq!(parsed.subblocks[3].qs, [4; 32]);
        assert!(Q8_1MmqBlock::parse(&packed[..143]).is_err());
        packed[0..2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        assert!(Q8_1MmqBlock::parse(&packed).unwrap_err().contains("finite"));
    }

    #[test]
    fn real_kimi_geometry_has_four_edge_tiles_and_compact_groups() {
        let tiles = plan_tiles(&kimi_signature()).unwrap();
        assert_eq!(tiles.len(), 4);
        assert_eq!((tiles[3].valid_out, tiles[3].valid_in), (2048, 1024));
        assert_eq!(
            tiles.iter().map(|tile| tile.group_count).sum::<usize>(),
            224
        );

        let parsed = Iq1sBlock::parse(&block(15, true, 0x7ff), &grid()).unwrap();
        let component = ComponentTile::from_rows(
            TileGeometry {
                row_tile: 0,
                column_tile: 0,
                valid_out: 1,
                valid_in: 256,
                group_count: 8,
            },
            0,
            ComponentKind::Delta,
            &[parsed],
        )
        .unwrap();
        assert_eq!(component.values.len(), 32);
        assert_eq!(component.values, vec![-1; 32]);
        let full = component.pack_ternary2().unwrap();
        assert_eq!(full.len(), TILE_PACKED_BYTES);
        assert!(full[8..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn validates_both_signed_raw_components_and_mmq_half_boundary() {
        let group = Iq1sGroup {
            odd_scale: 15,
            delta_sign: -1,
            grid_indices: [0; 4],
            grid_values: [
                1, 0, -1, 1, -1, 0, 1, -1, 1, 0, -1, 1, -1, 0, 1, -1, 1, 0, -1, 1, -1, 0, 1, -1, 1,
                0, -1, 1, -1, 0, 1, -1,
            ],
        };
        let mut qs = [0_i8; 32];
        for (index, quant) in qs.iter_mut().enumerate() {
            *quant = index as i8 - 16;
        }
        let q8 = Q8_1Block {
            d: -0.1875,
            s: 1.5,
            qs,
        };
        let (grid_dot, delta_dot) = raw_component_dots(&group, &q8);
        assert_eq!(
            validate_raw_q8_8(grid_dot << 8, grid_dot).unwrap(),
            grid_dot
        );
        assert_eq!(
            validate_raw_q8_8(delta_dot << 8, delta_dot).unwrap(),
            delta_dot
        );
        assert!(validate_raw_q8_8((grid_dot << 8) + 1, grid_dot).is_err());
        let result =
            reconstruct_from_raw(&group, 0.333251953125, &q8, grid_dot << 8, delta_dot << 8)
                .unwrap();
        assert_eq!(result.to_bits(), 0x41ac8000);
    }

    #[test]
    fn cache_identity_changes_on_every_bound_field() {
        let base = MatrixCacheIdentity {
            matrix_ptr: 0x1000,
            signature: kimi_signature(),
            allocation_generation: 7,
            content_hash: [0x55; 32],
        };
        let mut cache = ComponentCache::default();
        assert!(!cache.matches(&base));
        let payload: Arc<[ComponentTile]> = Arc::from(Vec::<ComponentTile>::new());
        cache.insert(base.clone(), payload.clone());
        assert!(cache.matches(&base));
        assert!(Arc::ptr_eq(&cache.get(&base).unwrap(), &payload));
        let mut changed = base.clone();
        changed.matrix_ptr += 1;
        assert!(!cache.matches(&changed));
        let mut changed = base.clone();
        changed.signature.ne0 += 1;
        assert!(!cache.matches(&changed));
        let mut changed = base.clone();
        changed.allocation_generation += 1;
        assert!(!cache.matches(&changed));
        let mut changed = base;
        changed.content_hash[0] ^= 1;
        assert!(!cache.matches(&changed));
    }

    #[test]
    fn fixture_json_contains_packed_inputs_raw_parts_and_f32_bits() {
        let json = fixture_json(&[1, 2], &[3, 4], &[(256, -512)], &[1.5]).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["iq1s_hex"], "0102");
        assert_eq!(value["q8_1_mmq_hex"], "0304");
        assert_eq!(value["raw_components"][0][1], -512);
        assert_eq!(value["outputs_f32_bits"][0], 1.5_f32.to_bits());
    }

    #[test]
    fn matrix_decomposition_selects_each_physical_iq1s_block_without_dense_float() {
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 2304,
            ne01: 1,
            stride01: 9,
            ne10: 2304,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let mut matrix = Vec::new();
        for index in 0..9_u16 {
            matrix.extend_from_slice(&block(1, index == 1, index));
        }
        let components = decompose_component_tiles(&signature, &matrix, &grid()).unwrap();
        assert_eq!(components.len(), 144);
        let selected = components
            .iter()
            .find(|tile| {
                tile.geometry.column_tile == 0
                    && tile.group32 == 8
                    && tile.kind == ComponentKind::Delta
            })
            .unwrap();
        assert_eq!(selected.values, vec![-1; 32]);
        let edge = components.last().unwrap();
        assert_eq!(
            (edge.geometry.column_tile, edge.geometry.valid_in),
            (1, 256)
        );
        assert!(
            decompose_component_tiles(&signature, &matrix[..matrix.len() - 1], &grid()).is_err()
        );
    }

    #[test]
    fn host_capture_validates_ds4_and_reuses_only_the_full_matrix_identity() {
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 256,
            ne01: 1,
            stride01: 1,
            ne10: 256,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let launch = LogicalLaunch {
            matrix_ptr: 0x1000,
            activation_ptr: 0x2000,
            output_ptr: 0x3000,
            allocation_generation: 1,
            content_hash: [7; 32],
            signature,
        };
        let matrix = block(3, false, 0x700);
        let activations = vec![0_u8; 2 * Q8_1_MMQ_BYTES];
        let first = capture_from_host(launch.clone(), &matrix, &activations, &grid()).unwrap();
        let second = capture_from_host(launch.clone(), &matrix, &activations, &grid()).unwrap();
        assert!(Arc::ptr_eq(&first.components, &second.components));
        assert_eq!(first.activation_blocks().unwrap().count(), 2);

        let mut changed = launch;
        changed.content_hash[0] ^= 1;
        let third = capture_from_host(changed, &matrix, &activations, &grid()).unwrap();
        assert!(!Arc::ptr_eq(&first.components, &third.components));
        assert!(
            capture_from_host(first.launch.clone(), &matrix, &activations[..287], &grid()).is_err()
        );
    }
}
