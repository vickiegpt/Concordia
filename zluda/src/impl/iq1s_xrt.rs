//! AU250 D=1024 planning and execution for captured IQ1_S launches.

use super::iq1s_tmatmul::{ComponentKind, GgmlType19Signature, MatrixSource, Q8_1Block};
use std::collections::HashSet;

pub(crate) const AU250_DIM: usize = 1024;
pub(crate) const AU250_GROUP_VALUES: usize = 32;
pub(crate) const AU250_GROUPS_PER_K_TILE: usize = 32;
pub(crate) const AU250_MATRIX_BYTES: usize = AU250_DIM * AU250_DIM / 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Au250Tile {
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) valid_out: usize,
    pub(crate) valid_in: usize,
    pub(crate) group_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Au250MatrixKey {
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) kind: ComponentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaneAssignment {
    pub(crate) lane: usize,
    pub(crate) batch_index: usize,
    pub(crate) global_group: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedAu250Job {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) matrix_key: Au250MatrixKey,
    pub(crate) assignments: Vec<LaneAssignment>,
}

pub(crate) fn plan_au250_tiles(signature: &GgmlType19Signature) -> Result<Vec<Au250Tile>, String> {
    signature.validate()?;
    let rows = usize::try_from(signature.ne0).map_err(|_| "row count does not fit usize")?;
    let columns = usize::try_from(signature.ne00).map_err(|_| "column count does not fit usize")?;
    let mut result = Vec::new();
    for row_start in (0..rows).step_by(AU250_DIM) {
        for k_start in (0..columns).step_by(AU250_DIM) {
            let valid_in = (columns - k_start).min(AU250_DIM);
            result.push(Au250Tile {
                row_tile: row_start / AU250_DIM,
                k_tile: k_start / AU250_DIM,
                valid_out: (rows - row_start).min(AU250_DIM),
                valid_in,
                group_count: valid_in.div_ceil(AU250_GROUP_VALUES),
            });
        }
    }
    Ok(result)
}

pub(crate) fn pack_component_matrix(
    source: &MatrixSource,
    tile: Au250Tile,
    kind: ComponentKind,
) -> Result<Vec<u8>, String> {
    if tile.valid_out == 0
        || tile.valid_out > AU250_DIM
        || tile.valid_in == 0
        || tile.valid_in > AU250_DIM
        || tile.group_count != tile.valid_in.div_ceil(AU250_GROUP_VALUES)
        || tile.group_count > AU250_GROUPS_PER_K_TILE
    {
        return Err("invalid AU250 tile geometry".to_string());
    }
    let row_start = tile
        .row_tile
        .checked_mul(AU250_DIM)
        .ok_or("AU250 row tile overflow")?;
    let group_start = tile
        .k_tile
        .checked_mul(AU250_GROUPS_PER_K_TILE)
        .ok_or("AU250 K tile overflow")?;
    let mut packed = vec![0u8; AU250_MATRIX_BYTES];
    for local_row in 0..tile.valid_out {
        let global_row = row_start
            .checked_add(local_row)
            .ok_or("AU250 row coordinate overflow")?;
        for local_group in 0..tile.group_count {
            let global_group = group_start
                .checked_add(local_group)
                .ok_or("AU250 group coordinate overflow")?;
            let (_, group) = source.group(global_row, global_group)?;
            for position in 0..AU250_GROUP_VALUES {
                let value = match kind {
                    ComponentKind::Grid => group.grid_values[position],
                    ComponentKind::Delta => group.delta_sign,
                };
                let code = match value {
                    -1 => 3u8,
                    0 => 0u8,
                    1 => 1u8,
                    _ => return Err(format!("component value {value} is not ternary")),
                };
                let column = local_group * AU250_GROUP_VALUES + position;
                if column >= tile.valid_in {
                    continue;
                }
                let element = local_row * AU250_DIM + column;
                packed[element / 4] |= code << (2 * (element % 4));
            }
        }
    }
    Ok(packed)
}

pub(crate) fn pack_lane_input(
    lanes: usize,
    assignments: &[(usize, usize, Q8_1Block)],
) -> Result<Vec<u8>, String> {
    if lanes == 0 {
        return Err("AU250 lane count must be positive".to_string());
    }
    let elements = AU250_DIM
        .checked_mul(lanes)
        .ok_or("AU250 lane input element count overflow")?;
    let byte_count = elements
        .checked_mul(std::mem::size_of::<i16>())
        .ok_or("AU250 lane input byte count overflow")?;
    let mut bytes = vec![0u8; byte_count];
    let mut used_lanes = HashSet::new();
    for &(lane, group_slot, q8) in assignments {
        if lane >= lanes {
            return Err(format!("AU250 lane {lane} is outside {lanes} lanes"));
        }
        if !used_lanes.insert(lane) {
            return Err(format!("duplicate AU250 lane assignment {lane}"));
        }
        if group_slot >= AU250_GROUPS_PER_K_TILE {
            return Err(format!(
                "AU250 group slot {group_slot} is outside one K tile"
            ));
        }
        if !q8.d.is_finite() || !q8.s.is_finite() {
            return Err("Q8 lane assignment has non-finite factors".to_string());
        }
        for (position, quant) in q8.qs.into_iter().enumerate() {
            let dimension = group_slot * AU250_GROUP_VALUES + position;
            let element = dimension * lanes + lane;
            let offset = element * std::mem::size_of::<i16>();
            bytes[offset..offset + 2].copy_from_slice(&i16::from(quant).to_le_bytes());
        }
    }
    Ok(bytes)
}

pub(crate) fn raw_dot_bounds() -> (i16, i16) {
    (-4096, 4096)
}

pub(crate) fn plan_au250_jobs(
    signature: &GgmlType19Signature,
    lane_capacities: &[usize],
) -> Result<Vec<Vec<PlannedAu250Job>>, String> {
    if lane_capacities.is_empty() || lane_capacities.iter().any(|capacity| *capacity == 0) {
        return Err("AU250 CU lane capacities must be nonempty and positive".to_string());
    }
    let batch_count =
        usize::try_from(signature.ne11).map_err(|_| "batch count does not fit usize")?;
    let mut waves = Vec::new();
    let mut next_request_id = 0u64;
    for tile in plan_au250_tiles(signature)? {
        let group_start = tile
            .k_tile
            .checked_mul(AU250_GROUPS_PER_K_TILE)
            .ok_or("AU250 group coordinate overflow")?;
        let assignment_count = batch_count
            .checked_mul(tile.group_count)
            .ok_or("AU250 assignment count overflow")?;
        for kind in [ComponentKind::Grid, ComponentKind::Delta] {
            let matrix_key = Au250MatrixKey {
                row_tile: tile.row_tile,
                k_tile: tile.k_tile,
                kind,
            };
            let mut cursor = 0usize;
            while cursor < assignment_count {
                let mut wave = Vec::new();
                for (cu_index, &capacity) in lane_capacities.iter().enumerate() {
                    if cursor == assignment_count {
                        break;
                    }
                    let take = capacity.min(assignment_count - cursor);
                    let mut assignments = Vec::with_capacity(take);
                    for lane in 0..take {
                        let ordinal = cursor
                            .checked_add(lane)
                            .ok_or("AU250 assignment ordinal overflow")?;
                        let batch_index = ordinal / tile.group_count;
                        let local_group = ordinal % tile.group_count;
                        let global_group = group_start
                            .checked_add(local_group)
                            .ok_or("AU250 global group overflow")?;
                        assignments.push(LaneAssignment {
                            lane,
                            batch_index,
                            global_group,
                        });
                    }
                    let request_id = next_request_id;
                    next_request_id = next_request_id
                        .checked_add(1)
                        .ok_or("AU250 request ID overflow")?;
                    wave.push(PlannedAu250Job {
                        request_id,
                        cu_index,
                        matrix_key,
                        assignments,
                    });
                    cursor += take;
                }
                if wave.is_empty() {
                    return Err("AU250 planner made no progress".to_string());
                }
                validate_planned_wave(&wave, lane_capacities, tile, batch_count)?;
                waves.push(wave);
            }
        }
    }
    Ok(waves)
}

fn validate_planned_wave(
    wave: &[PlannedAu250Job],
    lane_capacities: &[usize],
    tile: Au250Tile,
    batch_count: usize,
) -> Result<(), String> {
    let mut cu_indices = HashSet::new();
    for job in wave {
        if !cu_indices.insert(job.cu_index) {
            return Err(format!("duplicate CU {} in AU250 wave", job.cu_index));
        }
        let capacity = *lane_capacities
            .get(job.cu_index)
            .ok_or("AU250 planned job selects an unknown CU")?;
        let mut lanes = HashSet::new();
        for assignment in &job.assignments {
            if assignment.lane >= capacity || !lanes.insert(assignment.lane) {
                return Err(format!(
                    "invalid or duplicate lane {} for CU {}",
                    assignment.lane, job.cu_index
                ));
            }
            let local_group = assignment
                .global_group
                .checked_sub(tile.k_tile * AU250_GROUPS_PER_K_TILE)
                .ok_or("AU250 assignment group precedes its K tile")?;
            if local_group >= tile.group_count || assignment.batch_index >= batch_count {
                return Err("AU250 assignment is outside its tile or batch".to_string());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::iq1s_tmatmul::{ComponentKind, GridTable, MatrixSource};
    use super::*;
    use std::sync::Arc;

    fn kimi_signature() -> GgmlType19Signature {
        GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 7168,
            ne01: 2048,
            stride01: 28,
            ne10: 7168,
            ne11: 1,
            stride11: 8064,
            ne0: 2048,
        }
    }

    #[test]
    fn kimi_7168_by_2048_uses_seven_k_tiles_and_two_row_tiles() {
        let geometry = plan_au250_tiles(&kimi_signature()).unwrap();
        assert_eq!(geometry.iter().map(|tile| tile.k_tile).max(), Some(6));
        assert_eq!(geometry.iter().map(|tile| tile.row_tile).max(), Some(1));
        assert_eq!(geometry.last().unwrap().valid_in, 1024);
    }

    #[test]
    fn lane_input_is_dimension_major_and_group_sparse() {
        let q8 = Q8_1Block {
            d: 1.0,
            s: 0.0,
            qs: [7; 32],
        };
        let bytes = pack_lane_input(9, &[(3, 5, q8)]).unwrap();
        let raw = bytes
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(raw[5 * 32 * 9 + 3], 7);
        assert_eq!(raw.iter().filter(|value| **value == 7).count(), 32);
    }

    #[test]
    fn direct_q8_raw_values_keep_every_group_dot_in_i16() {
        assert_eq!(raw_dot_bounds(), (-4096, 4096));
    }

    #[test]
    fn component_matrix_packs_grid_and_delta_trits_with_zero_padding() {
        let signature = GgmlType19Signature {
            kernel: "mul_mat_q".into(),
            ne00: 1024,
            ne01: 1,
            stride01: 4,
            ne10: 1024,
            ne11: 1,
            stride11: 1,
            ne0: 1,
        };
        let mut grid: GridTable = [[0; 8]; 2048];
        grid[0] = [-1, 0, 1, -1, 0, 1, -1, 1];
        let source = MatrixSource::new(
            signature.clone(),
            Arc::from(vec![0_u8; signature.matrix_storage_bytes().unwrap()]),
            Arc::new(grid),
        )
        .unwrap();
        let tile = plan_au250_tiles(&signature).unwrap()[0];

        let packed_grid = pack_component_matrix(&source, tile, ComponentKind::Grid).unwrap();
        let first_group = (0..32)
            .map(|index| decode_trit(&packed_grid, index))
            .collect::<Vec<_>>();
        assert_eq!(first_group, [-1, 0, 1, -1, 0, 1, -1, 1].repeat(4));
        assert!(packed_grid[AU250_DIM / 4..].iter().all(|byte| *byte == 0));

        let packed_delta = pack_component_matrix(&source, tile, ComponentKind::Delta).unwrap();
        assert!((0..32).all(|index| decode_trit(&packed_delta, index) == 1));
    }

    #[test]
    fn jobs_are_batch_major_and_fill_each_cu_once_per_wave() {
        let mut signature = kimi_signature();
        signature.ne00 = 1024;
        signature.ne10 = 1024;
        signature.ne01 = 1024;
        signature.ne0 = 1024;
        signature.stride01 = 4;
        signature.ne11 = 2;
        signature.stride11 = 2;
        let waves = plan_au250_jobs(&signature, &[9, 9, 9, 6]).unwrap();
        let first = &waves[0];
        assert_eq!(first.len(), 4);
        assert_eq!(
            first[0]
                .assignments
                .iter()
                .map(|assignment| (
                    assignment.lane,
                    assignment.batch_index,
                    assignment.global_group,
                ))
                .collect::<Vec<_>>(),
            (0..9).map(|group| (group, 0, group)).collect::<Vec<_>>()
        );
        assert_eq!(first[3].assignments.len(), 6);
        assert_eq!(first[3].cu_index, 3);
        assert!(waves.iter().all(|wave| wave
            .iter()
            .map(|job| job.cu_index)
            .collect::<std::collections::HashSet<_>>()
            .len()
            == wave.len()));
    }

    fn decode_trit(packed: &[u8], element: usize) -> i8 {
        match (packed[element / 4] >> (2 * (element % 4))) & 3 {
            0 => 0,
            1 => 1,
            3 => -1,
            code => panic!("invalid ternary code {code}"),
        }
    }
}
