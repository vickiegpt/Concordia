use super::*;
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    heads: i64,
    head_log2: u32,
    cols: i64,
    rows_x: i64,
    rows_y: i64,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb11: i64,
    nb12: i64,
    nb13: i64,
    ne12: i64,
    ne13: i64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
}
pub(super) unsafe fn launch(
    name: &str,
    grid: [u32; 3],
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let run = (|| -> Option<bool> {
        if !name.ends_with("15soft_max_params") || params.is_null() {
            return None;
        }
        let half = name.contains("E6__halfEv");
        if !half && !name.contains("EfEv") {
            return None;
        }
        let pp = *params.add(4) as *const Params;
        if pp.is_null() {
            return None;
        }
        let p = pp.read_unaligned();
        if p.cols <= 0
            || p.cols > 32768
            || p.ne00 != p.cols
            || [p.ne01, p.ne02, p.ne03] != grid.map(|v| v as i64)
            || grid.iter().any(|&v| v == 0)
            || !p.scale.is_finite()
            || !p.max_bias.is_finite()
        {
            return None;
        }
        let n = p.cols as usize;
        let rows = (grid[0] as usize)
            .checked_mul(grid[1] as usize)?
            .checked_mul(grid[2] as usize)?;
        let x = read_param_u64(params, 0)?;
        let mask = read_param_u64(params, 1)?;
        let sink = read_param_u64(params, 2)?;
        let dst = read_param_u64(params, 3)?;
        let bytes = rows.checked_mul(n)?.checked_mul(4)?;
        if !sifive_host_or_cuda_alloc_has_bytes(x, bytes, false)
            || !sifive_host_or_cuda_alloc_has_bytes(dst, bytes, true)
        {
            return None;
        }
        if sink != 0 && !sifive_host_or_cuda_alloc_has_bytes(sink, grid[1] as usize * 4, false) {
            return None;
        }
        if mask != 0 && (p.ne12 <= 0 || p.ne13 <= 0 || p.nb11 < 0 || p.nb12 < 0 || p.nb13 < 0) {
            return None;
        }
        let mut masks = Vec::with_capacity(rows);
        let mut slopes = Vec::with_capacity(rows);
        let mut sinks = Vec::with_capacity(rows);
        for sample in 0..grid[2] {
            for head in 0..grid[1] {
                for row in 0..grid[0] {
                    let a = if mask == 0 {
                        0
                    } else {
                        let off = (row as u64)
                            .checked_mul(p.nb11 as u64)?
                            .checked_add((head as u64 % p.ne12 as u64).checked_mul(p.nb12 as u64)?)?
                            .checked_add(
                                (sample as u64 % p.ne13 as u64).checked_mul(p.nb13 as u64)?,
                            )?;
                        let a = mask.checked_add(off)?;
                        if !sifive_host_or_cuda_alloc_has_bytes(
                            a,
                            n * if half { 2 } else { 4 },
                            false,
                        ) {
                            return None;
                        }
                        a
                    };
                    let slope = if p.max_bias <= 0. {
                        1.
                    } else if head < p.head_log2 {
                        p.m0.powi((head + 1) as i32)
                    } else {
                        p.m1.powi((2 * (head - p.head_log2) + 1) as i32)
                    };
                    if !slope.is_finite() {
                        return None;
                    }
                    masks.push(a);
                    slopes.push(slope);
                    sinks.push(if sink != 0 {
                        (sink as *const f32).add(head as usize).read_unaligned()
                    } else {
                        0.
                    });
                }
            }
        }
        let tile_rows = (sifive_runtime_sys::kernel_payload_capacity().checked_sub(65536)? / ((3 * n + 4) * 4)).min(1024);
        if tile_rows == 0 {
            return None;
        }
        for row0 in (0..rows).step_by(tile_rows) {
            let r = (rows - row0).min(tile_rows);
            let mut input = vec![0f32; r * (2 * n + 4)];
            let mut out = vec![f32::NAN; r * n];
            for row in 0..r {
                let i = row0 + row;
                let buf = &mut input[row * (2 * n + 4)..(row + 1) * (2 * n + 4)];
                buf[..4].copy_from_slice(&[
                    p.scale,
                    slopes[i],
                    sinks[i],
                    if sink != 0 { 1. } else { 0. },
                ]);
                buf[4..4 + n]
                    .copy_from_slice(std::slice::from_raw_parts((x as *const f32).add(i * n), n));
                if masks[i] != 0 {
                    for col in 0..n {
                        buf[4 + n + col] = if half {
                            sifive_f16_to_f32((masks[i] as *const u16).add(col).read_unaligned())
                        } else {
                            (masks[i] as *const f32).add(col).read_unaligned()
                        };
                    }
                }
            }
            if sifive_runtime_sys::attention_softmax::hetgpu_sifive_submit_attention_softmax(
                current_sifive_device_id_or_zero(),
                input.as_ptr(),
                out.as_mut_ptr(),
                r as u32,
                n as u32,
            ) != 0
            {
                return Some(false);
            }
            std::ptr::copy_nonoverlapping(out.as_ptr(), (dst as *mut f32).add(row0 * n), r * n);
        }
        eprintln!(
            "PACC attention softmax rows={} cols={} mask={} sinks={} completion=verified",
            rows,
            n,
            mask != 0,
            sink != 0
        );
        Some(true)
    })();
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(cuda_types::cuda::CUerror::UNKNOWN),
        None => Err(cuda_types::cuda::CUerror::NOT_SUPPORTED),
    }
}
