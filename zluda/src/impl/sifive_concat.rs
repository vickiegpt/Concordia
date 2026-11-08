use super::*;
pub(super) unsafe fn reduce(
    name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let run = (|| -> Option<bool> {
        let mean = match name {
            "_ZL15reduce_rows_f32ILb0EEvPKfPfi" => false,
            "_ZL15reduce_rows_f32ILb1EEvPKfPfi" => true,
            _ => return None,
        };
        if grid[0] == 0
            || grid[1..] != [1, 1]
            || !matches!(block[0], 32 | 128 | 512)
            || block[1..] != [1, 1]
        {
            return None;
        }
        let x = read_param_u64(params, 0)?;
        let out = read_param_u64(params, 1)?;
        let cols = usize::try_from(read_param_i32(params, 2)?).ok()?;
        let rows = grid[0] as usize;
        if cols == 0 {
            return None;
        }
        let rb = cols.checked_mul(4)?;
        if !sifive_host_or_cuda_alloc_has_bytes(x, rows.checked_mul(rb)?, false)
            || !sifive_host_or_cuda_alloc_has_bytes(out, rows.checked_mul(4)?, true)
        {
            return None;
        }
        let chunk = sifive_runtime_sys::q4k_tile::payload_capacity().checked_sub(65536 + 8)?
            / rb.checked_add(4)?;
        if chunk == 0 {
            return None;
        }
        for row in (0..rows).step_by(chunk) {
            if sifive_runtime_sys::concat_tile::reduce_rows(
                current_sifive_device_id_or_zero(),
                (x as *const f32).add(row * cols),
                (out as *mut f32).add(row),
                cols,
                (rows - row).min(chunk),
                block[0],
                mean,
            ) != 0
            {
                return Some(false);
            }
        }
        Some(true)
    })();
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(cuda_types::cuda::CUerror::UNKNOWN),
        None => Err(cuda_types::cuda::CUerror::NOT_SUPPORTED),
    }
}
pub(super) unsafe fn launch(
    name: &str,
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let run = (|| -> Option<bool> {
        let suffix = name.strip_prefix("_ZL11concat_contI")?;
        let elem = match suffix.as_bytes().first()? {
            b'j' | b'f' => 4,
            b't' => 2,
            b'm' => 8,
            _ => return None,
        };
        if !suffix[1..].starts_with("Li0EE") {
            return None;
        }
        let x = read_param_u64(params, 0)?;
        let y = read_param_u64(params, 1)?;
        let out = read_param_u64(params, 2)?;
        let rd = |i| usize::try_from(read_param_i64(params, i)?).ok();
        let ax = rd(3)?;
        let ay = rd(4)?;
        let az = rd(5)?;
        let dx = rd(6)?;
        let dy = rd(7)?;
        let dz = rd(8)?;
        if ax == 0 || ax >= dx || ay != dy || az != dz || dy == 0 || dz == 0 {
            return None;
        }
        let xb = ax.checked_mul(elem)?;
        let yb = (dx - ax).checked_mul(elem)?;
        let rows = dy.checked_mul(dz)?;
        let rb = xb.checked_add(yb)?;
        for (ptr, size, write) in [
            (x, rows.checked_mul(xb)?, false),
            (y, rows.checked_mul(yb)?, false),
            (out, rows.checked_mul(rb)?, true),
        ] {
            if !sifive_host_or_cuda_alloc_has_bytes(ptr, size, write) {
                return None;
            }
        }
        let limit = sifive_runtime_sys::q4k_tile::payload_capacity().checked_sub(65536 + 8)? / 2;
        let mut tiles = 0;
        if rb <= limit {
            let chunk = limit / rb;
            for row in (0..rows).step_by(chunk) {
                let n = (rows - row).min(chunk);
                if sifive_runtime_sys::concat_tile::submit(
                    current_sifive_device_id_or_zero(),
                    (x as *const u8).add(row * xb),
                    (y as *const u8).add(row * yb),
                    (out as *mut u8).add(row * rb),
                    xb,
                    yb,
                    n,
                ) != 0
                {
                    return Some(false);
                }
                tiles += 1;
            }
        } else {
            for row in 0..rows {
                for (src, bytes, dst) in [
                    (x as *const u8, xb, row * rb),
                    (y as *const u8, yb, row * rb + xb),
                ] {
                    for start in (0..bytes).step_by(limit) {
                        let n = (bytes - start).min(limit);
                        let src = src.add(row * bytes + start);
                        if sifive_runtime_sys::concat_tile::submit(
                            current_sifive_device_id_or_zero(),
                            src,
                            src,
                            (out as *mut u8).add(dst + start),
                            n,
                            0,
                            1,
                        ) != 0
                        {
                            return Some(false);
                        }
                        tiles += 1;
                    }
                }
            }
        }
        eprintln!(
            "PACC concat device-tiled rows={} row_bytes={} tiles={} completion=verified",
            rows, rb, tiles
        );
        Some(true)
    })();
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(cuda_types::cuda::CUerror::UNKNOWN),
        None => Err(cuda_types::cuda::CUerror::NOT_SUPPORTED),
    }
}
