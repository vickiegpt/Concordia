use super::*;

pub(super) unsafe fn launch(
    name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let mut stage = "template";
    let run = (|| -> Option<bool> {
        let (suffix, qtype) =
            if let Some(s) = name.strip_prefix("_ZL13mul_mat_vec_qIL9ggml_type8ELi") {
                (s, 8)
            } else if let Some(s) = name.strip_prefix("_ZL13mul_mat_vec_qIL9ggml_type12ELi") {
                (s, 4)
            } else {
                (name.strip_prefix("_ZL13mul_mat_vec_qIL9ggml_type13ELi")?, 5)
            };
        let block_bytes = if qtype == 8 {
            34
        } else if qtype == 5 {
            176
        } else {
            144
        };
        let block_k = if qtype == 8 { 32 } else { 256 };
        let (cols, rest) = suffix.split_once("ELb")?;
        let n = cols.parse::<usize>().ok()?;
        let fusion = rest.starts_with('1');
        let small = rest.contains("ELb1ELb0EEv");
        if n != 1
            || (!small && !rest.contains("ELb0ELb0EEv"))
            || grid[1] == 0
            || grid[1] > 256
            || grid[2] != 1
        {
            return None;
        }
        let x = read_param_u64(params, 0)?;
        let y = read_param_u64(params, 1)?;
        let ids = read_param_u64(params, 2)?;
        let dst = read_param_u64(params, 4)?;
        let rd = |i| read_param_u32(params, i).map(|v| v as usize);
        let k = rd(5)?;
        let sx = rd(7)?;
        let sy = rd(8)?;
        let column_stride = rd(9)?;
        // Single-token, non-small-K kernels compute one row per CUDA block.
        // In MoE, column_stride spans every selected expert, not just one.
        let m = (grid[0] as usize).checked_mul(if small { block[1] as usize } else { 1 })?;
        let cx = rd(11)?;
        let cy = rd(12)?;
        let cd = rd(13)?;
        let ratio = read_param_uint3_z(params, 10)? as usize;
        let ychannels = read_param_uint3_z(params, 6)? as usize;
        stage = "shape";
        if k == 0
            || k % 256 != 0
            || k > 32768
            || sx < k / block_k
            || sy < k / 32
            || m == 0
            || m > 131072
            || column_stride < m
            || (grid[1] > 1 && cd < m)
            || (ids == 0 && ratio == 0)
            || (ids != 0 && ychannels == 0)
        {
            return None;
        }
        let mut gate = 0u64;
        let mut xbias = 0u64;
        let mut gbias = 0u64;
        stage = "fusion";
        if fusion {
            let p = *params.add(3) as *const u8;
            if p.is_null() {
                return None;
            }
            let ptr = |i: usize| std::ptr::read_unaligned(p.add(i * 8) as *const u64);
            if ptr(3) != 0 || ptr(4) != 0 {
                return None;
            }
            gate = ptr(1);
            xbias = ptr(0);
            gbias = if gate != 0 { ptr(2) } else { 0 };
            if gate != 0 && std::ptr::read_unaligned(p.add(40) as *const i32) != 2 {
                return None;
            }
        }
        stage = "ids";
        if ids != 0 && !sifive_host_or_cuda_alloc_has_bytes(ids, grid[1] as usize * 4, false) {
            return None;
        }
        let mut channels = Vec::new();
        stage = "ranges";
        for ch in 0..grid[1] as usize {
            let expert = if ids != 0 {
                usize::try_from(*(ids as *const i32).add(ch)).ok()?
            } else {
                ch / ratio
            };
            let ych = if ids != 0 { ch % ychannels } else { ch };
            let xo = expert.checked_mul(cx)?.checked_mul(block_bytes)?;
            let yo = ych.checked_mul(cy)?.checked_mul(36)?;
            let doff = ch.checked_mul(cd)?.checked_mul(4)?;
            let xp = x.checked_add(xo as u64)?;
            let yp = y.checked_add(yo as u64)?;
            let dp = dst.checked_add(doff as u64)?;
            let xs =
                ((m - 1).checked_mul(sx)?.checked_add(k / block_k)?).checked_mul(block_bytes)?;
            let ys = (k / 32) * 36;
            let gp = if gate != 0 {
                gate.checked_add(xo as u64)?
            } else {
                0
            };
            let boff = (if ids != 0 { expert } else { ch })
                .checked_mul(cd)?
                .checked_mul(4)? as u64;
            let bp = if xbias != 0 {
                xbias.checked_add(boff)?
            } else {
                0
            };
            let gbp = if gbias != 0 {
                gbias.checked_add(boff)?
            } else {
                0
            };
            for b in [bp, gbp] {
                if b != 0 && !sifive_host_or_cuda_alloc_has_bytes(b, m * 4, false) {
                    return None;
                }
            }
            if !sifive_host_or_cuda_alloc_has_bytes(xp, xs, false)
                || !sifive_host_or_cuda_alloc_has_bytes(yp, ys, false)
                || !sifive_host_or_cuda_alloc_has_bytes(dp, m * 4, true)
                || (gate != 0 && !sifive_host_or_cuda_alloc_has_bytes(gp, xs, false))
            {
                return None;
            }
            channels.push((xp, yp, dp, gp, bp, gbp));
        }
        stage = "device";
        let rowbytes = (k / block_k) * block_bytes;
        let tile_rows = (sifive_runtime_sys::kernel_payload_capacity().checked_sub(65536 + (k / 32) * 36)?
            / (rowbytes * if gate != 0 { 2 } else { 1 } + 12))
            .min(512);
        if tile_rows == 0 {
            return None;
        }
        for (xp, yp, dp, gp, bp, gbp) in channels {
            for row0 in (0..m).step_by(tile_rows) {
                let rows = (m - row0).min(tile_rows);
                let mut xb = vec![0u8; rows * rowbytes * if gate != 0 { 2 } else { 1 }];
                for (t, base) in [xp, gp]
                    .into_iter()
                    .take(if gate != 0 { 2 } else { 1 })
                    .enumerate()
                {
                    for r in 0..rows {
                        std::ptr::copy_nonoverlapping(
                            (base as *const u8).add((row0 + r) * sx * block_bytes),
                            xb.as_mut_ptr().add((t * rows + r) * rowbytes),
                            rowbytes,
                        );
                    }
                }
                let mut out = vec![f32::NAN; rows];
                let mut bias = vec![0f32; 2 * rows];
                for (t, b) in [bp, gbp].into_iter().enumerate() {
                    if b != 0 {
                        bias[t * rows..(t + 1) * rows].copy_from_slice(std::slice::from_raw_parts(
                            (b as *const f32).add(row0),
                            rows,
                        ));
                    }
                }
                if sifive_runtime_sys::q4k_tile::submit_qk_tile(
                    current_sifive_device_id_or_zero(),
                    xb.as_ptr(),
                    yp as *const u8,
                    out.as_mut_ptr(),
                    rows as u32,
                    1,
                    (k / 256) as u32,
                    (gate != 0) as u32,
                    qtype,
                    bias.as_ptr(),
                ) != 0
                {
                    return Some(false);
                }
                std::ptr::copy_nonoverlapping(out.as_ptr(), (dp as *mut f32).add(row0), rows);
            }
        }
        eprintln!(
            "PACC Q{}_{} XM experts={} m={} n={} k={} fused_gate={} completion=verified",
            qtype,
            if qtype == 8 { "0" } else { "K" },
            grid[1],
            m,
            n,
            k,
            gate != 0
        );
        Some(true)
    })();
    if run.is_none() {
        eprintln!("PACC quantized layout rejected stage={} name={} grid={:?} block={:?} scalars={:?} ratios={:?}",stage,name,grid,block,
            [5,7,8,9,11,12,13].map(|i|read_param_u32(params,i)),[6,10,14].map(|i|read_param_uint3_z(params,i)));
        if stage == "fusion" {
            let p = *params.add(3) as *const u64;
            if !p.is_null() {
                eprintln!(
                    "PACC rejected fusion={:?} op={}",
                    (0..5)
                        .map(|i| p.add(i).read_unaligned())
                        .collect::<Vec<_>>(),
                    (p.add(5) as *const u32).read_unaligned()
                );
            }
        }
    }
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(cuda_types::cuda::CUerror::UNKNOWN),
        None => Err(cuda_types::cuda::CUerror::NOT_SUPPORTED),
    }
}
