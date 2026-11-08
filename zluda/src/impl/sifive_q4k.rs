use super::*;

pub(super) unsafe fn launch_requests(
    name: &str,
    grid: [u32; 3],
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let mut stage = "template";
    let run = (|| -> Option<bool> {
        if !sifive_env_truthy("HETGPU_SIFIVE_QK_REQUEST_BATCH") {
            return None;
        }
        let moe = name.starts_with("_ZL17mul_mat_vec_q_moeI");
        let prefix = if moe {
            "_ZL17mul_mat_vec_q_moeIL9ggml_type"
        } else {
            "_ZL13mul_mat_vec_qIL9ggml_type"
        };
        let (dtype, tail) = name.strip_prefix(prefix)?.split_once("ELi")?;
        let qtype = match dtype {
            "12" => 4,
            "13" => 5,
            "8" => 8,
            _ => return None,
        };
        let block_bytes = match qtype {
            4 => 144,
            5 => 176,
            _ => 34,
        };
        let block_k = if qtype == 8 { 32 } else { 256 };
        let rd = |i| read_param_u32(params, i).map(|v| v as usize);
        let x = read_param_u64(params, 0)?;
        let y = read_param_u64(params, 1)?;
        let ids = read_param_u64(params, 2)?;
        let (dst, k, m, n, sx, sy, sd, cx, cy, cd, yheads, ratio, ids_stride) = if moe {
            if !tail.starts_with("2EE") || ids == 0 {
                return None;
            }
            (
                read_param_u64(params, 3)?,
                rd(4)?,
                rd(6)?,
                rd(13)?,
                rd(7)?,
                rd(8)?,
                rd(9)?,
                rd(10)?,
                rd(11)?,
                rd(12)?,
                read_param_uint3_z(params, 5)? as usize,
                0,
                rd(14)?,
            )
        } else {
            let (cols, flags) = tail.split_once("ELb")?;
            if flags
                != "0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"
                || ids != 0
            {
                return None;
            }
            (
                read_param_u64(params, 4)?,
                rd(5)?,
                rd(9)?,
                cols.parse::<usize>().ok()?,
                rd(7)?,
                rd(8)?,
                rd(9)?,
                rd(11)?,
                rd(12)?,
                rd(13)?,
                0,
                read_param_uint3_z(params, 10)? as usize,
                0,
            )
        };
        stage = "shape";
        if grid[0] as usize != m.div_ceil(if !moe && n == 1 { 1 } else { 2 })
            || grid[1] == 0
            || grid[1] > 256
            || grid[2] != 1
            || !(1..=8).contains(&n)
            || (moe && n == 1)
            || k == 0
            || k % 256 != 0
            || k > 32768
            || m == 0
            || m > 131072
            || sx < k / block_k
            || sy < k / 32
            || sd < m
            || (grid[1] > 1 && cd < m)
            || (moe && (yheads == 0 || ids_stride < grid[1] as usize))
            || (!moe && ratio == 0)
        {
            return None;
        }
        if moe
            && !sifive_host_or_cuda_alloc_has_bytes(
                ids,
                ((n - 1)
                    .checked_mul(ids_stride)?
                    .checked_add(grid[1] as usize)?)
                .checked_mul(4)?,
                false,
            )
        {
            return None;
        }
        let rowbytes = k / block_k * block_bytes;
        let ybytes = k / 32 * 36;
        let xspan =
            ((m - 1).checked_mul(sx)?.checked_add(k / block_k)?).checked_mul(block_bytes)?;
        // One group shares weights, while every column retains its own input/output.
        let mut groups: Vec<(u64, Vec<(u64, u64)>)> = Vec::new();
        stage = "ranges";
        for col in 0..n {
            for ch in 0..grid[1] as usize {
                let expert = if moe {
                    usize::try_from(*((ids as *const i32).add(col * ids_stride + ch))).ok()?
                } else {
                    ch / ratio
                };
                let ych = if moe { ch % yheads } else { ch };
                let xp = x.checked_add(expert.checked_mul(cx)?.checked_mul(block_bytes)? as u64)?;
                let yp = y.checked_add(
                    (ych.checked_mul(cy)?.checked_add(col.checked_mul(sy)?)?).checked_mul(36)?
                        as u64,
                )?;
                let dp = dst.checked_add(
                    (ch.checked_mul(cd)?.checked_add(col.checked_mul(sd)?)?).checked_mul(4)? as u64,
                )?;
                if !sifive_host_or_cuda_alloc_has_bytes(xp, xspan, false)
                    || !sifive_host_or_cuda_alloc_has_bytes(yp, ybytes, false)
                    || !sifive_host_or_cuda_alloc_has_bytes(dp, m * 4, true)
                {
                    return None;
                }
                if let Some(group) = groups.iter_mut().find(|g| g.0 == xp) {
                    group.1.push((yp, dp));
                } else {
                    groups.push((xp, vec![(yp, dp)]));
                }
            }
        }
        let group_count = groups.len();
        let mut tiles = 0;
        let mut multi_tiles = 0;
        let mut max_n = 0;
        stage = "submit";
        for (xp, columns) in groups {
            for columns in columns.chunks(8) {
                let nc = columns.len();
                max_n = max_n.max(nc);
                let mut yb = vec![0u8; nc * ybytes];
                for (col, (yp, _)) in columns.iter().enumerate() {
                    std::ptr::copy_nonoverlapping(
                        *yp as *const u8,
                        yb.as_mut_ptr().add(col * ybytes),
                        ybytes,
                    );
                }
                let rows = (sifive_runtime_sys::q4k_tile::payload_capacity()
                    .checked_sub(65536 + 64 + yb.len())?
                    / (rowbytes + nc * 4 + 8))
                    .min(512);
                if rows == 0 {
                    return None;
                }
                let rows = if rows >= 32 { rows / 32 * 32 } else { rows };
                for row0 in (0..m).step_by(rows) {
                    let nr = (m - row0).min(rows);
                    let mut xb = vec![0u8; nr * rowbytes];
                    let mut out = vec![f32::NAN; nr * nc];
                    for r in 0..nr {
                        std::ptr::copy_nonoverlapping(
                            (xp as *const u8).add((row0 + r) * sx * block_bytes),
                            xb.as_mut_ptr().add(r * rowbytes),
                            rowbytes,
                        );
                    }
                    if sifive_runtime_sys::q4k_tile::submit_qk_tile(
                        current_sifive_device_id_or_zero(),
                        xb.as_ptr(),
                        yb.as_ptr(),
                        out.as_mut_ptr(),
                        nr as u32,
                        nc as u32,
                        (k / 256) as u32,
                        0,
                        qtype,
                        std::ptr::null(),
                    ) != 0
                    {
                        return Some(false);
                    }
                    for (col, (_, dp)) in columns.iter().enumerate() {
                        for r in 0..nr {
                            *((*dp as *mut f32).add(row0 + r)) = out[r * nc + col];
                        }
                    }
                    tiles += 1;
                    multi_tiles += usize::from(nc > 1);
                }
            }
        }
        eprintln!("PACC_XM_REQUEST_BATCH moe={} requests={} channels={} groups={} m={} k={} tiles={} multi_tiles={} max_n={} completion=verified",
            moe,n,grid[1],group_count,m,k,tiles,multi_tiles,max_n);
        Some(true)
    })();
    if run.is_none() {
        eprintln!(
            "PACC request batch rejected stage={} name={} grid={:?}",
            stage, name, grid
        );
    }
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(cuda_types::cuda::CUerror::UNKNOWN),
        None => Err(cuda_types::cuda::CUerror::NOT_SUPPORTED),
    }
}

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
        let tile_rows = (sifive_runtime_sys::q4k_tile::payload_capacity()
            .checked_sub(65536 + 64 + (k / 32) * 36)?
            / (rowbytes * if gate != 0 { 2 } else { 1 } + 12))
            .min(512);
        if tile_rows == 0 {
            return None;
        }
        let mut groups: Vec<Vec<(u64, u64, u64, u64, u64, u64)>> = Vec::new();
        let batch = std::env::var("HETGPU_SIFIVE_QK_EXPERT_BATCH")
            .ok()
            .as_deref()
            == Some("1");
        // Avoid creating a partial XM M=32 tile at every host partition boundary.
        let tile_rows = if batch && tile_rows >= 32 {
            tile_rows / 32 * 32
        } else {
            tile_rows
        };
        for channel in channels {
            if let Some(group) = groups
                .iter_mut()
                .find(|group| batch && group[0].1 == channel.1)
            {
                group.push(channel);
            } else {
                groups.push(vec![channel]);
            }
        }
        let group_count = groups.len();
        let mut tiles = 0;
        let mut pack_us = 0;
        let mut submit_us = 0;
        let mut scatter_us = 0;
        for group in groups {
            let total_rows = m.checked_mul(group.len())?;
            let yp = group[0].1;
            for row0 in (0..total_rows).step_by(tile_rows) {
                let before = std::time::Instant::now();
                let rows = (total_rows - row0).min(tile_rows);
                let mut xb = vec![0u8; rows * rowbytes * if gate != 0 { 2 } else { 1 }];
                for t in 0..if gate != 0 { 2 } else { 1 } {
                    for r in 0..rows {
                        let flat = row0 + r;
                        let channel = group[flat / m];
                        let base = if t == 0 { channel.0 } else { channel.3 };
                        std::ptr::copy_nonoverlapping(
                            (base as *const u8).add((flat % m) * sx * block_bytes),
                            xb.as_mut_ptr().add((t * rows + r) * rowbytes),
                            rowbytes,
                        );
                    }
                }
                let mut out = vec![f32::NAN; rows];
                let mut bias = vec![0f32; 2 * rows];
                for r in 0..rows {
                    let flat = row0 + r;
                    for (t, b) in [group[flat / m].4, group[flat / m].5]
                        .into_iter()
                        .enumerate()
                    {
                        if b != 0 {
                            bias[t * rows + r] = *((b as *const f32).add(flat % m));
                        }
                    }
                }
                pack_us += before.elapsed().as_micros();
                let before = std::time::Instant::now();
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
                submit_us += before.elapsed().as_micros();
                let before = std::time::Instant::now();
                for r in 0..rows {
                    let flat = row0 + r;
                    *((group[flat / m].2 as *mut f32).add(flat % m)) = out[r];
                }
                scatter_us += before.elapsed().as_micros();
                tiles += 1;
            }
        }
        if std::env::var("HETGPU_SIFIVE_ELF_PROFILE").ok().as_deref() == Some("1") {
            eprintln!(
                "PACC_QK_BATCH experts={} groups={} tiles={} pack_us={} submit_us={} scatter_us={}",
                grid[1], group_count, tiles, pack_us, submit_us, scatter_us
            );
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
