use super::*;

pub(super) unsafe fn launch_matrix(
    name: &str,
    grid: [u32; 3],
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let run = (|| -> Option<bool> {
        let tail = name.strip_prefix("_ZL9mul_mat_fI")?;
        let (dtype, tail) = tail.split_once("Li32ELi")?;
        let (tile, tail) = tail.split_once("ELi")?;
        let (warps, tail) = tail.split_once("ELb0EE")?;
        if !matches!(tile, "8" | "16")
            || !(1..=8).contains(&warps.parse::<usize>().ok()?)
            || tail != "vPKT_PKfPKiPfiiiiiiiiiiiiiiii"
            || grid.iter().any(|&v| v == 0)
            || grid[1] > 256
            || grid[2] > 64
        {
            return None;
        }
        let (atype, elem, vals) = match dtype {
            "f" => (4, 4usize, 1usize),
            "7__half2" => (3, 2, 2),
            _ => return None,
        };
        let x = read_param_u64(params, 0)?;
        let y = read_param_u64(params, 1)?;
        let dst = read_param_u64(params, 3)?;
        if read_param_u64(params, 2)? != 0 {
            return None;
        }
        let rd = |i| usize::try_from(read_param_i32(params, i)?).ok();
        let k = rd(4)?.checked_mul(vals)?;
        let n = rd(5)?;
        let m = (grid[0] as usize).checked_mul(32)?;
        let sx = rd(7)?.checked_mul(vals)?;
        let sy = rd(8)?.checked_mul(vals)?;
        let sd = rd(9)?;
        let cr = rd(12)?;
        let cx = rd(13)?.checked_mul(vals)?;
        let cy = rd(14)?;
        let cd = rd(15)?;
        let sr = rd(16)?;
        let px = rd(17)?.checked_mul(vals)?;
        let py = rd(18)?;
        let pd = rd(19)?;
        if rd(6)? != grid[1] as usize
            || cr == 0
            || sr == 0
            || k == 0
            || k > 32768
            || n == 0
            || n > 16
            || m > 131072
            || sx < k
            || sy < k
            || sd < m
        {
            return None;
        }
        let span = |rows: usize, stride: usize, tail: usize, bytes: usize| {
            (rows - 1)
                .checked_mul(stride)?
                .checked_add(tail)?
                .checked_mul(bytes)
        };
        let xs = span(m, sx, k, elem)?;
        let ys = span(n, sy, k, 4)?;
        let ds = span(n, sd, m, 4)?;
        let addr = |base: u64, sample: usize, ps: usize, head: usize, cs: usize, bytes: usize| {
            base.checked_add(
                sample
                    .checked_mul(ps)?
                    .checked_add(head.checked_mul(cs)?)?
                    .checked_mul(bytes)? as u64,
            )
        };
        let mut tasks = Vec::new();
        for sample in 0..grid[2] as usize {
            for head in 0..grid[1] as usize {
                let a = addr(x, sample / sr, px, head / cr, cx, elem)?;
                let b = addr(y, sample, py, head, cy, 4)?;
                let c = addr(dst, sample, pd, head, cd, 4)?;
                if !sifive_host_or_cuda_alloc_has_bytes(a, xs, false)
                    || !sifive_host_or_cuda_alloc_has_bytes(b, ys, false)
                    || !sifive_host_or_cuda_alloc_has_bytes(c, ds, true)
                {
                    return None;
                }
                tasks.push((a, b, c));
            }
        }
        let alpha = 1f32;
        let beta = 0f32;
        for (a, b, c) in tasks {
            for col in (0..n).step_by(8) {
                if sifive_runtime_sys::hetgpu_sifive_submit_gemm_mmvf_small_n(
                    1,
                    0,
                    m as i32,
                    (n - col).min(8) as i32,
                    k as i32,
                    (&alpha as *const f32).cast(),
                    a as *const _,
                    atype,
                    sx as i32,
                    0,
                    (b as *const f32).add(col * sy).cast(),
                    4,
                    sy as i32,
                    0,
                    (&beta as *const f32).cast(),
                    (c as *mut f32).add(col * sd).cast(),
                    4,
                    sd as i32,
                    0,
                    1,
                    0,
                ) != 0
                {
                    return Some(false);
                }
            }
        }
        eprintln!(
            "PACC MMF device-MMVF heads={} samples={} m={} n={} k={} atype={} completion=verified",
            grid[1], grid[2], m, n, k, atype
        );
        Some(true)
    })();
    if run.is_none() {
        eprintln!(
            "PACC MMF unsupported template/layout name={} grid={:?}",
            name, grid
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
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let run = (|| -> Option<bool> {
        let (n, fusion, multi) = sifive_parse_mmvf_template(name)?;
        if fusion
            || multi
            || n == 0
            || n > 8
            || grid.iter().any(|&v| v == 0)
            || grid[1] > 256
            || grid[2] > 64
        {
            return None;
        }
        let xtype = sifive_mmvf_x_type(name)?;
        let (atype, elem) = match xtype {
            1 => (4, 4usize),
            2 => (3, 2),
            3 => (5, 2),
            _ => return None,
        };
        let x = read_param_u64(params, 0)?;
        let y = read_param_u64(params, 1)?;
        let dst = read_param_u64(params, 4)?;
        if read_param_u64(params, 2)? != 0 {
            return None;
        }
        let rd = |i| usize::try_from(read_param_i32(params, i)?).ok();
        let k = rd(5)?.checked_mul(2)?;
        let m = grid[0] as usize;
        let n = n as usize;
        let sx = rd(7)?;
        let sy = rd(8)?.checked_mul(2)?;
        let sd = rd(9)?;
        let cx = rd(11)?;
        let cy = rd(12)?;
        let cd = rd(13)?;
        let px = rd(15)?;
        let py = rd(16)?;
        let pd = rd(17)?;
        let cr = read_param_uint3_z(params, 10)? as usize;
        let sr = read_param_uint3_z(params, 14)? as usize;
        if cr == 0 || sr == 0 || k == 0 || k > 32768 || m > 131072 || sx < k || sy < k || sd < m {
            return None;
        }
        let span = |rows: usize, stride: usize, tail: usize, bytes: usize| {
            (rows - 1)
                .checked_mul(stride)?
                .checked_add(tail)?
                .checked_mul(bytes)
        };
        let xs = span(m, sx, k, elem)?;
        let ys = span(n, sy, k, 4)?;
        let ds = span(n, sd, m, 4)?;
        let addr = |base: u64, sample: usize, ps: usize, head: usize, cs: usize, bytes: usize| {
            base.checked_add(
                sample
                    .checked_mul(ps)?
                    .checked_add(head.checked_mul(cs)?)?
                    .checked_mul(bytes)? as u64,
            )
        };
        let mut tasks = Vec::new();
        for sample in 0..grid[2] as usize {
            for head in 0..grid[1] as usize {
                let a = addr(x, sample / sr, px, head / cr, cx, elem)?;
                let b = addr(y, sample, py, head, cy, 4)?;
                let c = addr(dst, sample, pd, head, cd, 4)?;
                if !sifive_host_or_cuda_alloc_has_bytes(a, xs, false)
                    || !sifive_host_or_cuda_alloc_has_bytes(b, ys, false)
                    || !sifive_host_or_cuda_alloc_has_bytes(c, ds, true)
                {
                    return None;
                }
                tasks.push((a, b, c));
            }
        }
        let alpha = 1f32;
        let beta = 0f32;
        for (a, b, c) in tasks {
            if sifive_runtime_sys::hetgpu_sifive_submit_gemm_mmvf_small_n(
                1,
                0,
                m as i32,
                n as i32,
                k as i32,
                (&alpha as *const f32).cast(),
                a as *const _,
                atype,
                sx as i32,
                0,
                b as *const _,
                4,
                sy as i32,
                0,
                (&beta as *const f32).cast(),
                c as *mut _,
                4,
                sd as i32,
                0,
                1,
                0,
            ) != 0
            {
                return Some(false);
            }
        }
        eprintln!(
            "PACC MMVF staged heads={} samples={} m={} n={} k={} atype={} completion=verified",
            grid[1], grid[2], m, n, k, atype
        );
        Some(true)
    })();
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(cuda_types::cuda::CUerror::UNKNOWN),
        None => Err(cuda_types::cuda::CUerror::NOT_SUPPORTED),
    }
}
