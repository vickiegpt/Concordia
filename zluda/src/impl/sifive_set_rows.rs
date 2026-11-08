use super::*;

pub(super) unsafe fn launch(
    name: &str,
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let run = (|| -> Option<bool> {
        let (src_elem, idx_elem, dst_elem) = sifive_set_rows_element_sizes(name)?;
        if src_elem != 4
            || !matches!(idx_elem, 4 | 8)
            || !matches!(dst_elem, 2 | 4)
            || name.contains("bfloat")
        {
            return None;
        }
        // Validate entire registered ranges and index values before any scatter.
        for i in 0..3 {
            sifive_set_rows_binding_metadata(name, params, i)?;
        }
        let src = read_param_u64(params, 0)?;
        let ids = read_param_u64(params, 1)?;
        let dst = read_param_u64(params, 2)?;
        let rd = |i| usize::try_from(read_param_i64(params, i)?).ok();
        let total = rd(3)?;
        let w = read_param_uint3_z(params, 17)? as usize;
        let n1 = read_param_uint3_z(params, 18)? as usize;
        let n2 = read_param_uint3_z(params, 19)? as usize;
        let n11 = read_param_uint3_z(params, 20)? as usize;
        let n12 = read_param_uint3_z(params, 21)? as usize;
        if [w, n1, n2, n11, n12].contains(&0)
            || total == 0
            || total % w != 0
            || total > 16 * 1024 * 1024
        {
            return None;
        }
        let (s01, s02, s03) = (rd(8)?, rd(9)?, rd(10)?);
        let (s10, s11, s12) = (rd(11)?, rd(12)?, rd(13)?);
        let (s1, s2, s3) = (rd(14)?, rd(15)?, rd(16)?);
        let de = dst_elem as usize;
        let mut tasks = Vec::new();
        for r in 0..total / w {
            let i1 = r % n1;
            let i2 = (r / n1) % n2;
            let i3 = r / n1 / n2;
            let index = i1
                .checked_mul(s10)?
                .checked_add((i2 % n11).checked_mul(s11)?)?
                .checked_add((i3 % n12).checked_mul(s12)?)?;
            let p = ids.checked_add(index.checked_mul(idx_elem as usize)? as u64)?;
            let row = if idx_elem == 8 {
                (p as *const i64).read_unaligned()
            } else {
                (p as *const i32).read_unaligned() as i64
            };
            let row = usize::try_from(row).ok()?;
            let si = i1
                .checked_mul(s01)?
                .checked_add(i2.checked_mul(s02)?)?
                .checked_add(i3.checked_mul(s03)?)?;
            let di = row
                .checked_mul(s1)?
                .checked_add(i2.checked_mul(s2)?)?
                .checked_add(i3.checked_mul(s3)?)?;
            let sp = src.checked_add(si.checked_mul(4)? as u64)?;
            let dp = dst.checked_add(di.checked_mul(de)? as u64)?;
            if !sifive_host_or_cuda_alloc_has_bytes(sp, w.checked_mul(4)?, false)
                || !sifive_host_or_cuda_alloc_has_bytes(dp, w.checked_mul(de)?, true)
            {
                return None;
            }
            tasks.push((sp, dp));
        }
        let capacity = sifive_runtime_sys::q4k_tile::payload_capacity().checked_sub(65536 + 8)?;
        let chunk = capacity / ((4 + de) * w);
        if chunk == 0 {
            return None;
        }
        let mut tiles = 0;
        for tasks in tasks.chunks(chunk) {
            let mut packed = vec![0u8; tasks.len() * w * 4];
            let mut result = vec![0u8; tasks.len() * w * de];
            for (r, (sp, _)) in tasks.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    *sp as *const u8,
                    packed.as_mut_ptr().add(r * w * 4),
                    w * 4,
                );
            }
            if sifive_runtime_sys::concat_tile::cast_f32(
                current_sifive_device_id_or_zero(),
                packed.as_ptr(),
                result.as_mut_ptr(),
                tasks.len() * w,
                de == 2,
            ) != 0
            {
                return Some(false);
            }
            for (r, (_, dp)) in tasks.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    result.as_ptr().add(r * w * de),
                    *dp as *mut u8,
                    w * de,
                );
            }
            tiles += 1;
        }
        eprintln!("PACC set_rows device-cast sparse-staging elements={} width={} tiles={} completion=verified",total,w,tiles);
        Some(true)
    })();
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(cuda_types::cuda::CUerror::UNKNOWN),
        None => Err(cuda_types::cuda::CUerror::NOT_SUPPORTED),
    }
}
