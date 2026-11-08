use super::*;

pub(super) unsafe fn launch_argsort(
    name: &str,
    grid: [u32; 3],
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    let run = (|| -> Option<bool> {
        let order = name
            .strip_prefix("_ZL17k_argsort_f32_i32IL15ggml_sort_order")?
            .strip_suffix("EEvPKfPiii")?;
        let flags = match order {
            "0" => 192,
            "1" => 64,
            _ => return None,
        };
        let n = usize::try_from(read_param_i32(params, 2)?).ok()?;
        let pad = usize::try_from(read_param_i32(params, 3)?).ok()?;
        if n == 0
            || n > 576
            || pad != n.next_power_of_two()
            || grid[0] == 0
            || grid[0] > 4096
            || grid[1..] != [1, 1]
        {
            return None;
        }
        let x = read_param_u64(params, 0)?;
        let dst = read_param_u64(params, 1)?;
        let bytes = n.checked_mul(grid[0] as usize)?.checked_mul(4)?;
        if !sifive_host_or_cuda_alloc_has_bytes(x, bytes, false)
            || !sifive_host_or_cuda_alloc_has_bytes(dst, bytes, true)
        {
            return None;
        }
        let mut output = vec![f32::NAN; 2 * n];
        for row in 0..grid[0] as usize {
            if sifive_runtime_sys::topk_moe::hetgpu_sifive_submit_topk_row(
                current_sifive_device_id_or_zero(),
                (x as *const f32).add(row * n),
                output.as_mut_ptr(),
                n as u32,
                n as u32,
                flags,
                0.,
                1.,
            ) != 0
            {
                return Some(false);
            }
            for col in 0..n {
                *((dst as *mut i32).add(row * n + col)) = output[n + col] as i32;
            }
        }
        eprintln!(
            "PACC argsort device-bitonic rows={} cols={} order={} completion=verified",
            grid[0], n, order
        );
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
        let suffix = name.strip_prefix("_Z13topk_moe_cudaILi")?;
        let (experts, rest) = suffix.split_once("ELb")?;
        let n = experts.parse::<usize>().ok()?;
        let bias = rest.starts_with('1');
        let rows = usize::try_from(read_param_i32(params, 4)?).ok()?;
        let k = usize::try_from(read_param_i32(params, 5)?).ok()?;
        let clamp = read_param_f32(params, 6)?;
        let scale = read_param_f32(params, 7)?;
        let raw = read_param_u32(params, 8)?;
        if (0..4).any(|i| ((raw >> (8 * i)) & 255) > 1) {
            return None;
        }
        let mut flags = 0u32;
        for i in 0..4 {
            flags |= ((raw >> (8 * i)) & 1) << i;
        }
        if bias {
            flags |= 16;
        }
        if n == 0 || n > 576 || k == 0 || k > n || rows == 0 || rows > 4096 || flags & 2 != 0 {
            return None;
        }
        let x = read_param_u64(params, 0)?;
        let w = read_param_u64(params, 1)?;
        let ids = read_param_u64(params, 2)?;
        let b = read_param_u64(params, 3)?;
        if !sifive_host_or_cuda_alloc_has_bytes(x, rows * n * 4, false)
            || !sifive_host_or_cuda_alloc_has_bytes(w, rows * k * 4, true)
            || !sifive_host_or_cuda_alloc_has_bytes(ids, rows * n * 4, true)
            || (bias && !sifive_host_or_cuda_alloc_has_bytes(b, n * 4, false))
        {
            return None;
        }
        let mut input = vec![0f32; n * if bias { 2 } else { 1 }];
        let mut output = vec![f32::NAN; 2 * k];
        if bias {
            input[n..].copy_from_slice(std::slice::from_raw_parts(b as *const f32, n));
        }
        for row in 0..rows {
            input[..n].copy_from_slice(std::slice::from_raw_parts(
                (x as *const f32).add(row * n),
                n,
            ));
            if sifive_runtime_sys::topk_moe::hetgpu_sifive_submit_topk_row(
                current_sifive_device_id_or_zero(),
                input.as_ptr(),
                output.as_mut_ptr(),
                n as u32,
                k as u32,
                flags,
                clamp,
                scale,
            ) != 0
            {
                return Some(false);
            }
            std::ptr::copy_nonoverlapping(output.as_ptr(), (w as *mut f32).add(row * k), k);
            for j in 0..k {
                *(ids as *mut i32).add(row * n + j) = output[k + j] as i32;
            }
        }
        eprintln!(
            "PACC topk device-core experts={} rows={} selected={} flags={} completion=verified",
            n, rows, k, flags
        );
        Some(true)
    })();
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(cuda_types::cuda::CUerror::UNKNOWN),
        None => Err(cuda_types::cuda::CUerror::NOT_SUPPORTED),
    }
}
