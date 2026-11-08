use crate::*;
use std::sync::Mutex;
static KERNELS: Mutex<[Option<usize>; 4]> = Mutex::new([None; 4]);

#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_submit_topk_row(
    dev: i32,
    input: *const f32,
    output: *mut f32,
    n: u32,
    k: u32,
    flags: u32,
    clamp: f32,
    scale: f32,
) -> i32 {
    if !(0..4).contains(&dev)
        || input.is_null()
        || output.is_null()
        || n == 0
        || n > 576
        || k == 0
        || k > n
        || flags & !29 != 0
        || !clamp.is_finite()
        || clamp < 0.
        || !scale.is_finite()
        || env_flag_enabled("HETGPU_SIFIVE_SKIP_JOB_WAIT")
        || env_flag_enabled("HETGPU_SIFIVE_ASSUME_SUCCESS_ON_WAIT_ERROR")
    {
        return -1;
    }
    let mut cache = match KERNELS.lock() {
        Ok(v) => v,
        Err(_) => return -1,
    };
    let kernel = if let Some(v) = cache[dev as usize] {
        v as *mut sifive_Kernel
    } else {
        let d = sifive_CreateDevice(dev as u32);
        let p = sifive_CreateProgram();
        if d.is_null() || p.is_null() {
            return -1;
        }
        let source = include_bytes!("topk_moe.c");
        if sifive_LoadProgramSource(
            p,
            std::ptr::null(),
            b"topk_moe.c\0".as_ptr().cast(),
            source.as_ptr(),
            source.len() as u64,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        ) != 0
        {
            return -1;
        }
        let q = sifive_CreateKernelOnDevice(p, d, b"pacc_topk_row_v1\0".as_ptr().cast());
        if q.is_null() {
            return -1;
        }
        cache[dev as usize] = Some(q as usize);
        q
    };
    let mut result = vec![f32::NAN; 2 * k as usize];
    if sifive_KernelClearLaunchState(kernel) != 0 {
        return -1;
    }
    for (i, value) in [
        input as u64,
        result.as_mut_ptr() as u64,
        n as u64,
        k as u64,
        flags as u64,
        clamp.to_bits() as u64,
        scale.to_bits() as u64,
    ]
    .into_iter()
    .enumerate()
    {
        let rec = SifiveKernelArgRecord {
            kind: if i < 2 {
                SIFIVE_KERNEL_ARG_KIND_POINTER
            } else {
                SIFIVE_KERNEL_ARG_KIND_SCALAR
            },
            size: 8,
            flags: 0,
            reserved: 0,
            value,
            value_hi: 0,
        };
        if sifive_KernelPushArgRecord(kernel, &rec) != 0 {
            return -1;
        }
        if i < 2 {
            let binding = SifiveKernelBufferBinding {
                arg_index: i as u32,
                flags: if i == 0 {
                    SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT
                } else {
                    SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT | SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT
                },
                addr: value,
                size: 4 * if i == 0 {
                    n as u64 * if flags & 16 != 0 { 2 } else { 1 }
                } else {
                    2 * k as u64
                },
            };
            if sifive_KernelAddBufferBinding(kernel, &binding) != 0 {
                return -1;
            }
        }
    }
    if sifive_LaunchKernel(kernel, 1, 1, 1, 1, 1, 1) != 0 || result.iter().any(|v| !v.is_finite()) {
        return -1;
    }
    let ids = &result[k as usize..];
    for (i, &v) in ids.iter().enumerate() {
        if v < 0. || v >= n as f32 || v.fract() != 0. || ids[..i].contains(&v) {
            return -1;
        }
    }
    std::ptr::copy_nonoverlapping(result.as_ptr(), output, result.len());
    0
}
