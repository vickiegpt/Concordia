use crate::*;
use std::sync::Mutex;
static KERNELS: Mutex<[Option<usize>; 4]> = Mutex::new([None; 4]);
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_submit_attention_softmax(
    dev: i32,
    input: *const f32,
    out: *mut f32,
    rows: u32,
    n: u32,
) -> i32 {
    if !(0..4).contains(&dev)
        || input.is_null()
        || out.is_null()
        || rows == 0
        || rows > 1024
        || n == 0
        || n > 32768
        || rows as u64 * (3 * n as u64 + 4) * 4 > 2 * 1024 * 1024 - 65536
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
        let source = include_bytes!("attention_softmax.c");
        if sifive_LoadProgramSource(
            p,
            std::ptr::null(),
            b"attention_softmax.c\0".as_ptr().cast(),
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
        let k = sifive_CreateKernelOnDevice(p, d, b"pacc_attention_softmax_v1\0".as_ptr().cast());
        if k.is_null() {
            return -1;
        }
        cache[dev as usize] = Some(k as usize);
        k
    };
    let count = (rows * n) as usize;
    let mut result = vec![f32::NAN; count + 1];
    if sifive_KernelClearLaunchState(kernel) != 0 {
        return -1;
    }
    for (i, value) in [
        input as u64,
        result.as_mut_ptr() as u64,
        rows as u64,
        n as u64,
    ]
    .into_iter()
    .enumerate()
    {
        let arg = SifiveKernelArgRecord {
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
        if sifive_KernelPushArgRecord(kernel, &arg) != 0 {
            return -1;
        }
        if i < 2 {
            let binding = SifiveKernelBufferBinding {
                arg_index: i as u32,
                addr: value,
                size: if i == 0 {
                    4 * rows as u64 * (2 * n as u64 + 4)
                } else {
                    4 * (count as u64 + 1)
                },
                flags: if i == 0 {
                    SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT
                } else {
                    SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT | SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT
                },
            };
            if sifive_KernelAddBufferBinding(kernel, &binding) != 0 {
                return -1;
            }
        }
    }
    if sifive_LaunchKernel(kernel, 1, 1, 1, 1, 1, 1) != 0
        || result[count] != 0.
        || result[..count]
            .iter()
            .any(|v| !v.is_finite() || *v < 0. || *v > 1.)
    {
        return -1;
    }
    for (i, row) in result[..count].chunks(n as usize).enumerate() {
        let sum: f32 = row.iter().sum();
        if sum > 1.0001
            || (input.add(i * (2 * n as usize + 4) + 3).read() == 0. && (sum - 1.).abs() > 1e-4)
        {
            return -1;
        }
    }
    std::ptr::copy_nonoverlapping(result.as_ptr(), out, count);
    0
}
