use crate::*;
use std::sync::Mutex;

static KERNELS: Mutex<[Option<usize>; 4]> = Mutex::new([None; 4]);

// Packed input: q, k, v, g, beta, initial_state. Output: attention, state slots.
// All spans are in elements. This intentionally bounds each head below the
// existing 2 MiB shared-DDR transport; it does not enlarge advertised memory.
pub fn lengths(s: usize, n: usize, kda: bool, slots: usize) -> Option<(usize, usize)> {
    if !matches!(s, 16 | 32 | 64 | 128) || n == 0 || n > 256 || slots == 0 || slots > 16 {
        return None;
    }
    Some((
        3 * n * s + n * if kda { s } else { 1 } + n + s * s,
        n * s + slots * s * s,
    ))
}

unsafe fn kernel(dev: usize) -> Option<*mut sifive_Kernel> {
    let device = sifive_CreateDevice(dev as u32);
    if device.is_null() {
        return None;
    }
    let program = sifive_CreateProgram();
    if program.is_null() {
        return None;
    }
    let source = include_bytes!("gated_delta.c");
    let rc = sifive_LoadProgramSource(
        program,
        std::ptr::null(),
        b"pacc_gdn_head_v1.c\0".as_ptr().cast(),
        source.as_ptr(),
        source.len() as u64,
        std::ptr::null(),
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
    );
    if rc != sifive_Result_Success {
        return None;
    }
    let k = sifive_CreateKernelOnDevice(program, device, b"pacc_gdn_head_v1\0".as_ptr().cast());
    if k.is_null() {
        None
    } else {
        Some(k)
    }
}

/// Caller owns the packed input/output spans returned by lengths().
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_submit_gdn_head_f32(
    dev: i32,
    input: *const f32,
    output: *mut f32,
    s: u32,
    n: u32,
    kda: u32,
    keep: u32,
    slots: u32,
    scale: f32,
) -> i32 {
    if !(0..4).contains(&dev)
        || input.is_null()
        || output.is_null()
        || kda > 1
        || keep > 1
        || !scale.is_finite()
        || (keep == 0 && slots != 1)
        || (keep != 0 && slots > n)
        || env_flag_enabled("HETGPU_SIFIVE_SKIP_JOB_WAIT")
        || env_flag_enabled("HETGPU_SIFIVE_ASSUME_SUCCESS_ON_WAIT_ERROR")
    {
        return -1;
    }
    let Some((ins, outs)) = lengths(s as usize, n as usize, kda != 0, slots as usize) else {
        return -1;
    };
    let mut guard = match KERNELS.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    let k = match guard[dev as usize] {
        Some(k) => k as *mut sifive_Kernel,
        None => match kernel(dev as usize) {
            Some(k) => {
                guard[dev as usize] = Some(k as usize);
                k
            }
            None => return -1,
        },
    };
    // Separate staging also permits input/output aliasing and fail-closed output.
    let mut result = vec![f32::NAN; outs];
    if sifive_KernelClearLaunchState(k) != 0 {
        return -1;
    }
    let values = [
        input as u64,
        result.as_mut_ptr() as u64,
        s as u64,
        n as u64,
        kda as u64,
        keep as u64,
        slots as u64,
        scale.to_bits() as u64,
    ];
    for (i, value) in values.into_iter().enumerate() {
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
        if sifive_KernelPushArgRecord(k, &rec) != 0 {
            return -1;
        }
        if i < 2 {
            let b = SifiveKernelBufferBinding {
                arg_index: i as u32,
                flags: if i == 0 {
                    SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT
                } else {
                    SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT | SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT
                },
                addr: value,
                size: (if i == 0 { ins } else { outs } * 4) as u64,
            };
            if sifive_KernelAddBufferBinding(k, &b) != 0 {
                return -1;
            }
        }
    }
    if sifive_LaunchKernel(k, 1, 1, 1, s, 1, 1) != 0 || result.iter().any(|v| !v.is_finite()) {
        return -1;
    }
    std::ptr::copy_nonoverlapping(result.as_ptr(), output, outs);
    0
}
