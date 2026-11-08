use crate::*;
use std::sync::Mutex;
static KERNELS: Mutex<[Option<usize>; 4]> = Mutex::new([None; 4]);

pub fn payload_capacity() -> usize {
    let capacity = kernel_payload_capacity();
    if env_flag_enabled("HETGPU_SIFIVE_QK_LOCAL_BATCH") { capacity.min(512 * 1024) } else { capacity }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_submit_q4k_tile(
    dev: i32,
    x: *const u8,
    y: *const u8,
    out: *mut f32,
    m: u32,
    n: u32,
    nb: u32,
    gate: u32,
) -> i32 {
    submit_qk_tile(dev, x, y, out, m, n, nb, gate, 4, std::ptr::null())
}
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_submit_q5k_tile(
    dev: i32,
    x: *const u8,
    y: *const u8,
    out: *mut f32,
    m: u32,
    n: u32,
    nb: u32,
    gate: u32,
) -> i32 {
    submit_qk_tile(dev, x, y, out, m, n, nb, gate, 5, std::ptr::null())
}
#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_submit_q8_tile(
    dev: i32,
    x: *const u8,
    y: *const u8,
    out: *mut f32,
    m: u32,
    n: u32,
    nb: u32,
    gate: u32,
) -> i32 {
    submit_qk_tile(dev, x, y, out, m, n, nb, gate, 8, std::ptr::null())
}
pub unsafe fn submit_qk_tile(
    dev: i32,
    x: *const u8,
    y: *const u8,
    out: *mut f32,
    m: u32,
    n: u32,
    nb: u32,
    gate: u32,
    qtype: u32,
    bias: *const f32,
) -> i32 {
    if !(0..4).contains(&dev)
        || x.is_null()
        || y.is_null()
        || out.is_null()
        || m == 0
        || m > 1024
        || n == 0
        || n > 8
        || nb == 0
        || nb > 128
        || gate > 1
        || ![4, 5, 8].contains(&qtype)
        || env_flag_enabled("HETGPU_SIFIVE_SKIP_JOB_WAIT")
        || env_flag_enabled("HETGPU_SIFIVE_ASSUME_SUCCESS_ON_WAIT_ERROR")
    {
        return -1;
    }
    let block_bytes = if qtype == 8 {
        272
    } else if qtype == 5 {
        176
    } else {
        144
    };
    let xbytes = (m * nb * block_bytes * (1 + gate)) as u64;
    let ybytes = (n * nb * 288) as u64;
    if xbytes + ybytes + (m * n + 16 + 2 * m) as u64 * 4 > payload_capacity().saturating_sub(65536) as u64 {
        return -1;
    }
    let zeros = vec![0f32; 2 * m as usize];
    let bias = if bias.is_null() { zeros.as_ptr() } else { bias };
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
        let source = format!(
            "{}\n{}",
            include_str!("xsfmm_native_bf16.c"),
            include_str!("q4k_tile.c")
        );
        if sifive_LoadProgramSource(
            p,
            std::ptr::null(),
            b"q4k_tile.c\0".as_ptr().cast(),
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
        let q = sifive_CreateKernelOnDevice(p, d, b"pacc_q4k_tile_v1\0".as_ptr().cast());
        if q.is_null() {
            return -1;
        }
        cache[dev as usize] = Some(q as usize);
        q
    };
    let count = (m * n) as usize;
    let mut result = vec![f32::NAN; count + 16];
    if sifive_KernelClearLaunchState(kernel) != 0 {
        return -1;
    }
    for (i, value) in [
        x as u64,
        y as u64,
        result.as_mut_ptr() as u64,
        m as u64,
        n as u64,
        nb as u64,
        gate as u64,
        qtype as u64,
        bias as u64,
        env_flag_enabled("HETGPU_SIFIVE_QK_PROFILE") as u64 |
            ((env_flag_enabled("HETGPU_SIFIVE_QK_LOCAL_BATCH") as u64) << 1) |
            ((env_flag_enabled("HETGPU_SIFIVE_QK_VECTOR_PACK") as u64) << 2),
    ]
    .into_iter()
    .enumerate()
    {
        let rec = SifiveKernelArgRecord {
            kind: if i < 3 || i == 8 {
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
        if i < 3 || i == 8 {
            let binding = SifiveKernelBufferBinding {
                arg_index: i as u32,
                flags: if i != 2 {
                    SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT
                } else {
                    SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT | SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT
                },
                addr: value,
                size: match i {
                    0 => xbytes,
                    1 => ybytes,
                    8 => m as u64 * 8,
                    _ => ((count + 16) * 4) as u64,
                },
            };
            if sifive_KernelAddBufferBinding(kernel, &binding) != 0 {
                return -1;
            }
        }
    }
    if sifive_LaunchKernel(kernel, 1, 1, 1, 2, 1, 1) != 0 {
        return -1;
    }
    if result[count] != 0. || result[..count].iter().any(|v| !v.is_finite()) {
        eprintln!(
            "PACC Q4_K XM rejected output: diagnostics={:?} first={:?}",
            &result[count..],
            &result[..count.min(8)]
        );
        return -1;
    }
    if env_flag_enabled("HETGPU_SIFIVE_QK_PROFILE") {
        if !result[count+4].is_finite() || result[count+4] <= 0. ||
            result[count+5..count+11].iter().any(|v| !v.is_finite() || *v < 0.) {
            return -1;
        }
        eprintln!("PACC_QK_PROFILE dev={} qtype={} m={} n={} nb={} gate={} device_us={} activation_us={} weights_us={} xm_us={} reduce_us={} xm_calls={} local_copy_us={}",
            dev, qtype, m, n, nb, gate, result[count+4], result[count+5], result[count+6],
            result[count+7], result[count+8], result[count+9], result[count+10]);
    }
    std::ptr::copy_nonoverlapping(result.as_ptr(), out, count);
    0
}
