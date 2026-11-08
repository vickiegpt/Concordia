use crate::*;
use std::sync::Mutex;
static KERNELS: Mutex<[Option<usize>; 4]> = Mutex::new([None; 4]);

pub unsafe fn submit(
    dev: i32,
    x: *const u8,
    y: *const u8,
    out: *mut u8,
    xb: usize,
    yb: usize,
    rows: usize,
) -> i32 {
    submit_mode(dev, x, y, out, xb, yb, rows, 0)
}
pub unsafe fn cast_f32(dev: i32, x: *const u8, out: *mut u8, count: usize, half: bool) -> i32 {
    let Some(bytes) = count.checked_mul(4) else {
        return -1;
    };
    submit_mode(dev, x, x, out, bytes, 0, 1, if half { 1 } else { 0 })
}
pub unsafe fn reduce_rows(
    dev: i32,
    x: *const f32,
    out: *mut f32,
    cols: usize,
    rows: usize,
    threads: u32,
    mean: bool,
) -> i32 {
    if !matches!(threads, 32 | 128 | 512) {
        return -1;
    }
    let Some(bytes) = cols.checked_mul(4) else {
        return -1;
    };
    submit_mode(
        dev,
        x.cast(),
        x.cast(),
        out.cast(),
        bytes,
        0,
        rows,
        ((threads as u64) << 8) | if mean { 3 } else { 2 },
    )
}
unsafe fn submit_mode(
    dev: i32,
    x: *const u8,
    y: *const u8,
    out: *mut u8,
    xb: usize,
    yb: usize,
    rows: usize,
    mode: u64,
) -> i32 {
    let Some(input_bytes) = xb.checked_add(yb).and_then(|v| v.checked_mul(rows)) else {
        return -1;
    };
    let bytes = if matches!(mode & 255, 2 | 3) {
        let Some(n) = rows.checked_mul(4) else {
            return -1;
        };
        n
    } else if mode == 1 {
        input_bytes / 2
    } else {
        input_bytes
    };
    if !(0..4).contains(&dev)
        || rows == 0
        || bytes == 0
        || x.is_null()
        || y.is_null()
        || out.is_null()
        || input_bytes
            .checked_add(bytes)
            .is_none_or(|n| n > kernel_payload_capacity().saturating_sub(65536 + 8))
        || env_flag_enabled("HETGPU_SIFIVE_SKIP_JOB_WAIT")
        || env_flag_enabled("HETGPU_SIFIVE_ASSUME_SUCCESS_ON_WAIT_ERROR")
    {
        return -1;
    }
    let mut guard = match KERNELS.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    let kernel = if let Some(k) = guard[dev as usize] {
        k as *mut sifive_Kernel
    } else {
        let d = sifive_CreateDevice(dev as u32);
        let p = sifive_CreateProgram();
        if d.is_null() || p.is_null() {
            return -1;
        }
        let source = include_bytes!("concat_tile.c");
        if sifive_LoadProgramSource(
            p,
            std::ptr::null(),
            b"concat_tile.c\0".as_ptr().cast(),
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
        let k = sifive_CreateKernelOnDevice(p, d, b"pacc_concat_tile_v1\0".as_ptr().cast());
        if k.is_null() {
            return -1;
        }
        guard[dev as usize] = Some(k as usize);
        k
    };
    let mut result = vec![0u8; bytes + 8];
    if sifive_KernelClearLaunchState(kernel) != 0 {
        return -1;
    }
    for (i, value) in [
        x as u64,
        y as u64,
        result.as_mut_ptr() as u64,
        xb as u64,
        yb as u64,
        rows as u64,
        mode,
    ]
    .into_iter()
    .enumerate()
    {
        let rec = SifiveKernelArgRecord {
            kind: if i < 3 {
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
        if i < 3 {
            let size = if i == 0 {
                xb * rows
            } else if i == 1 {
                yb * rows
            } else {
                bytes + 8
            };
            if size == 0 {
                continue;
            }
            let b = SifiveKernelBufferBinding {
                arg_index: i as u32,
                addr: value,
                size: size as u64,
                flags: SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT
                    | if i == 2 {
                        SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT
                    } else {
                        0
                    },
            };
            if sifive_KernelAddBufferBinding(kernel, &b) != 0 {
                return -1;
            }
        }
    }
    let rc = sifive_LaunchKernel(kernel, 1, 1, 1, 1, 1, 1);
    if rc != 0 || result[bytes..] != [0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7] {
        eprintln!(
            "PACC layout tile failed mode={} rc={} marker={:?} sample={:?}",
            mode,
            rc,
            &result[bytes..],
            &result[..bytes.min(32)]
        );
        return -1;
    }
    std::ptr::copy_nonoverlapping(result.as_ptr(), out, bytes);
    0
}
