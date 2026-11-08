use super::*;

pub(super) unsafe fn launch(
    name: &str,
    grid: [u32; 3],
    params: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    use cuda_types::cuda::CUerror;
    let run = (|| -> Option<bool> {
        // Current GGML ABI has a separate state pointer and snapshot arguments.
        if !name.ends_with("S3_fli") {
            return None;
        }
        let (s, kda) = sifive_parse_gated_delta_net_template(name)?;
        let keep = name.split("ELb").nth(2)?.starts_with('1');
        let s = usize::try_from(s).ok()?;
        let read = |i| usize::try_from(read_param_i64(params, i)?).ok();
        let h = read(8)?;
        let n = read(9)?;
        let seqs = read(10)?;
        let qst = [read(11)?, read(12)?, read(13)?];
        let vst = [read(14)?, read(15)?, read(16)?];
        let bst = [read(17)?, read(18)?, read(19)?];
        let qheads = read_param_uint3_z(params, 20)? as usize;
        let seqratio = read_param_uint3_z(params, 21)? as usize;
        let scale = read_param_f32(params, 22)?;
        let slotstride = read(23)?;
        let capacity = usize::try_from(read_param_i32(params, 24)?).ok()?;
        let slots = if keep { capacity.min(n) } else { 1 };
        if h == 0
            || h > 256
            || seqs == 0
            || seqs > 64
            || qheads == 0
            || seqratio == 0
            || !scale.is_finite()
            || grid != [h as u32, seqs as u32, s.div_ceil(4) as u32]
        {
            return None;
        }
        let (ins, outs) = sifive_runtime_sys::gated_delta::lengths(s, n, kda, slots)?;
        let mut ptrs = [0u64; 8];
        for i in 0..8 {
            ptrs[i] = read_param_u64(params, i)?;
        }
        let span = |heads: usize, sequences: usize, stride: [usize; 3], tail: usize| {
            (sequences - 1)
                .checked_mul(stride[2])?
                .checked_add((n - 1).checked_mul(stride[1])?)?
                .checked_add((heads - 1).checked_mul(stride[0])?)?
                .checked_add(tail)
        };
        let qs = span(h.min(qheads), seqs.div_ceil(seqratio), qst, s)?;
        let vs = span(h, seqs, vst, s)?;
        let bs = span(h, seqs, bst, 1)?;
        let gs = if kda { bs.checked_mul(s)? } else { bs };
        let states = seqs.checked_mul(h)?.checked_mul(s)?.checked_mul(s)?;
        let ds = seqs.checked_mul(n)?.checked_mul(h)?.checked_mul(s)?;
        if keep && slotstride < states {
            return None;
        }
        let stateout = if keep {
            (slots - 1).checked_mul(slotstride)?.checked_add(states)?
        } else {
            states
        };
        for (i, elems) in [qs, qs, vs, gs, bs, states, ds, stateout]
            .into_iter()
            .enumerate()
        {
            if !sifive_host_or_cuda_alloc_has_bytes(ptrs[i], elems.checked_mul(4)?, i >= 6) {
                return None;
            }
        }
        let mut input = vec![0f32; ins];
        let mut output = vec![f32::NAN; outs];
        for seq in 0..seqs {
            for head in 0..h {
                let mut at = 0;
                for (which, strides, srcseq, srchead, width) in [
                    (0, qst, seq / seqratio, head % qheads, s),
                    (1, qst, seq / seqratio, head % qheads, s),
                    (2, vst, seq, head, s),
                    (3, bst, seq, head, if kda { s } else { 1 }),
                    (4, bst, seq, head, 1),
                ] {
                    for t in 0..n {
                        let mut offset =
                            srcseq * strides[2] + t * strides[1] + srchead * strides[0];
                        if which == 3 && kda {
                            offset *= s;
                        }
                        input[at..at + width].copy_from_slice(std::slice::from_raw_parts(
                            (ptrs[which] as *const f32).add(offset),
                            width,
                        ));
                        at += width;
                    }
                }
                let statebase = (seq * h + head) * s * s;
                input[at..].copy_from_slice(std::slice::from_raw_parts(
                    (ptrs[5] as *const f32).add(statebase),
                    s * s,
                ));
                let rc = sifive_runtime_sys::gated_delta::hetgpu_sifive_submit_gdn_head_f32(
                    current_sifive_device_id_or_zero(),
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    s as u32,
                    n as u32,
                    kda as u32,
                    keep as u32,
                    slots as u32,
                    scale,
                );
                if rc != 0 {
                    return Some(false);
                }
                for t in 0..n {
                    std::ptr::copy_nonoverlapping(
                        output.as_ptr().add(t * s),
                        (ptrs[6] as *mut f32).add(((seq * n + t) * h + head) * s),
                        s,
                    );
                }
                for slot in 0..slots {
                    std::ptr::copy_nonoverlapping(
                        output.as_ptr().add(n * s + slot * s * s),
                        (ptrs[7] as *mut f32).add(statebase + slot * slotstride),
                        s * s,
                    );
                }
            }
        }
        eprintln!("PACC GDN device-core heads={} seqs={} tokens={} width={} snapshots={} completion=verified",h,seqs,n,s,slots);
        Some(true)
    })();
    match run {
        Some(true) => Ok(()),
        Some(false) => Err(CUerror::UNKNOWN),
        None => Err(CUerror::NOT_SUPPORTED),
    }
}
