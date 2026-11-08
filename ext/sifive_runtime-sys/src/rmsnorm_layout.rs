#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RmsNormLayout {
    pub hidden: u64,
    pub dims: [u64; 3],
    pub x_strides: [u64; 3],
    pub w_shape: [u64; 4],
    pub w_strides: [u64; 3],
    pub eps: f32,
}

impl RmsNormLayout {
    // Element spans, not byte counts. Limit staging allocations independently
    // from the caller's advertised device memory capacity.
    pub fn spans(&self, weighted: bool) -> Option<(usize, usize, usize)> {
        if self.hidden == 0 || self.dims.contains(&0) || !self.eps.is_finite() || self.eps < 0.0 {
            return None;
        }
        let output = self.dims.iter().try_fold(self.hidden, |n, d| n.checked_mul(*d))?;
        if output > 16*1024*1024 { return None; }
        let mut input = self.hidden;
        for i in 0..3 {
            input = input.checked_add((self.dims[i]-1).checked_mul(self.x_strides[i])?)?;
        }
        let mut weight = 0;
        if weighted {
            if self.w_shape.contains(&0) { return None; }
            weight = self.hidden.min(self.w_shape[0]);
            for i in 0..3 {
                weight = weight.checked_add((self.dims[i].min(self.w_shape[i+1])-1)
                    .checked_mul(self.w_strides[i])?)?;
            }
        }
        if input > (isize::MAX as u64)/4 || weight > (isize::MAX as u64)/4 { return None; }
        Some((input as usize, weight as usize, output as usize))
    }
}

// Only layout/copy work is done here. The callback owns all normalization
// arithmetic and must return success only after device completion.
pub fn pack_and_run<F>(layout: &RmsNormLayout, x: &[f32], weight: Option<&[f32]>, mut run: F)
    -> Result<Vec<f32>, &'static str>
where F: FnMut(&[f32], Option<&[f32]>, &mut [f32], usize) -> bool {
    let (xs, ws, count) = layout.spans(weight.is_some()).ok_or("invalid RMSNorm layout")?;
    if x.len() < xs || weight.map_or(false, |w| w.len() < ws) { return Err("short RMSNorm input"); }
    let h = layout.hidden as usize;
    let mut packed = Vec::with_capacity(count);
    let mut w_offsets = Vec::with_capacity(count/h);
    for s in 0..layout.dims[2] {
        for c in 0..layout.dims[1] {
            for r in 0..layout.dims[0] {
                let at = (r*layout.x_strides[0]+c*layout.x_strides[1]+s*layout.x_strides[2]) as usize;
                packed.extend_from_slice(&x[at..at+h]);
                if weight.is_some() {
                    w_offsets.push(((r%layout.w_shape[1])*layout.w_strides[0]+
                        (c%layout.w_shape[2])*layout.w_strides[1]+
                        (s%layout.w_shape[3])*layout.w_strides[2]) as usize);
                }
            }
        }
    }
    let mut out = vec![f32::NAN; count];
    if let Some(index)=packed.iter().position(|v|!v.is_finite()) {
        eprintln!("PACC RMSNorm nonfinite input index={} hidden={} rows={} value={}",index,h,count/h,packed[index]);
        return Err("nonfinite RMSNorm input");
    }
    let shared = weight.is_none() || w_offsets.iter().all(|o| *o == w_offsets[0]);
    let mut w = vec![0.0; if weight.is_some() { h } else { 0 }];
    let step = if shared { count } else { h };
    for at in (0..count).step_by(step) {
        if let Some(src) = weight {
            let base = w_offsets[at/h];
            for col in 0..h { w[col] = src[base+col%(layout.w_shape[0] as usize)]; }
        }
        if !run(&packed[at..at+step], weight.map(|_| w.as_slice()), &mut out[at..at+step], step/h) {
            return Err("device RMSNorm failed");
        }
    }
    if let Some(index)=out.iter().position(|v|!v.is_finite()) {
        eprintln!("PACC RMSNorm nonfinite output index={} hidden={} rows={} input={} output_bits=0x{:08x} weight_finite={}",
            index,h,count/h,packed[index],out[index].to_bits(),w.iter().all(|v|v.is_finite()));
        return Err("nonfinite RMSNorm output");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn layout() -> RmsNormLayout {
        RmsNormLayout { hidden: 4, dims: [2,2,1], x_strides: [6,16,0],
            w_shape: [2,2,1,1], w_strides: [3,0,0], eps: 1e-5 }
    }
    #[test]
    fn packing_and_broadcast() {
        let l=layout(); let x: Vec<f32>=(0..26).map(|v| v as f32).collect();
        let w=[10.,20.,99.,30.,40.]; let mut calls=0;
        let out=pack_and_run(&l,&x,Some(&w),|a,b,d,rows| {
            assert_eq!(rows,1);
            let r=calls%2; let c=calls/2;
            assert_eq!(a,&x[c*16+r*6..c*16+r*6+4]);
            let expect=if r==0 { [10.,20.,10.,20.] } else { [30.,40.,30.,40.] };
            assert_eq!(b.unwrap(),expect); d.copy_from_slice(a); calls+=1; true
        }).unwrap();
        assert_eq!(calls,4); assert_eq!(out.len(),16);
    }
    #[test]
    fn shared_weight_batched_and_fail_closed() {
        let mut l=layout(); l.w_shape[1]=1;
        let x=vec![1.;26]; let w=[2.,3.]; let mut calls=0;
        assert!(pack_and_run(&l,&x,Some(&w),|_,_,out,n| { calls+=1; assert_eq!(n,4); out.fill(1.); true }).is_ok());
        assert_eq!(calls,1);
        assert!(pack_and_run(&l,&x,Some(&w),|_,_,_,_| false).is_err());
        assert!(pack_and_run(&l,&x,Some(&w),|_,_,_,_| true).is_err());
        assert!(pack_and_run(&l,&x[..2],Some(&w),|_,_,_,_| true).is_err());
        l.x_strides[1]=u64::MAX; assert!(l.spans(true).is_none());
        l=layout(); l.w_shape[0]=0; assert!(l.spans(true).is_none());
        l=layout(); l.dims[0]=u64::MAX; assert!(l.spans(false).is_none());
    }
}
