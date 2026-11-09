use crate::*;
use std::sync::Mutex;
static KERNELS: Mutex<[Option<usize>; 4]> = Mutex::new([None; 4]);

#[no_mangle]
pub unsafe extern "C" fn hetgpu_sifive_submit_rms_exact_on(
    dev: i32, x: *const f32, weight: *const f32, out: *mut f32,
    rows: u64, hidden: u64, eps: f32,
) -> i32 {
    if !(0..4).contains(&dev) || x.is_null() || out.is_null() || rows==0 || hidden==0 ||
        rows>4096 || hidden>32768 || !eps.is_finite() || eps<0.0 ||
        env_flag_enabled("HETGPU_SIFIVE_SKIP_JOB_WAIT") || env_flag_enabled("HETGPU_SIFIVE_ASSUME_SUCCESS_ON_WAIT_ERROR") {
        return -1;
    }
    let count=(rows*hidden) as usize;
    if (2*count+hidden as usize+16)*4>kernel_payload_capacity().min(512*1024).saturating_sub(65536) {return -1;}
    let ones=if weight.is_null(){vec![1.0f32;hidden as usize]}else{Vec::new()};
    let weight=if weight.is_null(){ones.as_ptr()}else{weight};
    let mut cache=match KERNELS.lock(){Ok(v)=>v,Err(_)=>return -1};
    let kernel=if let Some(v)=cache[dev as usize]{v as *mut sifive_Kernel}else{
        let d=sifive_CreateDevice(dev as u32);let p=sifive_CreateProgram();
        if d.is_null() || p.is_null(){return -1;}
        let source=include_str!("rms_exact.c");
        if sifive_LoadProgramSource(p,std::ptr::null(),b"rms_exact.c\0".as_ptr().cast(),source.as_ptr(),source.len() as u64,
            std::ptr::null(),std::ptr::null(),0,std::ptr::null(),0)!=0{return -1;}
        let q=sifive_CreateKernelOnDevice(p,d,b"pacc_rms_exact_v1\0".as_ptr().cast());
        if q.is_null(){return -1;}
        cache[dev as usize]=Some(q as usize);q
    };
    let mut result=vec![f32::NAN;count+16];
    result[count+1..].fill(12345.0);
    if sifive_KernelClearLaunchState(kernel)!=0{return -1;}
    for (i,value) in [x as u64,weight as u64,result.as_mut_ptr() as u64,rows,hidden,eps.to_bits() as u64].into_iter().enumerate(){
        let rec=SifiveKernelArgRecord{kind:if i<3{SIFIVE_KERNEL_ARG_KIND_POINTER}else{SIFIVE_KERNEL_ARG_KIND_SCALAR},size:8,flags:0,reserved:0,value,value_hi:0};
        if sifive_KernelPushArgRecord(kernel,&rec)!=0{return -1;}
        if i<3 {
            let binding=SifiveKernelBufferBinding{arg_index:i as u32,flags:if i<2{SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT}else{
                SIFIVE_KERNEL_ARG_FLAG_BUFFER_INPUT|SIFIVE_KERNEL_ARG_FLAG_BUFFER_OUTPUT},addr:value,
                size:match i{0=>(count*4) as u64,1=>hidden*4,_=>((count+16)*4) as u64}};
            if sifive_KernelAddBufferBinding(kernel,&binding)!=0{return -1;}
        }
    }
    if sifive_LaunchKernel(kernel,1,1,1,2,1,1)!=0 || result[count]!=0.0 ||
        result[..count].iter().any(|v|!v.is_finite()) || result[count+1..].iter().any(|v|*v!=12345.0){return -1;}
    std::ptr::copy_nonoverlapping(result.as_ptr(),out,count);
    0
}
